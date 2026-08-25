"""Run-labelled SPD wrapper for PhysioNet-MI and BCI Competition IV-2a.

All signal processing and SPD construction are delegated to the project's
canonical dataset preprocessors. This file canonicalizes configuration,
requests physical run provenance, and caches the aligned arrays.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.BCICompetitionIV2a_preprocess import (
    normalize_bci_iv_2a_events,
    normalize_bci_iv_2a_sessions,
    preprocess_bci_iv_2a_spd,
)
from src.datasets.PhysioNetMI_preprocess import (
    normalize_bool,
    normalize_brain_region_mode,
    normalize_float_dtype,
    preprocess_spd,
)
from src.training.config_grid import normalize_data_time_config, normalize_filter_bank


CACHE_VERSION = 4


def normalize_dataset_name(value: Any) -> str:
    normalized = (
        str(value or "physionet_mi").strip().lower().replace("-", "_")
    )
    aliases = {
        "physionet": "physionet_mi",
        "physionet_mi": "physionet_mi",
        "physionetmi": "physionet_mi",
        "eegbci": "physionet_mi",
        "bci_iv_2a": "bnci2014_001",
        "bci_competition_iv_2a": "bnci2014_001",
        "bciciv_2a": "bnci2014_001",
        "bnci2014_001": "bnci2014_001",
        "bnci2014001": "bnci2014_001",
    }
    if normalized not in aliases:
        raise ValueError(
            "Run-labelled preprocessing supports physionet_mi and "
            "bnci2014_001 only."
        )
    return aliases[normalized]


def parse_subjects(value: Any) -> list[int] | None:
    """Parse ``1``, ``1-10``, ``1-3,8``, or a sequence of those values."""

    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        subjects: list[int] = []
        for item in value:
            parsed = parse_subjects(item)
            if parsed:
                subjects.extend(parsed)
        return sorted(set(subjects)) or None
    text = str(value).strip()
    if text.lower() in {"", "all", "none", "null"}:
        return None
    subjects: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip().upper().removeprefix("S")
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid subject range: {raw_part!r}.")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(part))
    return sorted(subjects) or None


def parse_task_types(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("unilateral_fist", "both")
    parts = value.split(",") if isinstance(value, str) else value
    result = tuple(str(part).strip() for part in parts if str(part).strip())
    invalid = sorted(set(result) - {"unilateral_fist", "both"})
    if invalid:
        raise ValueError(f"Unsupported task_types: {invalid}.")
    if not result:
        raise ValueError("data.task_types cannot be empty.")
    return result


def selected_run_ids(
    *,
    imaged: bool,
    executed: bool,
    task_types: tuple[str, ...],
) -> list[int]:
    runs: list[int] = []
    if imaged:
        if "unilateral_fist" in task_types:
            runs.extend([4, 8, 12])
        if "both" in task_types:
            runs.extend([6, 10, 14])
    if executed:
        if "unilateral_fist" in task_types:
            runs.extend([3, 7, 11])
        if "both" in task_types:
            runs.extend([5, 9, 13])
    return sorted(set(runs))


def normalize_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = normalize_data_time_config(deepcopy(data_cfg))
    cfg["dataset"] = normalize_dataset_name(cfg.get("dataset"))
    cfg["filter_bank"] = normalize_filter_bank(cfg["filter_bank"])
    cfg["subjects"] = parse_subjects(cfg.get("subjects"))
    if cfg["dataset"] == "physionet_mi":
        cfg["task_types"] = list(parse_task_types(cfg.get("task_types")))
        cfg["imaged"] = normalize_bool(cfg.get("imaged", True), default=True)
        cfg["executed"] = normalize_bool(cfg.get("executed", False), default=False)
        if not cfg["imaged"] and not cfg["executed"]:
            raise ValueError(
                "At least one of data.imaged/data.executed must be true."
            )
    else:
        cfg["events"] = normalize_bci_iv_2a_events(
            cfg.get("events", cfg.get("bci_iv_2a_events"))
        )
        cfg["sessions"] = normalize_bci_iv_2a_sessions(
            cfg.get("sessions", cfg.get("bci_iv_2a_sessions"))
        )
    scale = cfg.get("covariance_signal_scale", "auto")
    if scale is None or str(scale).strip().lower() in {"", "auto"}:
        scale = 1.0e6 if cfg["dataset"] == "physionet_mi" else 1.0
    cfg["covariance_signal_scale"] = float(scale)
    cfg["use_ica"] = normalize_bool(cfg.get("use_ica", False), default=False)
    cfg["use_autoreject"] = normalize_bool(
        cfg.get("use_autoreject", False), default=False
    )
    cfg["autoreject_force_rebuild"] = normalize_bool(
        cfg.get("autoreject_force_rebuild", False), default=False
    )
    cfg["replace_covariance_diagonal_with_raw_energy"] = normalize_bool(
        cfg.get("replace_covariance_diagonal_with_raw_energy", False),
        default=False,
    )
    cfg["brain_region_mode"] = normalize_brain_region_mode(
        cfg.get("brain_region_mode")
    )
    return cfg


def preprocess_spd_with_runs(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Call the selected canonical preprocessor with run provenance enabled."""

    cfg = normalize_data_config(data_cfg)
    epoch_tmin, epoch_tmax = cfg["epoch_slice"]
    segment_duration, stride_duration = cfg["segment_slice"]
    common = {
        "filter_bank": cfg["filter_bank"],
        "root_dir": str(cfg.get("root_dir", "data")),
        "subjects": cfg.get("subjects"),
        "channels": cfg.get("channels"),
        "estimator": str(cfg.get("estimator", "lwf")),
        "sfreq": float(
            cfg.get(
                "sfreq",
                160.0 if cfg["dataset"] == "physionet_mi" else 250.0,
            )
        ),
        "eps": float(cfg.get("eps", 1e-6)),
        "segment_duration": float(segment_duration),
        "stride_duration": stride_duration,
        "reject_threshold_uv": cfg.get("reject_threshold_uv"),
        "baseline_correction": cfg.get("baseline_correction"),
        "baseline_window": cfg.get("baseline_window"),
        "epoch_tmin": float(epoch_tmin),
        "epoch_tmax": float(epoch_tmax),
        "use_ica": bool(cfg["use_ica"]),
        "use_autoreject": bool(cfg["use_autoreject"]),
        "autoreject_random_state": int(cfg.get("autoreject_random_state", 42)),
        "autoreject_n_jobs": int(cfg.get("autoreject_n_jobs", 1)),
        "autoreject_cv": int(cfg.get("autoreject_cv", 10)),
        "return_subjects": True,
        "return_runs": True,
        "covariance_signal_scale": float(cfg["covariance_signal_scale"]),
        "replace_covariance_diagonal_with_raw_energy": bool(
            cfg["replace_covariance_diagonal_with_raw_energy"]
        ),
        "brain_region_mode": cfg.get("brain_region_mode"),
        "output_dtype": cfg.get("covariance_output_dtype", "float32"),
    }
    if cfg["dataset"] == "physionet_mi":
        common["root_dir"] = str(
            cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")
        )
        x, y, class_names, subject_labels, run_labels = preprocess_spd(
            **common,
            imaged=bool(cfg["imaged"]),
            executed=bool(cfg["executed"]),
            task_types=tuple(cfg["task_types"]),
            ica_n_components=cfg.get("ica_n_components", 20),
            ica_random_state=int(cfg.get("ica_random_state", 42)),
            ica_eog_channels=cfg.get("ica_eog_channels"),
            autoreject_cache_dir=cfg.get("autoreject_cache_dir"),
            autoreject_force_rebuild=bool(cfg["autoreject_force_rebuild"]),
        )
    else:
        x, y, class_names, subject_labels, run_labels = preprocess_bci_iv_2a_spd(
            **common,
            events=cfg["events"],
            sessions=cfg["sessions"],
            moabb_accept_terms=normalize_bool(
                cfg.get("moabb_accept_terms", True), default=True
            ),
            moabb_force_update=normalize_bool(
                cfg.get("moabb_force_update", False), default=False
            ),
        )
    x = x.astype(
        normalize_float_dtype(cfg.get("covariance_output_dtype", "float32")),
        copy=False,
    )
    subject_labels = np.asarray(subject_labels, dtype=np.str_)
    run_labels = np.asarray(run_labels, dtype=np.int16)
    if not np.isfinite(x).all():
        raise ValueError("Preprocessed SPD data contains NaN or Inf values.")
    if not (len(x) == len(y) == len(subject_labels) == len(run_labels)):
        raise RuntimeError("Trial, class, subject, and run arrays are misaligned.")
    return (
        x,
        np.asarray(y, dtype=np.int64),
        subject_labels,
        run_labels,
        [str(value) for value in class_names],
    )


def _cache_path(cache_dir: Path, cfg: dict[str, Any]) -> Path:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{cfg['dataset']}_loro_spd_{digest}.npz"


def load_or_preprocess_spd_with_runs(
    data_cfg: dict[str, Any],
    cache_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cfg = normalize_data_config(data_cfg)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, cfg)
    if path.exists():
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata.get("cache_version") == CACHE_VERSION and metadata.get(
                "data_config"
            ) == cfg:
                print(f"Loaded canonical run-labelled SPD cache: {path}")
                return (
                    np.asarray(payload["x"]),
                    np.asarray(payload["y"], dtype=np.int64),
                    np.asarray(payload["subject_labels"], dtype=np.str_),
                    np.asarray(payload["run_labels"], dtype=np.int16),
                    [str(value) for value in payload["class_names"].tolist()],
                )

    result = preprocess_spd_with_runs(cfg)
    x, y, subject_labels, run_labels, class_names = result
    metadata = {"cache_version": CACHE_VERSION, "data_config": cfg}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            x=x,
            y=y,
            subject_labels=subject_labels,
            run_labels=run_labels,
            class_names=np.asarray(class_names, dtype=np.str_),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, default=str), dtype=np.str_
            ),
        )
    temporary.replace(path)
    print(f"Saved canonical run-labelled SPD cache: {path}")
    return result
