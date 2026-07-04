from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

MNE_HOME = Path(tempfile.gettempdir()) / "spdtransformer_mne"
MNE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(MNE_HOME))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne

from src.datasets.PhysioNetMI_preprocess import (
    build_dataset,
    encode_labels,
    normalize_baseline_correction_mode,
    pick_raw_channels,
    preprocess_eegnet_author,
    preprocess_spd,
    segment_epochs,
)
from src.training.shared_split import load_or_create_split_indices


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_grid.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def normalize_filter_bank(filter_bank: Any) -> list[list[float]]:
    if not isinstance(filter_bank, list) or not filter_bank:
        raise ValueError("data.filter_bank must be a non-empty list.")

    normalized = []
    for band in filter_bank:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError(
                "Each filter bank item must be [low_freq, high_freq], "
                f"got {band!r}."
            )
        normalized.append([float(band[0]), float(band[1])])
    return normalized


def is_filter_bank_value(key: str, value: Any) -> bool:
    if key != "filter_bank" or not isinstance(value, list):
        return False
    return bool(value) and all(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and all(isinstance(number, (int, float)) for number in item)
        for item in value
    )


def grid_values(key: str, value: Any) -> list[Any]:
    if is_filter_bank_value(key, value):
        return [normalize_filter_bank(value)]
    if isinstance(value, list):
        return value
    return [value]


def expand_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    if not section:
        return [{}]
    keys = list(section)
    value_lists = [grid_values(key, section[key]) for key in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def expand_data_training_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_grid(config.get("data", {}))
    training_grid = expand_grid(config.get("training", {}))
    output_cfg = deepcopy(config.get("output", {}))
    experiments = []
    for data_cfg, training_cfg in itertools.product(data_grid, training_grid):
        experiments.append(
            {
                "data": deepcopy(data_cfg),
                "training": deepcopy(training_cfg),
                "output": deepcopy(output_cfg),
            }
        )
    return experiments


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def parse_subjects(subjects: Any) -> list[int] | None:
    if subjects is None:
        return None
    if isinstance(subjects, str) and subjects.strip() == "":
        return None
    if isinstance(subjects, int):
        return [subjects]
    if isinstance(subjects, (list, tuple)):
        return sorted({int(subject) for subject in subjects})

    parsed = []
    for part in str(subjects).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise ValueError(f"Invalid subject range: {part}")
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(int(part))
    return sorted(set(parsed))


def parse_task_types(task_types: Any) -> tuple[str, ...]:
    if task_types is None:
        return ("unilateral_fist", "both")
    if isinstance(task_types, str):
        return tuple(part.strip() for part in task_types.split(",") if part.strip())
    return tuple(str(part).strip() for part in task_types if str(part).strip())


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_split_file(split_file: Any) -> Path | None:
    if split_file in {None, ""}:
        return None
    path = Path(str(split_file))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_split_indices(
    y: np.ndarray,
    training_cfg: dict[str, Any],
    subject_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return load_or_create_split_indices(
        y=y,
        test_size=float(training_cfg.get("test_size", 0.15)),
        val_size=float(training_cfg.get("val_size", 0.15)),
        seed=int(training_cfg.get("seed", 42)),
        split_file=resolve_split_file(training_cfg.get("split_file")),
        subjects=subject_labels,
        allow_subject_overlap=parse_bool(
            training_cfg.get("allow_subject_overlap", True),
            default=True,
        ),
    )


def load_spd_like_train(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    x_spd, y, class_names, subject_labels = preprocess_spd(
        filter_bank=filter_bank,
        root_dir=str(
            data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")
        ),
        subjects=parse_subjects(data_cfg.get("subjects")),
        channels=data_cfg.get("channels"),
        estimator=str(data_cfg.get("estimator", "lwf")),
        eps=float(data_cfg.get("eps", 1e-6)),
        sfreq=float(data_cfg.get("sfreq", 160)),
        segment_duration=float(data_cfg.get("segment_duration", 1.0)),
        stride_duration=data_cfg.get("stride_duration", 0.5),
        imaged=parse_bool(data_cfg.get("imaged", True), default=True),
        executed=parse_bool(data_cfg.get("executed", False), default=False),
        task_types=parse_task_types(data_cfg.get("task_types")),
        reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
        baseline_correction=data_cfg.get("baseline_correction"),
        baseline_window=data_cfg.get("baseline_window"),
        epoch_tmin=float(data_cfg.get("epoch_tmin", -2.0)),
        epoch_tmax=float(data_cfg.get("epoch_tmax", 4.0)),
        use_ica=parse_bool(data_cfg.get("use_ica", False), default=False),
        ica_n_components=data_cfg.get("ica_n_components", 20),
        ica_random_state=int(data_cfg.get("ica_random_state", 42)),
        ica_eog_channels=data_cfg.get("ica_eog_channels"),
        use_autoreject=parse_bool(
            data_cfg.get("use_autoreject", False),
            default=False,
        ),
        autoreject_random_state=int(data_cfg.get("autoreject_random_state", 42)),
        autoreject_n_jobs=int(data_cfg.get("autoreject_n_jobs", 1)),
        autoreject_cv=int(data_cfg.get("autoreject_cv", 10)),
        return_subjects=True,
    )
    if not np.isfinite(x_spd).all():
        raise ValueError("preprocess_spd returned NaN or Inf values.")
    return (
        x_spd.astype(np.float64),
        y.astype(np.int64),
        np.asarray(subject_labels, dtype=np.str_),
        list(class_names),
    )


def load_segmented_epochs_like_train(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[list[float]]]:
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    root_dir = str(data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0"))
    subjects = parse_subjects(data_cfg.get("subjects"))
    channels = data_cfg.get("channels")
    sfreq = float(data_cfg.get("sfreq", 160))
    segment_duration = float(data_cfg.get("segment_duration", 1.0))
    stride_duration = data_cfg.get("stride_duration", 0.5)
    task_types = parse_task_types(data_cfg.get("task_types"))
    imaged = parse_bool(data_cfg.get("imaged", True), default=True)
    executed = parse_bool(data_cfg.get("executed", False), default=False)
    use_ica = parse_bool(data_cfg.get("use_ica", False), default=False)
    use_autoreject = parse_bool(data_cfg.get("use_autoreject", False), default=False)
    if normalize_baseline_correction_mode(data_cfg.get("baseline_correction")) is not None:
        print(
            "Ignoring SPD covariance baseline_correction for raw segmented "
            "epoch baseline. It is only applied in preprocess_spd()."
        )

    bands = []
    labels = None
    subject_labels = None
    for low_freq, high_freq in filter_bank:
        dataset = build_dataset(
            root_dir,
            tmin=float(data_cfg.get("epoch_tmin", -2.0)),
            tmax=float(data_cfg.get("epoch_tmax", 4.0)),
            subjects=subjects,
            imaged=imaged,
            executed=executed,
            task_types=task_types,
            low_freq=low_freq,
            high_freq=high_freq,
            channels=channels,
            reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
            use_ica=use_ica,
            ica_n_components=data_cfg.get("ica_n_components", 20),
            ica_random_state=int(data_cfg.get("ica_random_state", 42)),
            ica_eog_channels=data_cfg.get("ica_eog_channels"),
            use_autoreject=use_autoreject,
            autoreject_random_state=int(data_cfg.get("autoreject_random_state", 42)),
            autoreject_n_jobs=int(data_cfg.get("autoreject_n_jobs", 1)),
            autoreject_cv=int(data_cfg.get("autoreject_cv", 10)),
        )
        x_band = dataset["X"]
        if labels is None:
            labels = dataset["y"]
            subject_labels = dataset["subject"]
        else:
            if not np.array_equal(labels, dataset["y"]):
                raise RuntimeError("Labels changed across frequency bands.")
            if not np.array_equal(subject_labels, dataset["subject"]):
                raise RuntimeError("Subject order changed across frequency bands.")

        x_segments = segment_epochs(
            x_band,
            sfreq=sfreq,
            segment_duration=segment_duration,
            stride_duration=stride_duration,
        )
        bands.append(x_segments.astype(np.float32))

    if labels is None:
        raise RuntimeError("No data was loaded.")
    y, class_names = encode_labels(labels)
    x = np.stack(bands, axis=2)
    # Shape: (n_trials, n_segments, n_frequency_bands, n_channels, n_samples)
    if not np.isfinite(x).all():
        raise ValueError("Segmented EEG dataset contains NaN or Inf values.")
    return (
        x,
        y.astype(np.int64),
        np.asarray(subject_labels, dtype=np.str_),
        list(class_names),
        filter_bank,
    )


def load_eegnet_author_data(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x, y, class_names, subject_labels = preprocess_eegnet_author(
        root_dir=str(
            data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")
        ),
        subjects=parse_subjects(data_cfg.get("subjects")),
        n_classes=int(data_cfg.get("eegnet_num_classes", 2)),
        excluded_subjects=parse_subjects(
            data_cfg.get("eegnet_excluded_subjects", "88,92,100,104")
        ),
        T=float(data_cfg.get("eegnet_T", data_cfg.get("epoch_tmax", 3.0))),
        n_ds=int(data_cfg.get("eegnet_downsample", 1)),
        n_ch=int(data_cfg.get("eegnet_n_channels", 64)),
        normalization=int(data_cfg.get("eegnet_normalization", 0)),
        sfreq=float(data_cfg.get("sfreq", 160)),
        scale_to_uv=parse_bool(data_cfg.get("eegnet_scale_to_uv", True), default=True),
        random_state=int(data_cfg.get("eegnet_random_state", 7)),
        return_subjects=True,
    )
    return (
        x.astype(np.float32, copy=False),
        y.astype(np.int64, copy=False),
        np.asarray(subject_labels, dtype=np.str_),
        list(class_names),
    )


def matrix_log(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = 0.5 * (x + np.swapaxes(x, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(x)
    log_eigvals = np.log(np.clip(eigvals, eps, None))
    return (eigvecs * log_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)


def matrix_exp(x: np.ndarray) -> np.ndarray:
    x = 0.5 * (x + np.swapaxes(x, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(x)
    y = (eigvecs * np.exp(eigvals)[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    return 0.5 * (y + np.swapaxes(y, -1, -2))


def log_euclidean_token_mean(x_spd: np.ndarray) -> np.ndarray:
    # Input: (n_trials, segment, frequency, channels, channels)
    log_x = matrix_log(x_spd)
    pooled_log = log_x.mean(axis=(1, 2))
    return matrix_exp(pooled_log).astype(np.float64)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def split_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for split_name, split_idx in indices.items():
        row = {
            "split": split_name,
            "n_samples": int(len(split_idx)),
        }
        row.update(compute_metrics(y[split_idx], predictions[split_name]))
        rows.append(row)
    return rows


def save_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
