"""PhysioNet-MI SPD preprocessing with subject and EDF-run provenance.

The regular project preprocessor intentionally returns only subject labels.
Leave-one-run-out adaptation additionally needs the physical EDF run for every
trial, so this module keeps subject/run labels aligned through filtering,
artifact rejection, segmentation, and covariance construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import mne
import numpy as np
from pyriemann.estimation import Covariances

from src.datasets.PhysioNetMI_preprocess import (
    apply_ica_artifact_removal,
    autoreject_keep_mask,
    baseline_correct_covariances,
    batch_spd_diagonal_mean_normalize,
    encode_labels,
    epoch_peak_to_peak_uv,
    filter_raw_band,
    map_event_to_label,
    normalize_baseline_correction_mode,
    normalize_bool,
    normalize_brain_region_mode,
    normalize_channels,
    normalize_float_dtype,
    pick_raw_channels,
    regularize_spd,
    reject_bad_epochs,
    replace_covariance_diagonal_with_segment_energy,
    resolve_brain_region_indices,
    segment_epochs,
    set_standard_eeg_montage,
    trace_normalize,
)
from src.training.config_grid import normalize_data_time_config, normalize_filter_bank


CACHE_VERSION = 2


def parse_subjects(value: Any) -> list[int] | None:
    """Parse ``1``, ``1-10``, ``1-3,8``, or a sequence of those values."""

    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, (list, tuple, set)):
        result: list[int] = []
        for item in value:
            parsed = parse_subjects(item)
            if parsed:
                result.extend(parsed)
        return sorted(set(result)) or None

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
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = [str(part).strip() for part in value]
    result = tuple(part for part in values if part)
    if not result:
        raise ValueError("data.task_types cannot be empty.")
    invalid = sorted(set(result) - {"unilateral_fist", "both"})
    if invalid:
        raise ValueError(f"Unsupported task_types: {invalid}.")
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
    dataset = str(cfg.get("dataset", "physionet_mi")).strip().lower().replace("-", "_")
    if dataset not in {"physionet", "physionet_mi", "physionetmi", "eegbci"}:
        raise ValueError("This preprocessor supports PhysioNet-MI only.")
    cfg["dataset"] = "physionet_mi"
    cfg["filter_bank"] = normalize_filter_bank(cfg["filter_bank"])
    cfg["subjects"] = parse_subjects(cfg.get("subjects"))
    cfg["task_types"] = list(parse_task_types(cfg.get("task_types")))
    cfg["imaged"] = normalize_bool(cfg.get("imaged", True), default=True)
    cfg["executed"] = normalize_bool(cfg.get("executed", False), default=False)
    if not cfg["imaged"] and not cfg["executed"]:
        raise ValueError("At least one of data.imaged/data.executed must be true.")
    scale = cfg.get("covariance_signal_scale", "auto")
    if scale is None or str(scale).strip().lower() in {"", "auto"}:
        scale = 1.0e6
    cfg["covariance_signal_scale"] = float(scale)
    cfg["use_ica"] = normalize_bool(cfg.get("use_ica", False), default=False)
    cfg["use_autoreject"] = normalize_bool(
        cfg.get("use_autoreject", False), default=False
    )
    cfg["replace_covariance_diagonal_with_raw_energy"] = normalize_bool(
        cfg.get("replace_covariance_diagonal_with_raw_energy", False),
        default=False,
    )
    cfg["brain_region_mode"] = normalize_brain_region_mode(
        cfg.get("brain_region_mode")
    )
    return cfg


def _subject_directories(root_dir: Path, subjects: list[int] | None) -> list[Path]:
    available = {path.name.upper(): path for path in root_dir.glob("S*") if path.is_dir()}
    if subjects is None:
        selected = [available[name] for name in sorted(available)]
    else:
        requested = [f"S{subject:03d}" for subject in subjects]
        missing = [name for name in requested if name not in available]
        if missing:
            raise ValueError(
                f"Subject directories not found under {root_dir}: {', '.join(missing)}"
            )
        selected = [available[name] for name in requested]
    if not selected:
        raise ValueError(f"No subject directories found under {root_dir}.")
    return selected


def _load_run_epochs(
    edf_path: Path,
    *,
    run_id: int,
    low_freq: float,
    high_freq: float,
    epoch_tmin: float,
    epoch_tmax: float,
    expected_sfreq: float,
    channels: Any,
    reject_threshold_uv: float | None,
    use_ica: bool,
    ica_n_components: int | None,
    ica_random_state: int,
    ica_eog_channels: Any,
    use_autoreject: bool,
    autoreject_random_state: int,
    autoreject_n_jobs: int,
    autoreject_cv: int,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.set_eeg_reference("average", projection=False)
    raw = pick_raw_channels(raw, channels=channels)
    raw = set_standard_eeg_montage(raw)
    raw.filter(l_freq=0.5, h_freq=None, verbose=False)
    raw.filter(l_freq=None, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=60.0, verbose=False)
    if use_ica:
        raw = apply_ica_artifact_removal(
            raw,
            n_components=ica_n_components,
            random_state=ica_random_state,
            eog_channels=ica_eog_channels,
        )

    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, expected_sfreq, atol=1e-5):
        print(
            f"Sampling rate mismatch: expected {expected_sfreq}, got {sfreq}; "
            f"skipping {edf_path.name}."
        )
        return None

    artifact_data = raw.get_data()
    band_raw = filter_raw_band(raw.copy(), low_freq=low_freq, high_freq=high_freq)
    band_data = band_raw.get_data()
    epochs: list[np.ndarray] = []
    artifact_epochs: list[np.ndarray] = []
    labels: list[str] = []
    for onset, description in zip(
        band_raw.annotations.onset,
        band_raw.annotations.description,
    ):
        if description not in {"T1", "T2"}:
            continue
        label = map_event_to_label(run_id, description)
        if label is None:
            continue
        center = int(round(float(onset) * sfreq))
        start = int(round(center + epoch_tmin * sfreq))
        end = int(round(center + epoch_tmax * sfreq))
        if start < 0 or end > band_data.shape[1]:
            continue
        epochs.append(band_data[:, start:end])
        artifact_epochs.append(artifact_data[:, start:end])
        labels.append(label)
    if not epochs:
        return None

    x = np.asarray(epochs)
    artifact_x = np.asarray(artifact_epochs)
    y = np.asarray(labels, dtype=np.str_)
    if use_autoreject:
        keep = autoreject_keep_mask(
            artifact_x,
            info=raw.info,
            tmin=epoch_tmin,
            random_state=autoreject_random_state,
            n_jobs=autoreject_n_jobs,
            cv=autoreject_cv,
        )
        x, artifact_x, y = x[keep], artifact_x[keep], y[keep]
    _, y, keep = reject_bad_epochs(
        artifact_x,
        y,
        threshold_uv=reject_threshold_uv,
    )
    if reject_threshold_uv is not None and int((~keep).sum()):
        ptp = epoch_peak_to_peak_uv(artifact_x)
        print(
            f"{edf_path.name}: rejected {int((~keep).sum())} trial(s), "
            f"peak-to-peak max={float(ptp.max()):.1f} uV."
        )
    x = x[keep]
    if not len(x):
        return None
    return x, y, list(raw.ch_names)


def _build_band_dataset(
    cfg: dict[str, Any],
    *,
    low_freq: float,
    high_freq: float,
) -> dict[str, Any]:
    root_dir = Path(str(cfg["root_dir"]))
    subject_dirs = _subject_directories(root_dir, cfg.get("subjects"))
    task_types = tuple(cfg["task_types"])
    run_ids = selected_run_ids(
        imaged=bool(cfg["imaged"]),
        executed=bool(cfg["executed"]),
        task_types=task_types,
    )
    if not run_ids:
        raise ValueError("The selected task settings contain no PhysioNet runs.")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    channel_names: list[str] | None = None
    for subject_dir in subject_dirs:
        subject_trials = 0
        for run_id in run_ids:
            edf_path = subject_dir / f"{subject_dir.name}R{run_id:02d}.edf"
            if not edf_path.exists():
                print(f"Missing run, skipping: {edf_path}")
                continue
            loaded = _load_run_epochs(
                edf_path,
                run_id=run_id,
                low_freq=low_freq,
                high_freq=high_freq,
                epoch_tmin=float(cfg["epoch_slice"][0]),
                epoch_tmax=float(cfg["epoch_slice"][1]),
                expected_sfreq=float(cfg.get("sfreq", 160.0)),
                channels=cfg.get("channels"),
                reject_threshold_uv=cfg.get("reject_threshold_uv"),
                use_ica=bool(cfg["use_ica"]),
                ica_n_components=cfg.get("ica_n_components", 20),
                ica_random_state=int(cfg.get("ica_random_state", 42)),
                ica_eog_channels=cfg.get("ica_eog_channels"),
                use_autoreject=bool(cfg["use_autoreject"]),
                autoreject_random_state=int(cfg.get("autoreject_random_state", 42)),
                autoreject_n_jobs=int(cfg.get("autoreject_n_jobs", 1)),
                autoreject_cv=int(cfg.get("autoreject_cv", 10)),
            )
            if loaded is None:
                continue
            run_x, run_y, run_channels = loaded
            if channel_names is None:
                channel_names = run_channels
            elif run_channels != channel_names:
                raise ValueError(
                    f"Channel order changed in {edf_path}: expected "
                    f"{channel_names}, got {run_channels}."
                )
            n_trials = len(run_y)
            x_parts.append(run_x)
            y_parts.append(run_y)
            subject_parts.append(
                np.full(
                    n_trials,
                    subject_dir.name,
                    dtype=f"<U{len(subject_dir.name)}",
                )
            )
            run_parts.append(np.full(n_trials, run_id, dtype=np.int16))
            subject_trials += n_trials
        print(
            f"{subject_dir.name}: retained {subject_trials} trial(s) "
            f"for {low_freq:g}-{high_freq:g} Hz."
        )
    if not x_parts or channel_names is None:
        raise RuntimeError("No PhysioNet trials remained after preprocessing.")
    return {
        "X": np.concatenate(x_parts),
        "y": np.concatenate(y_parts),
        "subject": np.concatenate(subject_parts),
        "run": np.concatenate(run_parts),
        "ch_names": channel_names,
    }


def preprocess_spd_with_runs(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return SPD tensors, class labels, subject labels, and physical run IDs."""

    cfg = normalize_data_config(data_cfg)
    frequencies: list[np.ndarray] = []
    labels: np.ndarray | None = None
    subject_labels: np.ndarray | None = None
    run_labels: np.ndarray | None = None
    channel_names: list[str] | None = None
    brain_region_indices: np.ndarray | None = None
    baseline_mode = normalize_baseline_correction_mode(
        cfg.get("baseline_correction")
    )
    output_dtype = normalize_float_dtype(
        cfg.get("covariance_output_dtype", "float32")
    )
    segment_duration = float(cfg["segment_slice"][0])
    stride_duration = cfg["segment_slice"][1]
    if stride_duration is not None:
        stride_duration = float(stride_duration)
    eps = float(cfg.get("eps", 1e-6))

    for band_index, (low_freq, high_freq) in enumerate(cfg["filter_bank"]):
        dataset = _build_band_dataset(
            cfg,
            low_freq=float(low_freq),
            high_freq=float(high_freq),
        )
        if channel_names is None:
            channel_names = list(dataset["ch_names"])
            _, brain_region_indices = resolve_brain_region_indices(
                channel_names,
                cfg.get("brain_region_mode"),
            )
        elif list(dataset["ch_names"]) != channel_names:
            raise RuntimeError("Channel order changed across frequency bands.")

        for name, current, expected in (
            ("class", dataset["y"], labels),
            ("subject", dataset["subject"], subject_labels),
            ("run", dataset["run"], run_labels),
        ):
            if expected is not None and not np.array_equal(current, expected):
                raise RuntimeError(f"{name} labels changed across frequency bands.")
        if band_index == 0:
            labels = np.asarray(dataset["y"], dtype=np.str_)
            subject_labels = np.asarray(dataset["subject"], dtype=np.str_)
            run_labels = np.asarray(dataset["run"], dtype=np.int16)

        segmented = segment_epochs(
            dataset["X"],
            sfreq=float(cfg.get("sfreq", 160.0)),
            segment_duration=segment_duration,
            stride_duration=stride_duration,
        )
        n_trials, n_segments, n_channels, n_samples = segmented.shape
        covariance_input = segmented * float(cfg["covariance_signal_scale"])
        if brain_region_indices is None:
            covariance_input_for_energy = covariance_input
            covariances = Covariances(
                estimator=str(cfg.get("estimator", "lwf"))
            ).fit_transform(
                covariance_input.reshape(n_trials * n_segments, n_channels, n_samples)
            )
            covariances = covariances.reshape(
                n_trials, n_segments, n_channels, n_channels
            )
        else:
            n_regions, region_size = brain_region_indices.shape
            covariance_input = covariance_input[:, :, brain_region_indices, :]
            covariance_input_for_energy = covariance_input
            covariances = Covariances(
                estimator=str(cfg.get("estimator", "lwf"))
            ).fit_transform(
                covariance_input.reshape(
                    n_trials * n_segments * n_regions,
                    region_size,
                    n_samples,
                )
            )
            covariances = covariances.reshape(
                n_trials,
                n_segments,
                n_regions,
                region_size,
                region_size,
            )
        if cfg["replace_covariance_diagonal_with_raw_energy"]:
            covariances = replace_covariance_diagonal_with_segment_energy(
                covariances,
                covariance_input_for_energy,
            )
        covariances = batch_spd_diagonal_mean_normalize(covariances, eps=eps)
        covariances = regularize_spd(covariances, eps=eps)
        if baseline_mode == "rest-whitening":
            covariances = baseline_correct_covariances(
                covs=covariances,
                n_samples=dataset["X"].shape[-1],
                sfreq=float(cfg.get("sfreq", 160.0)),
                segment_duration=segment_duration,
                stride_duration=stride_duration,
                epoch_tmin=float(cfg["epoch_slice"][0]),
                baseline_window=cfg.get("baseline_window"),
                eps=eps,
            )
            covariances = trace_normalize(covariances, eps=eps)
            covariances = regularize_spd(covariances, eps=eps)
        frequencies.append(covariances.astype(output_dtype, copy=False))

    if labels is None or subject_labels is None or run_labels is None:
        raise RuntimeError("No data was loaded.")
    y, class_names = encode_labels(labels)
    x_spd = np.stack(frequencies, axis=2)
    if not np.isfinite(x_spd).all():
        raise ValueError("Preprocessed SPD data contains NaN or Inf values.")
    if not (len(x_spd) == len(y) == len(subject_labels) == len(run_labels)):
        raise RuntimeError("Trial, label, subject, and run arrays are misaligned.")
    return (
        x_spd,
        y.astype(np.int64, copy=False),
        subject_labels,
        run_labels,
        list(class_names),
    )


def _cache_path(cache_dir: Path, cfg: dict[str, Any]) -> Path:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"physionet_loro_spd_{digest}.npz"


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
                result = (
                    np.asarray(payload["x"]),
                    np.asarray(payload["y"], dtype=np.int64),
                    np.asarray(payload["subject_labels"], dtype=np.str_),
                    np.asarray(payload["run_labels"], dtype=np.int16),
                    [str(value) for value in payload["class_names"].tolist()],
                )
                print(f"Loaded run-labelled SPD cache: {path}")
                return result

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
    print(f"Saved run-labelled SPD cache: {path}")
    return result
