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

from src.datasets.BCICompetitionIV2a_preprocess import (
    load_bci_iv_2a_epochs,
    preprocess_bci_iv_2a_spd,
)
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
from src.training.config_grid import (
    expand_data_grid,
    expand_grid,
    normalize_data_time_config,
    normalize_filter_bank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_grid.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def expand_data_training_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_data_grid(config.get("data", {}))
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
    if isinstance(subjects, str):
        cleaned = subjects.strip()
        if cleaned.lower() in {"", "none", "null", "all"}:
            return None
    if isinstance(subjects, (int, np.integer)):
        return [int(subjects)]
    if isinstance(subjects, (list, tuple)):
        parsed = []
        for subject in subjects:
            values = parse_subjects(subject)
            if values is not None:
                parsed.extend(values)
        return sorted(set(parsed)) or None

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


def normalize_dataset_name(name: Any) -> str:
    normalized = str(name or "physionet_mi").strip().lower().replace("-", "_")
    aliases = {
        "physionet": "physionet_mi",
        "physionetmi": "physionet_mi",
        "physionet_mi": "physionet_mi",
        "eegbci": "physionet_mi",
        "eeg_bci": "physionet_mi",
        "bci_iv_2a": "bnci2014_001",
        "bci_competition_iv_2a": "bnci2014_001",
        "bciciv_2a": "bnci2014_001",
        "bcic_iv_2a": "bnci2014_001",
        "bnci2014_001": "bnci2014_001",
        "bnci2014001": "bnci2014_001",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown data.dataset {name!r}. Valid datasets: {valid}.")
    return aliases[normalized]


def resolve_covariance_signal_scale(value: Any, dataset_name: str) -> float:
    if value is None:
        value = "auto"
    if isinstance(value, str) and value.strip().lower() in {"auto", ""}:
        return 1.0e6 if dataset_name == "physionet_mi" else 1.0
    return float(value)


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


def optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "none",
        "null",
        "false",
    }:
        return None
    value = float(value)
    if value <= 0:
        return None
    return value


def encode_labels_in_order(labels: Any, class_names: list[str]) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(labels, dtype=np.str_)
    mapping = {name: index for index, name in enumerate(class_names)}
    unknown = sorted(set(labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}. Expected {class_names}.")
    missing = [name for name in class_names if name not in set(labels.tolist())]
    if missing:
        raise ValueError(f"Selected classes have no epochs after filtering: {missing}.")
    y = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    return y, list(class_names)


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
    data_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_cfg = dict(training_cfg)
    if data_cfg is not None:
        for key in ("seed", "test_size", "val_size", "allow_subject_overlap"):
            if key in data_cfg:
                split_cfg[key] = data_cfg[key]
    return load_or_create_split_indices(
        y=y,
        test_size=float(split_cfg.get("test_size", 0.15)),
        val_size=float(split_cfg.get("val_size", 0.15)),
        seed=int(split_cfg.get("seed", 42)),
        split_file=resolve_split_file(split_cfg.get("split_file")),
        subjects=subject_labels,
        allow_subject_overlap=parse_bool(
            split_cfg.get("allow_subject_overlap", True),
            default=True,
        ),
    )


def load_spd_like_train(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_cfg = normalize_data_time_config(data_cfg)
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "physionet_mi"))
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    epoch_tmin, epoch_tmax = data_cfg["epoch_slice"]
    segment_duration, stride_duration = data_cfg["segment_slice"]
    subjects = parse_subjects(data_cfg.get("subjects"))
    covariance_signal_scale = resolve_covariance_signal_scale(
        data_cfg.get("covariance_signal_scale", "auto"),
        dataset_name=dataset_name,
    )

    if dataset_name == "physionet_mi":
        x_spd, y, class_names, subject_labels = preprocess_spd(
            filter_bank=filter_bank,
            root_dir=str(
                data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")
            ),
            subjects=subjects,
            channels=data_cfg.get("channels"),
            estimator=str(data_cfg.get("estimator", "lwf")),
            eps=float(data_cfg.get("eps", 1e-6)),
            sfreq=float(data_cfg.get("sfreq", 160)),
            segment_duration=float(segment_duration),
            stride_duration=stride_duration,
            imaged=parse_bool(data_cfg.get("imaged", True), default=True),
            executed=parse_bool(data_cfg.get("executed", False), default=False),
            task_types=parse_task_types(data_cfg.get("task_types")),
            reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
            baseline_correction=data_cfg.get("baseline_correction"),
            baseline_window=data_cfg.get("baseline_window"),
            epoch_tmin=float(epoch_tmin),
            epoch_tmax=float(epoch_tmax),
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
            autoreject_cache_dir=data_cfg.get("autoreject_cache_dir"),
            autoreject_force_rebuild=parse_bool(
                data_cfg.get("autoreject_force_rebuild", False),
                default=False,
            ),
            return_subjects=True,
            covariance_signal_scale=covariance_signal_scale,
            replace_covariance_diagonal_with_raw_energy=parse_bool(
                data_cfg.get("replace_covariance_diagonal_with_raw_energy", False),
                default=False,
            ),
            brain_region_mode=data_cfg.get("brain_region_mode"),
            output_dtype=data_cfg.get("covariance_output_dtype", "float32"),
        )
    elif dataset_name == "bnci2014_001":
        x_spd, y, class_names, subject_labels = preprocess_bci_iv_2a_spd(
            filter_bank=filter_bank,
            root_dir=str(data_cfg.get("root_dir", "data")),
            subjects=subjects,
            channels=data_cfg.get("channels"),
            events=data_cfg.get("events", data_cfg.get("bci_iv_2a_events")),
            sessions=data_cfg.get("sessions", data_cfg.get("bci_iv_2a_sessions")),
            estimator=str(data_cfg.get("estimator", "lwf")),
            eps=float(data_cfg.get("eps", 1e-6)),
            sfreq=float(data_cfg.get("sfreq", 250)),
            segment_duration=float(segment_duration),
            stride_duration=stride_duration,
            reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
            baseline_correction=data_cfg.get("baseline_correction"),
            baseline_window=data_cfg.get("baseline_window"),
            epoch_tmin=float(epoch_tmin),
            epoch_tmax=float(epoch_tmax),
            use_ica=parse_bool(data_cfg.get("use_ica", False), default=False),
            use_autoreject=parse_bool(
                data_cfg.get("use_autoreject", False),
                default=False,
            ),
            autoreject_random_state=int(data_cfg.get("autoreject_random_state", 42)),
            autoreject_n_jobs=int(data_cfg.get("autoreject_n_jobs", 1)),
            autoreject_cv=int(data_cfg.get("autoreject_cv", 10)),
            return_subjects=True,
            covariance_signal_scale=covariance_signal_scale,
            replace_covariance_diagonal_with_raw_energy=parse_bool(
                data_cfg.get("replace_covariance_diagonal_with_raw_energy", False),
                default=False,
            ),
            output_dtype=data_cfg.get("covariance_output_dtype", "float32"),
            moabb_accept_terms=parse_bool(
                data_cfg.get("moabb_accept_terms", True),
                default=True,
            ),
            moabb_force_update=parse_bool(
                data_cfg.get("moabb_force_update", False),
                default=False,
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not np.isfinite(x_spd).all():
        raise ValueError("SPD preprocessing returned NaN or Inf values.")
    return (
        x_spd.astype(np.float64),
        y.astype(np.int64),
        np.asarray(subject_labels, dtype=np.str_),
        list(class_names),
    )


def load_segmented_epochs_like_train(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[list[float]]]:
    data_cfg = normalize_data_time_config(data_cfg)
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "physionet_mi"))
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    epoch_tmin, epoch_tmax = data_cfg["epoch_slice"]
    segment_duration, stride_duration = data_cfg["segment_slice"]
    subjects = parse_subjects(data_cfg.get("subjects"))
    sfreq = float(data_cfg.get("sfreq", 160 if dataset_name == "physionet_mi" else 250))
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
    class_names = None

    if dataset_name == "physionet_mi":
        root_dir = str(data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0"))
        channels = data_cfg.get("channels")
        task_types = parse_task_types(data_cfg.get("task_types"))
        imaged = parse_bool(data_cfg.get("imaged", True), default=True)
        executed = parse_bool(data_cfg.get("executed", False), default=False)
        for low_freq, high_freq in filter_bank:
            dataset = build_dataset(
                root_dir,
                tmin=float(epoch_tmin),
                tmax=float(epoch_tmax),
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
                autoreject_cache_dir=data_cfg.get("autoreject_cache_dir"),
                autoreject_force_rebuild=parse_bool(
                    data_cfg.get("autoreject_force_rebuild", False),
                    default=False,
                ),
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
    elif dataset_name == "bnci2014_001":
        if use_ica:
            print(
                "BCI IV-2a MOABB segmented loading uses epoched EEG arrays; "
                "data.use_ica is ignored for CSP+LDA."
            )
        if use_autoreject:
            print(
                "BCI IV-2a MOABB segmented loading uses epoched EEG arrays; "
                "data.use_autoreject is ignored for CSP+LDA."
            )
        keep_mask = None
        reject_threshold_uv = optional_positive_float(data_cfg.get("reject_threshold_uv"))
        accept_terms = parse_bool(data_cfg.get("moabb_accept_terms", True), default=True)
        force_update = parse_bool(data_cfg.get("moabb_force_update", False), default=False)
        download_checked = False
        for low_freq, high_freq in filter_bank:
            dataset = load_bci_iv_2a_epochs(
                download_dir=str(data_cfg.get("root_dir", "data")),
                subjects=subjects,
                channels=data_cfg.get("channels"),
                events=data_cfg.get("events", data_cfg.get("bci_iv_2a_events")),
                sessions=data_cfg.get("sessions", data_cfg.get("bci_iv_2a_sessions")),
                sfreq=sfreq,
                low_freq=low_freq,
                high_freq=high_freq,
                epoch_tmin=float(epoch_tmin),
                epoch_tmax=float(epoch_tmax),
                moabb_accept_terms=accept_terms and not download_checked,
                moabb_force_update=force_update and not download_checked,
            )
            download_checked = True
            x_band = np.asarray(dataset["X"])
            current_labels = np.asarray(dataset["y"], dtype=np.str_)
            current_subject_labels = np.asarray(dataset["subject"], dtype=np.str_)
            if reject_threshold_uv is not None:
                if keep_mask is None:
                    peak_to_peak_uv = np.ptp(x_band, axis=-1).max(axis=1)
                    keep_mask = peak_to_peak_uv <= reject_threshold_uv
                    rejected = int((~keep_mask).sum())
                    if rejected:
                        print(
                            f"Peak-to-peak rejected {rejected} BCI IV-2a epoch(s) "
                            f"(threshold={reject_threshold_uv:.1f} uV)."
                        )
                x_band = x_band[keep_mask]
                current_labels = current_labels[keep_mask]
                current_subject_labels = current_subject_labels[keep_mask]

            if labels is None:
                labels = current_labels
                subject_labels = current_subject_labels
                class_names = list(dataset["events"])
            else:
                if not np.array_equal(labels, current_labels):
                    raise RuntimeError("Labels changed across frequency bands.")
                if not np.array_equal(subject_labels, current_subject_labels):
                    raise RuntimeError("Subject order changed across frequency bands.")

            x_segments = segment_epochs(
                x_band,
                sfreq=sfreq,
                segment_duration=segment_duration,
                stride_duration=stride_duration,
            )
            bands.append(x_segments.astype(np.float32))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if labels is None:
        raise RuntimeError("No data was loaded.")
    if class_names is None:
        y, class_names = encode_labels(labels)
    else:
        y, class_names = encode_labels_in_order(labels, class_names)
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
    data_cfg = normalize_data_time_config(data_cfg)
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "physionet_mi"))
    epoch_tmin, epoch_tmax = data_cfg["epoch_slice"]
    subjects = parse_subjects(data_cfg.get("subjects"))
    n_ds = int(data_cfg.get("eegnet_downsample", 1))

    if dataset_name == "physionet_mi":
        x, y, class_names, subject_labels = preprocess_eegnet_author(
            root_dir=str(
                data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")
            ),
            subjects=subjects,
            n_classes=int(data_cfg.get("eegnet_num_classes", 2)),
            excluded_subjects=parse_subjects(
                data_cfg.get("eegnet_excluded_subjects", "88,92,100,104")
            ),
            T=float(data_cfg.get("eegnet_T", epoch_tmax - epoch_tmin)),
            n_ds=n_ds,
            n_ch=int(data_cfg.get("eegnet_n_channels", 64)),
            normalization=int(data_cfg.get("eegnet_normalization", 0)),
            sfreq=float(data_cfg.get("sfreq", 160)),
            scale_to_uv=parse_bool(data_cfg.get("eegnet_scale_to_uv", True), default=True),
            random_state=int(data_cfg.get("eegnet_random_state", 7)),
            max_trials_per_class=data_cfg.get("eegnet_max_trials_per_class", 7),
            return_subjects=True,
        )
    elif dataset_name == "bnci2014_001":
        filter_bank = normalize_filter_bank(data_cfg.get("filter_bank", [[8, 30]]))
        if len(filter_bank) != 1:
            print(
                "EEGNet baseline uses one band-passed trial tensor; "
                f"using the first configured band {filter_bank[0]}."
            )
        low_freq, high_freq = filter_bank[0]
        epoch_tmin = float(epoch_tmin)
        epoch_tmax = float(epoch_tmax)
        dataset = load_bci_iv_2a_epochs(
            download_dir=str(data_cfg.get("root_dir", "data")),
            subjects=subjects,
            channels=data_cfg.get("channels"),
            events=data_cfg.get("events", data_cfg.get("bci_iv_2a_events")),
            sessions=data_cfg.get("sessions", data_cfg.get("bci_iv_2a_sessions")),
            sfreq=float(data_cfg.get("sfreq", 250)),
            low_freq=low_freq,
            high_freq=high_freq,
            epoch_tmin=epoch_tmin,
            epoch_tmax=epoch_tmax,
            moabb_accept_terms=parse_bool(
                data_cfg.get("moabb_accept_terms", True),
                default=True,
            ),
            moabb_force_update=parse_bool(
                data_cfg.get("moabb_force_update", False),
                default=False,
            ),
        )
        x = np.asarray(dataset["X"], dtype=np.float32)
        labels = np.asarray(dataset["y"], dtype=np.str_)
        subject_labels = np.asarray(dataset["subject"], dtype=np.str_)

        reject_threshold_uv = optional_positive_float(data_cfg.get("reject_threshold_uv"))
        if reject_threshold_uv is not None:
            peak_to_peak_uv = np.ptp(x, axis=-1).max(axis=1)
            keep_mask = peak_to_peak_uv <= reject_threshold_uv
            rejected = int((~keep_mask).sum())
            if rejected:
                print(
                    f"Peak-to-peak rejected {rejected} BCI IV-2a epoch(s) "
                    f"(threshold={reject_threshold_uv:.1f} uV)."
                )
            x = x[keep_mask]
            labels = labels[keep_mask]
            subject_labels = subject_labels[keep_mask]

        if n_ds > 1:
            x = x[:, :, ::n_ds]

        expected_channels = data_cfg.get("eegnet_n_channels")
        expected_channels_text = str(expected_channels).strip().lower()
        if expected_channels is not None and expected_channels_text not in {
            "",
            "none",
            "null",
        }:
            expected_channels = int(expected_channels)
            if x.shape[1] != expected_channels:
                raise ValueError(
                    "data.eegnet_n_channels does not match loaded BCI IV-2a "
                    f"channels: expected {expected_channels}, got {x.shape[1]}."
                )

        if int(data_cfg.get("eegnet_normalization", 0)) == 1:
            mean = x.mean(axis=(1, 2), keepdims=True)
            std = x.std(axis=(1, 2), keepdims=True)
            x = (x - mean) / np.maximum(std, 1e-6)

        y, class_names = encode_labels_in_order(labels, list(dataset["events"]))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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
