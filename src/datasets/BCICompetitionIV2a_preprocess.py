from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.PhysioNetMI_preprocess import (
    baseline_correct_covariances,
    normalize_baseline_correction_mode,
    normalize_bool,
    normalize_float_dtype,
    regularize_spd,
    replace_covariance_diagonal_with_segment_energy,
    segment_epochs,
    trace_normalize,
)


BCI_IV_2A_EVENTS = ("left_hand", "right_hand", "feet", "tongue")

BCI_IV_2A_CHANNELS = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)


def _split_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "none", "null", "false", "all"}:
            return None
        return [part.strip() for part in cleaned.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def normalize_bci_iv_2a_events(events: Any = None) -> list[str]:
    event_names = _split_names(events)
    if event_names is None:
        return list(BCI_IV_2A_EVENTS)

    aliases = {
        "left": "left_hand",
        "left_hand": "left_hand",
        "right": "right_hand",
        "right_hand": "right_hand",
        "foot": "feet",
        "feet": "feet",
        "both_feet": "feet",
        "tongue": "tongue",
    }
    normalized = []
    for event in event_names:
        token = _normalize_token(event)
        if token not in aliases:
            valid = ", ".join(BCI_IV_2A_EVENTS)
            raise ValueError(
                f"Unknown BCI IV-2a event {event!r}. Valid events: {valid}."
            )
        normalized.append(aliases[token])

    deduplicated = list(dict.fromkeys(normalized))
    if not deduplicated:
        raise ValueError("At least one BCI IV-2a event must be selected.")
    return deduplicated


def normalize_bci_iv_2a_channels(channels: Any = None) -> list[str] | None:
    requested = _split_names(channels)
    if requested is None:
        return None

    canonical_by_key = {
        _normalize_token(channel): channel for channel in BCI_IV_2A_CHANNELS
    }
    normalized = []
    missing = []
    for channel in requested:
        key = _normalize_token(channel)
        if key not in canonical_by_key:
            missing.append(channel)
        else:
            normalized.append(canonical_by_key[key])

    if missing:
        available = ", ".join(BCI_IV_2A_CHANNELS)
        raise ValueError(
            f"Unknown BCI IV-2a EEG channel(s): {', '.join(missing)}. "
            f"Available channels: {available}."
        )
    return normalized


def normalize_bci_iv_2a_sessions(sessions: Any = None) -> list[str] | None:
    session_names = _split_names(sessions)
    if session_names is None:
        return None

    aliases = {
        "t": "train",
        "training": "train",
        "train": "train",
        "e": "test",
        "evaluation": "test",
        "eval": "test",
        "test": "test",
    }
    normalized = []
    for session in session_names:
        token = _normalize_token(session)
        normalized.append(aliases.get(token, token))
    return list(dict.fromkeys(normalized))


def _session_keep_mask(metadata, sessions: Any = None) -> np.ndarray:
    requested = normalize_bci_iv_2a_sessions(sessions)
    if requested is None:
        return np.ones(len(metadata), dtype=bool)
    if "session" not in metadata:
        raise ValueError("MOABB metadata did not contain a 'session' column.")

    session_values = [
        _normalize_token(session) for session in metadata["session"].astype(str)
    ]
    keep = []
    for session_value in session_values:
        keep.append(
            any(
                token in session_value if token in {"train", "test"}
                else token == session_value
                for token in requested
            )
        )
    return np.asarray(keep, dtype=bool)


def _format_subject_label(value: Any) -> str:
    try:
        return f"A{int(value):02d}"
    except (TypeError, ValueError):
        return str(value)


def _optional_positive_float(value: Any) -> float | None:
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


def _configure_moabb_download_dir(download_dir: str | Path | None) -> str | None:
    if download_dir in {None, ""}:
        return None

    from moabb import set_download_dir

    path = Path(str(download_dir))
    path.mkdir(parents=True, exist_ok=True)
    download_path = path.as_posix()
    set_download_dir(download_path)
    return download_path


def _download_dataset(dataset, subjects, download_dir, force_update, accept_terms):
    if not accept_terms:
        return

    kwargs = {
        "subject_list": subjects,
        "path": download_dir,
        "force_update": bool(force_update),
        "update_path": True,
    }
    try:
        dataset.download(**kwargs, accept=True)
    except TypeError:
        dataset.download(**kwargs)


def load_bci_iv_2a_epochs(
    download_dir: str | Path | None = "data",
    subjects: list[int] | None = None,
    channels: Any = None,
    events: Any = None,
    sessions: Any = None,
    sfreq: float | None = 250.0,
    low_freq: float = 8.0,
    high_freq: float = 30.0,
    epoch_tmin: float = 0.0,
    epoch_tmax: float = 4.0,
    moabb_accept_terms: bool = True,
    moabb_force_update: bool = False,
):
    try:
        import moabb
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as error:
        raise ImportError(
            "BCI Competition IV-2a preprocessing requires MOABB. "
            "Install the project environment from environment.yml first."
        ) from error

    moabb.set_log_level("warning")
    download_path = _configure_moabb_download_dir(download_dir)
    subjects = list(range(1, 10)) if subjects is None else [int(s) for s in subjects]
    invalid_subjects = [subject for subject in subjects if subject < 1 or subject > 9]
    if invalid_subjects:
        raise ValueError(
            f"BCI Competition IV-2a subject ids must be in [1, 9], "
            f"got {invalid_subjects}."
        )

    event_names = normalize_bci_iv_2a_events(events)
    channel_names = normalize_bci_iv_2a_channels(channels)

    dataset = BNCI2014_001()
    _download_dataset(
        dataset=dataset,
        subjects=subjects,
        download_dir=download_path,
        force_update=moabb_force_update,
        accept_terms=moabb_accept_terms,
    )

    paradigm = MotorImagery(
        n_classes=len(event_names),
        fmin=float(low_freq),
        fmax=float(high_freq),
        events=event_names,
        tmin=float(epoch_tmin),
        tmax=float(epoch_tmax),
        channels=channel_names,
        resample=None if sfreq is None else float(sfreq),
    )
    x, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    keep_mask = _session_keep_mask(metadata, sessions=sessions)
    if not np.any(keep_mask):
        raise RuntimeError(
            "No BCI IV-2a epochs matched the requested session filter "
            f"{sessions!r}."
        )

    metadata = metadata.loc[keep_mask].reset_index(drop=True)
    subject_labels = np.asarray(
        [_format_subject_label(subject) for subject in metadata["subject"]],
        dtype=np.str_,
    )
    return {
        "X": np.asarray(x)[keep_mask],
        "y": np.asarray(labels, dtype=np.str_)[keep_mask],
        "subject": subject_labels,
        "metadata": metadata,
        "events": event_names,
    }


def _encode_labels_in_event_order(labels, event_names):
    labels = np.asarray(labels, dtype=np.str_)
    class_names = list(event_names)
    mapping = {name: index for index, name in enumerate(class_names)}
    unknown = sorted(set(labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"Unexpected BCI IV-2a labels: {unknown}.")

    missing = [name for name in class_names if name not in set(labels.tolist())]
    if missing:
        raise ValueError(
            "Selected BCI IV-2a classes have no epochs after filtering: "
            f"{missing}."
        )

    y = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    return y, class_names


def preprocess_bci_iv_2a_spd(
    filter_bank,
    root_dir="data",
    subjects=None,
    channels=None,
    events=None,
    sessions=None,
    estimator="cov",
    eps=1e-10,
    sfreq=250,
    segment_duration=0.75,
    stride_duration=None,
    reject_threshold_uv=None,
    baseline_correction=None,
    baseline_window=None,
    epoch_tmin=0.0,
    epoch_tmax=4.0,
    use_ica=False,
    use_autoreject=False,
    autoreject_random_state=42,
    autoreject_n_jobs=1,
    autoreject_cv=10,
    return_subjects=False,
    covariance_signal_scale=1.0,
    replace_covariance_diagonal_with_raw_energy=False,
    output_dtype="float32",
    moabb_accept_terms=True,
    moabb_force_update=False,
):
    from pyriemann.estimation import Covariances

    _ = autoreject_random_state, autoreject_n_jobs, autoreject_cv
    if normalize_bool(use_ica, default=False):
        print(
            "BCI IV-2a MOABB preprocessing uses epoched EEG arrays; "
            "data.use_ica is ignored for this dataset."
        )
    if normalize_bool(use_autoreject, default=False):
        print(
            "BCI IV-2a MOABB preprocessing uses epoched EEG arrays; "
            "data.use_autoreject is ignored for this dataset."
        )

    replace_covariance_diagonal_with_raw_energy = normalize_bool(
        replace_covariance_diagonal_with_raw_energy,
        default=False,
    )
    output_dtype = normalize_float_dtype(output_dtype, default="float32")
    baseline_correction = normalize_baseline_correction_mode(baseline_correction)
    threshold_uv = _optional_positive_float(reject_threshold_uv)

    event_names = normalize_bci_iv_2a_events(events)
    frequencies = []
    labels = None
    subject_labels = None
    keep_mask = None
    download_checked = False
    accept_terms = normalize_bool(moabb_accept_terms, default=True)
    force_update = normalize_bool(moabb_force_update, default=False)

    for filter_band in filter_bank:
        dataset = load_bci_iv_2a_epochs(
            download_dir=root_dir,
            subjects=subjects,
            channels=channels,
            events=event_names,
            sessions=sessions,
            sfreq=sfreq,
            low_freq=filter_band[0],
            high_freq=filter_band[1],
            epoch_tmin=epoch_tmin,
            epoch_tmax=epoch_tmax,
            moabb_accept_terms=accept_terms and not download_checked,
            moabb_force_update=force_update and not download_checked,
        )
        download_checked = True
        temp_x = np.asarray(dataset["X"])
        current_labels = np.asarray(dataset["y"], dtype=np.str_)
        current_subject_labels = np.asarray(dataset["subject"], dtype=np.str_)
        print(f"BCI IV-2a band {filter_band}: MOABB epoch shape {temp_x.shape}")

        if threshold_uv is not None:
            if keep_mask is None:
                peak_to_peak_uv = np.ptp(temp_x, axis=-1).max(axis=1)
                keep_mask = peak_to_peak_uv <= threshold_uv
                rejected = int((~keep_mask).sum())
                if rejected:
                    print(
                        f"Peak-to-peak rejected {rejected} BCI IV-2a epoch(s) "
                        f"(threshold={threshold_uv:.1f} uV, "
                        f"median={np.median(peak_to_peak_uv):.1f}, "
                        f"p90={np.percentile(peak_to_peak_uv, 90):.1f}, "
                        f"max={np.max(peak_to_peak_uv):.1f})."
                    )
            temp_x = temp_x[keep_mask]
            current_labels = current_labels[keep_mask]
            current_subject_labels = current_subject_labels[keep_mask]

        if labels is None:
            labels = current_labels
            subject_labels = current_subject_labels
        else:
            if not np.array_equal(labels, current_labels):
                raise RuntimeError(
                    "Labels changed across BCI IV-2a frequency bands. "
                    "Check filtering, session, and rejection settings."
                )
            if not np.array_equal(subject_labels, current_subject_labels):
                raise RuntimeError(
                    "Subject order changed across BCI IV-2a frequency bands. "
                    "Check filtering, session, and rejection settings."
                )

        temp_x = segment_epochs(
            temp_x,
            sfreq=sfreq,
            segment_duration=segment_duration,
            stride_duration=stride_duration,
        )
        n_epochs, n_segments, n_channels, segment_samples = temp_x.shape
        print(
            f"BCI IV-2a band {filter_band}: segmented shape {temp_x.shape} "
            f"trials: {n_epochs}, n_segments: {n_segments}, "
            f"n_channels: {n_channels}, in_segment_samples: {segment_samples}"
        )

        covariance_input = temp_x * float(covariance_signal_scale)
        cov_x = Covariances(estimator=estimator).fit_transform(
            covariance_input.reshape(
                n_epochs * n_segments,
                n_channels,
                segment_samples,
            )
        )
        cov_x = cov_x.reshape(n_epochs, n_segments, n_channels, n_channels)

        if replace_covariance_diagonal_with_raw_energy:
            cov_x = replace_covariance_diagonal_with_segment_energy(
                cov_x,
                covariance_input,
            )
            print(
                "Replaced covariance diagonal with per-channel segment raw "
                "energy from the scaled covariance input."
            )

        print(f"Covariance matrix min: {np.min(np.abs(cov_x))}")
        print(f"Covariance matrix max: {np.max(cov_x)}")
        eigvals = np.linalg.eigvalsh(0.5 * (cov_x + np.swapaxes(cov_x, -1, -2)))
        print("eig min:", eigvals.min())
        print("eig p1:", np.percentile(eigvals, 1))
        print("eig median:", np.median(eigvals))
        print("eig max:", eigvals.max())
        print(
            "cond p99:",
            np.percentile(eigvals[..., -1] / np.maximum(eigvals[..., 0], eps), 99),
        )

        cov_x = regularize_spd(cov_x, eps=eps)

        if baseline_correction == "rest-whitening":
            cov_x = baseline_correct_covariances(
                covs=cov_x,
                n_samples=dataset["X"].shape[-1],
                sfreq=sfreq,
                segment_duration=segment_duration,
                stride_duration=stride_duration,
                epoch_tmin=epoch_tmin,
                baseline_window=baseline_window,
                eps=eps,
            )
            cov_x = trace_normalize(cov_x, eps=eps)
            cov_x = regularize_spd(cov_x, eps=eps)

        frequencies.append(cov_x.astype(output_dtype, copy=False))

    if labels is None:
        raise RuntimeError("No BCI IV-2a data was loaded.")

    y, class_names = _encode_labels_in_event_order(labels, event_names)
    x_spd = np.stack(frequencies, axis=2)

    if return_subjects:
        return x_spd, y, class_names, np.asarray(subject_labels)
    return x_spd, y, class_names
