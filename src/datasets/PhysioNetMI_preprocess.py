import random
import re
from pathlib import Path

import mne

import numpy as np


def normalize_channel_name(name):
    return str(name).strip().rstrip(".").upper()


def normalize_channels(channels):
    if channels is None:
        return None
    if isinstance(channels, str):
        channels = [channel.strip() for channel in channels.split(",")]
    channels = [channel for channel in channels if str(channel).strip()]
    return channels or None


def normalize_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def normalize_float_dtype(dtype, default="float32"):
    normalized = str(dtype or default).strip().lower()
    if normalized in {"float32", "single", "fp32", "float"}:
        return np.float32
    if normalized in {"float64", "double", "fp64"}:
        return np.float64
    raise ValueError(
        "dtype must be one of: float32, single, fp32, float, "
        "float64, double, fp64."
    )


def bandpass_filter(X, sfreq=160.0, low_freq=8.0, high_freq=30.0, **kwargs):
    """
    X shape: (n_trials, n_channels, n_times).

    This wrapper keeps backward compatibility for code that still filters
    epoched arrays. The main preprocessing path filters the continuous MNE
    Raw object before epoch extraction.
    """
    return mne.filter.filter_data(
        np.asarray(X),
        sfreq=sfreq,
        l_freq=low_freq,
        h_freq=high_freq,
        verbose=False,
        **kwargs,
    )


def pick_raw_channels(raw, channels=None):
    raw.pick("eeg", exclude=[])
    rename_mapping = {
        channel_name: normalize_channel_name(channel_name)
        for channel_name in raw.ch_names
        if normalize_channel_name(channel_name) != channel_name
    }
    if rename_mapping:
        raw.rename_channels(rename_mapping)
    channels = normalize_channels(channels)
    if channels is None:
        return raw

    requested = [normalize_channel_name(channel) for channel in channels]
    normalized_to_name = {
        normalize_channel_name(channel_name): channel_name
        for channel_name in raw.ch_names
    }
    missing = [
        channel
        for channel in requested
        if channel not in normalized_to_name
    ]
    if missing:
        available = ", ".join(normalize_channel_name(name) for name in raw.ch_names)
        raise ValueError(
            f"Unknown EEG channels: {', '.join(missing)}. "
            f"Available channels: {available}"
        )

    raw.pick([normalized_to_name[channel] for channel in requested])
    return raw


def set_standard_eeg_montage(raw, montage_name="standard_1005"):
    try:
        raw.set_montage(
            montage_name,
            match_case=False,
            on_missing="ignore",
            verbose=False,
        )
    except Exception as error:
        print(
            f"Could not set EEG montage {montage_name!r}; "
            f"AutoReject interpolation may be limited. Error: {error}"
        )
    return raw


def filter_raw_band(raw, low_freq, high_freq):
    raw = raw.copy()
    raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        phase="zero-double",
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=False,
    )
    return raw


def normalize_optional_channels(channels):
    if channels is None:
        return None
    if isinstance(channels, str):
        cleaned = channels.strip()
        if cleaned.lower() in {"", "none", "null", "false"}:
            return None
        return [channel.strip() for channel in cleaned.split(",") if channel.strip()]
    return [str(channel).strip() for channel in channels if str(channel).strip()]


def apply_ica_artifact_removal(
    raw,
    n_components=20,
    random_state=42,
    eog_channels=None,
):
    """
    Fit ICA on continuous Raw and remove EOG-related components when they can
    be detected.

    PhysioNet EEGBCI usually has no dedicated EOG channel, so this function is
    intentionally conservative: if no EOG/proxy channel is available, it leaves
    Raw unchanged instead of excluding arbitrary ICA components.
    """
    eog_channels = normalize_optional_channels(eog_channels)
    n_eeg_channels = len(raw.ch_names)
    if n_eeg_channels < 2:
        print("ICA artifact removal skipped: fewer than 2 EEG channels.")
        return raw

    if n_components is None:
        effective_n_components = None
    else:
        effective_n_components = min(int(n_components), n_eeg_channels - 1)
        if effective_n_components < 1:
            print("ICA artifact removal skipped: n_components < 1.")
            return raw

    raw = raw.copy()
    print(
        "Fitting ICA artifact removal "
        f"(n_components={effective_n_components}, random_state={random_state})."
    )
    ica = mne.preprocessing.ICA(
        n_components=effective_n_components,
        random_state=int(random_state),
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)

    try:
        eog_indices, _ = ica.find_bads_eog(
            raw,
            ch_name=eog_channels,
            verbose=False,
        )
    except Exception as error:
        print(
            "ICA fitted, but EOG component detection was skipped/failed "
            f"({error}). Raw is left unchanged."
        )
        return raw

    if not eog_indices:
        print("ICA found no EOG-related components to exclude.")
        return raw

    ica.exclude = eog_indices
    raw = ica.apply(raw, verbose=False)
    print(f"ICA excluded EOG-related component(s): {eog_indices}")
    return raw


def autoreject_keep_mask(
    epochs,
    info,
    tmin,
    random_state=42,
    n_jobs=1,
    cv=10,
):
    """
    Use autoreject.AutoReject on an EpochsArray and return the kept epoch mask.

    The cleaned Epochs object may also contain interpolated channels, but this
    preprocessing pipeline uses AutoReject primarily to decide which trials to
    drop consistently before SPD covariance estimation.
    """
    epochs = np.asarray(epochs)
    if len(epochs) < 2:
        return np.ones(len(epochs), dtype=bool)

    try:
        from autoreject import AutoReject
    except ImportError as error:
        raise ImportError(
            "data.use_autoreject=True requires the 'autoreject' package. "
            "Install it in the spd_transformer environment first."
        ) from error

    cv = int(cv)
    cv = max(2, min(cv, len(epochs)))
    events = np.column_stack(
        [
            np.arange(len(epochs), dtype=int),
            np.zeros(len(epochs), dtype=int),
            np.ones(len(epochs), dtype=int),
        ]
    )
    epochs_mne = mne.EpochsArray(
        epochs,
        info.copy(),
        events=events,
        event_id={"artifact": 1},
        tmin=float(tmin),
        verbose=False,
    )
    ar = AutoReject(
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        cv=cv,
    )
    epochs_clean = ar.fit_transform(epochs_mne)
    keep_mask = np.zeros(len(epochs), dtype=bool)
    keep_mask[np.asarray(epochs_clean.selection, dtype=int)] = True
    return keep_mask



def trace_normalize(covs, eps=1e-10):

    """
    trace=d
    :param covs:
    :param eps:
    :return:
    """
    n_channels = covs.shape[-1]
    traces = np.trace(covs, axis1=-2, axis2=-1)
    traces = np.maximum(traces, eps)
    return covs * n_channels / traces[..., None, None]


def regularize_spd(covs, eps=1e-10):
    n_channels = covs.shape[-1]
    eye = np.eye(n_channels, dtype=covs.dtype)
    return covs + eps * eye


def matrix_log_spd(covs, eps=1e-10):
    covs = 0.5 * (covs + np.swapaxes(covs, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(covs)
    log_eigvals = np.log(np.clip(eigvals, eps, None))
    return (eigvecs * log_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)


def matrix_exp_sym(mats):
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(mats)
    exp_eigvals = np.exp(eigvals)
    out = (eigvecs * exp_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def matrix_inv_sqrt_spd(covs, eps=1e-10):
    covs = 0.5 * (covs + np.swapaxes(covs, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(covs)
    inv_sqrt_eigvals = 1.0 / np.sqrt(np.clip(eigvals, eps, None))
    out = (eigvecs * inv_sqrt_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def encode_labels(labels):
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return y.astype(np.int64), encoder.classes_


def segment_epochs(
    X,
    sfreq=160,
    segment_duration=0.75,
    stride_duration=None,
):
    """
    Split each epoch into fixed-length temporal segments.

    X shape: (n_epochs, n_channels, n_samples_per_epoch)
    return shape: (n_epochs, n_segments, n_channels, segment_samples)

    If stride_duration is None, use non-overlapping segments.
     50% overlap  segment_duration=0.75, stride_duration=0.375,
    """
    if segment_duration <= 0:
        raise ValueError(f"segment_duration must be positive, got {segment_duration}.")

    if stride_duration is None:
        stride_duration = segment_duration
    if stride_duration <= 0:
        raise ValueError(f"stride_duration must be positive, got {stride_duration}.")

    segment_samples = int(round(segment_duration * sfreq))
    stride_samples = int(round(stride_duration * sfreq))

    n_samples = X.shape[-1]
    if segment_samples > n_samples:
        raise ValueError(
            f"segment_samples={segment_samples} is longer than epoch length {n_samples}."
        )

    starts = range(0, n_samples - segment_samples + 1, stride_samples)
    segments = [
        X[..., start:start + segment_samples]
        for start in starts
    ]

    if not segments:
        raise ValueError(
            "No segments were created. Check segment_duration and stride_duration."
        )

    return np.stack(segments, axis=1)


def segment_center_times(
    n_samples,
    sfreq=160,
    segment_duration=0.75,
    stride_duration=None,
    epoch_tmin=-2.0,
):
    if stride_duration is None:
        stride_duration = segment_duration
    segment_samples = int(round(segment_duration * sfreq))
    stride_samples = int(round(stride_duration * sfreq))
    starts = np.array(
        list(range(0, n_samples - segment_samples + 1, stride_samples)),
        dtype=float,
    )
    return epoch_tmin + (starts + 0.5 * segment_samples) / sfreq


def parse_baseline_window(baseline_window):
    if baseline_window is None:
        return -2.0, 0.0
    if isinstance(baseline_window, str):
        cleaned = baseline_window.strip().strip("[]()")
        parts = [part.strip() for part in cleaned.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "baseline_window string must contain two comma-separated values, "
                f"got {baseline_window!r}."
            )
        return float(parts[0]), float(parts[1])
    if len(baseline_window) != 2:
        raise ValueError(
            f"baseline_window must contain two values, got {baseline_window!r}."
        )
    return float(baseline_window[0]), float(baseline_window[1])


def normalize_baseline_correction_mode(mode):
    if mode is None or mode is False:
        return None
    mode = str(mode).strip().lower().replace("_", "-")
    if mode in {"none", "null", "false", ""}:
        return None
    aliases = {
        "rest-whitening": "rest-whitening",
        "covariance-whitening": "rest-whitening",
        "riemannian-whitening": "rest-whitening",
        "spd-whitening": "rest-whitening",
    }
    if mode not in aliases:
        raise ValueError(
            "baseline_correction must be null or one of "
            "'rest-whitening', 'covariance-whitening', "
            f"'riemannian-whitening'. Got {mode!r}."
        )
    return aliases[mode]


def baseline_correct_covariances(
    covs,
    n_samples,
    sfreq=160,
    segment_duration=0.75,
    stride_duration=None,
    epoch_tmin=-2.0,
    baseline_window=None,
    eps=1e-10,
):
    """
    Correct segment covariances using each epoch's rest-state covariance.

    For every epoch, baseline segments are selected from baseline_window and
    pooled by the Log-Euclidean mean B. All segment covariances C are whitened
    as B^{-1/2} C B^{-1/2}, preserving SPD structure.
    """
    baseline_start, baseline_end = parse_baseline_window(baseline_window)
    centers = segment_center_times(
        n_samples=n_samples,
        sfreq=sfreq,
        segment_duration=segment_duration,
        stride_duration=stride_duration,
        epoch_tmin=epoch_tmin,
    )
    baseline_mask = (centers >= baseline_start) & (centers < baseline_end)
    if not np.any(baseline_mask):
        raise ValueError(
            "No baseline segments found for baseline correction. "
            f"segment centers={centers.tolist()}, "
            f"baseline_window=({baseline_start}, {baseline_end})."
        )

    baseline_covs = covs[:, baseline_mask]
    baseline_log_mean = matrix_log_spd(baseline_covs, eps=eps).mean(axis=1)
    baseline_cov = matrix_exp_sym(baseline_log_mean)
    baseline_inv_sqrt = matrix_inv_sqrt_spd(baseline_cov, eps=eps)
    corrected = baseline_inv_sqrt[:, None] @ covs @ baseline_inv_sqrt[:, None]
    corrected = 0.5 * (corrected + np.swapaxes(corrected, -1, -2))
    print(
        "Applied rest-state covariance baseline correction "
        f"with window=({baseline_start}, {baseline_end}) and "
        f"{int(baseline_mask.sum())} baseline segment(s)."
    )
    return corrected


def is_bad_epoch(epoch, threshold_uv=None):
    """
    Return True when an epoch contains excessive peak-to-peak amplitude.

    MNE raw data are in Volts, while EEG rejection thresholds are commonly
    specified in microvolts. If threshold_uv is None or non-positive, rejection
    is disabled.
    """
    if threshold_uv is None:
        return False

    threshold_uv = float(threshold_uv)
    if threshold_uv <= 0:
        return False

    threshold_v = threshold_uv * 1e-6
    peak_to_peak = np.ptp(epoch, axis=-1)
    return bool(np.any(peak_to_peak > threshold_v))


def reject_bad_epochs(epochs, labels, threshold_uv=None):
    """
    Reject epochs whose peak-to-peak amplitude exceeds threshold_uv.

    Parameters
    ----------
    epochs : array-like, shape (n_epochs, n_channels, n_times)
        Epochs used for artifact detection. Values are expected in Volts.
    labels : array-like, shape (n_epochs,)
        Labels aligned with epochs.
    threshold_uv : float | None
        Peak-to-peak rejection threshold in microvolts. None or <= 0 disables
        rejection.

    Returns
    -------
    kept_epochs, kept_labels, keep_mask
    """
    epochs = np.asarray(epochs)
    labels = np.asarray(labels)
    if len(epochs) == 0:
        return epochs, labels, np.array([], dtype=bool)

    if threshold_uv is None or float(threshold_uv) <= 0:
        return epochs, labels, np.ones(len(epochs), dtype=bool)

    threshold_v = float(threshold_uv) * 1e-6
    peak_to_peak = np.ptp(epochs, axis=-1)
    keep_mask = (peak_to_peak <= threshold_v).all(axis=1)
    return epochs[keep_mask], labels[keep_mask], keep_mask


def epoch_peak_to_peak_uv(epochs):
    """
    Return each epoch's maximum channel-wise peak-to-peak amplitude in uV.
    """
    epochs = np.asarray(epochs)
    if len(epochs) == 0:
        return np.array([], dtype=float)
    return np.ptp(epochs, axis=-1).max(axis=1) * 1e6


def map_event_to_label(run_id, event_name):
    """
    T1/T2 -> label
    """
    LEFT_RIGHT_RUNS = {4, 8, 12}
    HANDS_FEET_RUNS = {6, 10, 14}

    if run_id in LEFT_RIGHT_RUNS:

        if event_name == "T1":
            return "left_hand"

        if event_name == "T2":
            return "right_hand"

    elif run_id in HANDS_FEET_RUNS:

        if event_name == "T1":
            return "hands"

        if event_name == "T2":
            return "feet"

    return None


EEGNET_AUTHOR_EXCLUDED_SUBJECTS = (88, 92, 100, 104)

EEGNET_AUTHOR_CHANNEL_INDICES = {
    64: np.arange(64),
    38: np.array(
        [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22,
            23, 24, 26, 28, 29, 31, 33, 35, 37, 40, 41, 42,
            43, 46, 48, 50, 52, 54, 55, 57, 59, 60, 61, 62, 63,
        ]
    ),
    27: np.array(
        [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 38, 39, 40, 41, 44, 45,
        ]
    ),
    19: np.array(
        [8, 10, 12, 21, 23, 29, 31, 33, 35, 37, 40, 41, 46, 48, 50, 52, 54, 60, 62]
    ),
    8: np.array([8, 10, 12, 25, 27, 48, 52, 57]),
}


def eegnet_author_class_names(n_classes):
    if int(n_classes) == 2:
        return ["left_hand", "right_hand"]
    if int(n_classes) == 3:
        return ["left_hand", "right_hand", "rest"]
    if int(n_classes) == 4:
        return ["left_hand", "right_hand", "rest", "feet"]
    raise ValueError(f"EEGNet author preprocessing supports 2, 3, or 4 classes, got {n_classes}.")


def eegnet_author_runs(n_classes):
    n_classes = int(n_classes)
    if n_classes == 2:
        return [4, 8, 12]
    if n_classes == 3:
        return [1, 4, 8, 12]
    if n_classes == 4:
        return [1, 4, 6, 8, 10, 12, 14]
    raise ValueError(f"EEGNet author preprocessing supports 2, 3, or 4 classes, got {n_classes}.")


def _normalize_subject_ids(subjects):
    if subjects is None:
        return None
    normalized = []
    for subject in subjects:
        text = str(subject).strip().upper()
        if not text:
            continue
        if text.startswith("S"):
            text = text[1:]
        normalized.append(int(text))
    return sorted(set(normalized))


def _eegnet_author_label(run_id, event_name):
    first_set = {4, 8, 12}
    second_set = {6, 10, 14}
    if run_id in first_set:
        if event_name == "T1":
            return 0
        if event_name == "T2":
            return 1
    if run_id in second_set and event_name == "T2":
        return 3
    return None


def _eegnet_author_reduce(x, n_ds=1, n_ch=64, T=3.0, fs=160.0):
    n_ds = int(n_ds)
    n_ch = int(n_ch)
    if n_ds < 1:
        raise ValueError(f"n_ds must be >= 1, got {n_ds}.")
    if n_ch not in EEGNET_AUTHOR_CHANNEL_INDICES:
        raise ValueError(
            "n_ch must be one of "
            f"{sorted(EEGNET_AUTHOR_CHANNEL_INDICES)}, got {n_ch}."
        )

    channels = EEGNET_AUTHOR_CHANNEL_INDICES[n_ch]
    n_samples_original = int(T * fs)
    n_samples = int(np.ceil(T * fs / n_ds))
    x = x[:, channels, :n_samples_original]
    if n_ds == 1:
        return x

    import scipy.signal as scipy_signal

    reduced = np.zeros((x.shape[0], n_ch, n_samples), dtype=x.dtype)
    for trial in range(x.shape[0]):
        for channel in range(n_ch):
            reduced[trial, channel] = scipy_signal.decimate(
                x[trial, channel],
                n_ds,
            )
    return reduced


def _eegnet_author_normalize_trials(x):
    x = np.asarray(x, dtype=np.float32).copy()
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std


def preprocess_eegnet_author(
    root_dir,
    subjects=None,
    n_classes=2,
    excluded_subjects=EEGNET_AUTHOR_EXCLUDED_SUBJECTS,
    T=3.0,
    n_ds=1,
    n_ch=64,
    normalization=0,
    sfreq=160.0,
    scale_to_uv=True,
    random_state=7,
    max_trials_per_class=7,
    return_subjects=True,
):
    """
    Load PhysioNet EEGBCI trials using the EEGNet author's global-model setup.

    This intentionally avoids the SPD preprocessing path: no re-reference, no
    filtering, no notch filter, no ICA/autoreject, and no covariance estimation.
    Trials are cut directly from EDF annotations, then optionally channel-
    reduced/downsampled like eeg_reduction.py in the author repository.
    """
    n_classes = int(n_classes)
    if max_trials_per_class is not None:
        max_trials_per_class = int(max_trials_per_class)
        if max_trials_per_class <= 0:
            max_trials_per_class = None
    class_names = eegnet_author_class_names(n_classes)
    target_runs = eegnet_author_runs(n_classes)
    excluded_subjects = set(_normalize_subject_ids(excluded_subjects) or [])
    subject_filter = _normalize_subject_ids(subjects)
    if subject_filter is None:
        subject_filter = list(range(1, 110))
    subject_filter = [
        subject for subject in subject_filter if subject not in excluded_subjects
    ]

    root = Path(root_dir)
    n_samples = int(round(float(T) * float(sfreq)))
    rng = random.Random(random_state)
    x_all = []
    y_all = []
    subject_labels = []

    for subject in subject_filter:
        subject_name = f"S{subject:03d}"
        subject_dir = root / subject_name
        if not subject_dir.exists():
            continue

        for run_id in target_runs:
            edf_file = subject_dir / f"{subject_name}R{run_id:02d}.edf"
            if not edf_file.exists():
                continue

            raw = mne.io.read_raw_edf(
                edf_file,
                preload=True,
                verbose=False,
            )
            run_sfreq = float(raw.info["sfreq"])
            if not np.isclose(run_sfreq, float(sfreq), atol=1e-5):
                raise ValueError(
                    f"Sampling rate mismatch for {edf_file}: "
                    f"expected {sfreq}, got {run_sfreq}."
                )
            data = raw.get_data()
            if scale_to_uv:
                data = data * 1e6

            if run_id == 1:
                for index in range(20):
                    start = index * n_samples
                    end = start + n_samples
                    if end <= data.shape[1]:
                        x_all.append(data[:, start:end])
                        y_all.append(2)
                        subject_labels.append(subject_name)
                max_start = max(0, int(data.shape[1] - n_samples))
                if max_start > 0:
                    start = rng.randint(0, max_start)
                    end = start + n_samples
                    x_all.append(data[:, start:end])
                    y_all.append(2)
                    subject_labels.append(subject_name)
                continue

            counters = {0: 0, 1: 0, 3: 0}
            for onset, event_name in zip(
                raw.annotations.onset,
                raw.annotations.description,
            ):
                label = _eegnet_author_label(run_id, event_name)
                if label is None:
                    continue
                if (
                    max_trials_per_class is not None
                    and counters[label] >= max_trials_per_class
                ):
                    continue

                start = int(onset * run_sfreq)
                end = start + n_samples
                if start < 0 or end > data.shape[1]:
                    continue
                x_all.append(data[:, start:end])
                y_all.append(label)
                subject_labels.append(subject_name)
                counters[label] += 1

    if not x_all:
        raise RuntimeError(
            "No EEGNet author-style trials were loaded. Check root_dir, "
            "subjects, n_classes, and EDF file availability."
        )

    x = np.asarray(x_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    x = _eegnet_author_reduce(
        x,
        n_ds=n_ds,
        n_ch=n_ch,
        T=T,
        fs=sfreq,
    ).astype(np.float32, copy=False)
    if int(normalization) == 1:
        x = _eegnet_author_normalize_trials(x)

    if not np.isfinite(x).all():
        raise ValueError("EEGNet author-style data contains NaN or Inf values.")

    if return_subjects:
        return x, y, class_names, np.asarray(subject_labels, dtype=np.str_)
    return x, y, class_names


def extract_transition_epochs(
    edf_file,
    tmin=-3.0,
    tmax=3.8,
    low_freq=None,
    high_freq=None,
    channels=None,
    reject_threshold_uv=None,
    use_ica=False,
    ica_n_components=20,
    ica_random_state=42,
    ica_eog_channels=None,
    use_autoreject=False,
    autoreject_random_state=42,
    autoreject_n_jobs=1,
    autoreject_cv=10,
):
    """
    take T1/T2 onset as the anchor

    [-tmin, +tmax]

    label = task type
    """
    use_ica = normalize_bool(use_ica, default=False)
    use_autoreject = normalize_bool(use_autoreject, default=False)

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=True,
        verbose=False
    )
    # Re-reference before filtering and epoching
    raw.set_eeg_reference("average", projection=False)
    raw = pick_raw_channels(raw, channels=channels)
    raw = set_standard_eeg_montage(raw)
    raw.filter(l_freq=0.5, h_freq=None)
    raw.filter(l_freq=None, h_freq=40)
    raw.notch_filter(freqs=60)
    if use_ica:
        raw = apply_ica_artifact_removal(
            raw,
            n_components=ica_n_components,
            random_state=ica_random_state,
            eog_channels=ica_eog_channels,
        )
    artifact_info = raw.info.copy()
    artifact_data = raw.get_data()

    if low_freq is not None or high_freq is not None:
        if low_freq is None or high_freq is None:
            raise ValueError("low_freq and high_freq must be provided together.")
        raw = filter_raw_band(raw, low_freq=low_freq, high_freq=high_freq)

    sfreq = raw.info["sfreq"]
    if not np.isclose(sfreq, 160.0, atol=1e-5):
        print(f"Sampling Rate doesn't match: {sfreq}, skip {edf_file}")
        return [], []


    annotations = raw.annotations

    run_id = int(
        re.search(r"R(\d+)", edf_file).group(1)
    )

    X = []
    y = []
    artifact_epochs = []

    data = raw.get_data()

    for onset, desc in zip(
        annotations.onset,
        annotations.description,
    ):

        if desc not in ["T1", "T2"]:
            continue

        label = map_event_to_label(
            run_id,
            desc
        )

        if label is None:
            continue

        center = int(onset * sfreq)

        start = int(center + tmin * sfreq)
        end = int(center + tmax * sfreq)

        if start < 0:
            continue

        if end > data.shape[1]:
            continue

        epoch = data[:, start:end]
        X.append(epoch)
        artifact_epochs.append(artifact_data[:, start:end])
        y.append(label)

    if not X:
        return [], []

    artifact_epochs = np.asarray(artifact_epochs)
    X = np.asarray(X)
    y = np.asarray(y)

    if use_autoreject:
        autoreject_mask = autoreject_keep_mask(
            artifact_epochs,
            info=artifact_info,
            tmin=tmin,
            random_state=autoreject_random_state,
            n_jobs=autoreject_n_jobs,
            cv=autoreject_cv,
        )
        autoreject_rejected = int((~autoreject_mask).sum())
        if autoreject_rejected:
            print(
                f"AutoReject rejected {autoreject_rejected} epoch(s) "
                f"from {edf_file}."
            )
        X = X[autoreject_mask]
        y = y[autoreject_mask]
        artifact_epochs = artifact_epochs[autoreject_mask]
        if len(X) == 0:
            return [], []

    _, y, keep_mask = reject_bad_epochs(
        artifact_epochs,
        y,
        threshold_uv=reject_threshold_uv,
    )
    X = X[keep_mask]
    rejected = int((~keep_mask).sum())
    if rejected:
        ptp_uv = epoch_peak_to_peak_uv(artifact_epochs)
        print(
            f"Peak-to-peak rejected {rejected} bad epoch(s) from {edf_file} "
            f"(threshold={float(reject_threshold_uv):.1f} uV, "
            f"ptp_uv median={np.median(ptp_uv):.1f}, "
            f"p90={np.percentile(ptp_uv, 90):.1f}, "
            f"max={np.max(ptp_uv):.1f})."
        )

    return list(X), list(y)


def load_subject(
    subject_dir,
    tmin=-2.0,
    tmax=4.0,
    imaged=True,
    executed=False,
    task_types=("unilateral_fist", "both"),
    low_freq=None,
    high_freq=None,
    channels=None,
    reject_threshold_uv=None,
    use_ica=False,
    ica_n_components=20,
    ica_random_state=42,
    ica_eog_channels=None,
    use_autoreject=False,
    autoreject_random_state=42,
    autoreject_n_jobs=1,
    autoreject_cv=10,
):
    imaged = normalize_bool(imaged, default=True)
    executed = normalize_bool(executed, default=False)
    use_ica = normalize_bool(use_ica, default=False)
    use_autoreject = normalize_bool(use_autoreject, default=False)
    X_all = []
    y_all = []

    target_run_id = []
    if imaged:
        if "unilateral_fist" in task_types:
            target_run_id.extend([4, 8, 12])
        if "both" in task_types:
            target_run_id.extend([6, 10, 14])
    if executed:
        if "unilateral_fist" in task_types:
            target_run_id.extend([3, 7, 11])
        if "both" in task_types:
            target_run_id.extend([5, 9, 13])


    for edf_file in Path(subject_dir).glob("*.edf"):

        run_id = int(
            re.search(
                r"R(\d+)",
                edf_file.name
            ).group(1)
        )

        if run_id not in target_run_id:
            continue

        X, y = extract_transition_epochs(
            str(edf_file),
            tmin,
            tmax,
            low_freq=low_freq,
            high_freq=high_freq,
            channels=channels,
            reject_threshold_uv=reject_threshold_uv,
            use_ica=use_ica,
            ica_n_components=ica_n_components,
            ica_random_state=ica_random_state,
            ica_eog_channels=ica_eog_channels,
            use_autoreject=use_autoreject,
            autoreject_random_state=autoreject_random_state,
            autoreject_n_jobs=autoreject_n_jobs,
            autoreject_cv=autoreject_cv,
        )
        if X and y:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        return None, None

    return np.concatenate(X_all), np.concatenate(y_all)

def build_dataset(
    root_dir,
    tmin=-2.0,
    tmax=4.0,
    subjects=None,
    imaged=True,
    executed=False,
    task_types=("unilateral_fist", "both"),
    low_freq=None,
    high_freq=None,
    channels=None,
    reject_threshold_uv=None,
    use_ica=False,
    ica_n_components=20,
    ica_random_state=42,
    ica_eog_channels=None,
    use_autoreject=False,
    autoreject_random_state=42,
    autoreject_n_jobs=1,
    autoreject_cv=10,
):
    imaged = normalize_bool(imaged, default=True)
    executed = normalize_bool(executed, default=False)
    use_ica = normalize_bool(use_ica, default=False)
    use_autoreject = normalize_bool(use_autoreject, default=False)
    X_all = []
    y_all = []
    subject_labels = []

    subject_dirs = sorted(Path(root_dir).glob("S*"))
    if subjects is not None:
        requested_subjects = {
            (
                f"S{int(str(subject)[1:]):03d}"
                if str(subject).upper().startswith("S")
                else f"S{int(subject):03d}"
            )
            for subject in subjects
        }
        print(f"requested subjects: {requested_subjects}")
        subject_dirs = [
            subject_dir
            for subject_dir in subject_dirs
            if subject_dir.name.upper() in requested_subjects
        ]
        missing_subjects = requested_subjects - {
            subject_dir.name.upper() for subject_dir in subject_dirs
        }
        if missing_subjects:
            missing = ", ".join(sorted(missing_subjects))
            raise ValueError(f"Subject directories not found under {root_dir}: {missing}")

    if not subject_dirs:
        raise ValueError(f"No subject directories found under {root_dir}.")

    for subject_dir in subject_dirs:

        X, y = load_subject(
            subject_dir,
            tmin,
            tmax,
            imaged=imaged,
            executed=executed,
            task_types=task_types,
            low_freq=low_freq,
            high_freq=high_freq,
            channels=channels,
            reject_threshold_uv=reject_threshold_uv,
            use_ica=use_ica,
            ica_n_components=ica_n_components,
            ica_random_state=ica_random_state,
            ica_eog_channels=ica_eog_channels,
            use_autoreject=use_autoreject,
            autoreject_random_state=autoreject_random_state,
            autoreject_n_jobs=autoreject_n_jobs,
            autoreject_cv=autoreject_cv,
        )
        if X is None:
            print(
                f"Subject {subject_dir}: no epochs kept "
                f"(band={low_freq}-{high_freq} Hz, "
                f"reject_threshold_uv={reject_threshold_uv})."
            )
            continue
        X = np.array(X)
        y = np.array(y)

        X_all.append(X)
        y_all.append(y)

        print(f"Subject {subject_dir}: {X.shape}")

        subject_labels.extend(
            [subject_dir.name] * len(y)
        )

    if not X_all:
        task_type_text = ",".join(task_types) if task_types is not None else "None"
        raise RuntimeError(
            "No epochs were loaded after preprocessing. "
            f"root_dir={root_dir}, subjects={subjects}, imaged={imaged}, "
            f"executed={executed}, task_types={task_type_text}, "
            f"band={low_freq}-{high_freq} Hz, channels={channels}, "
            f"reject_threshold_uv={reject_threshold_uv}. "
            "If artifact rejection is enabled, the threshold is probably too "
            "strict for this dataset/window. Try reject_threshold_uv=null, "
            "300, or 500 first, then inspect the rejection counts."
        )

    return {
        "X": np.concatenate(X_all),
        "y": np.concatenate(y_all),
        "subject": np.array(subject_labels),
    }


def preprocess_spd(
    filter_bank,
    root_dir="data/MNE-eegbci-data/files/eegmmidb/1.0.0",
    subjects=None,
    channels=None,
    imaged=True,
    executed=False,
    task_types=("unilateral_fist", "both"),
    estimator="cov",
    eps=1e-10,
    sfreq=160,
    segment_duration=0.75,
    stride_duration=None,
    reject_threshold_uv=None,
    baseline_correction=None,
    baseline_window=None,
    epoch_tmin=-2.0,
    epoch_tmax=4.0,
    use_ica=False,
    ica_n_components=20,
    ica_random_state=42,
    ica_eog_channels=None,
    use_autoreject=False,
    autoreject_random_state=42,
    autoreject_n_jobs=1,
    autoreject_cv=10,
    return_subjects=False,
    covariance_signal_scale=1e6,
    output_dtype="float32",
):
    from pyriemann.estimation import Covariances

    imaged = normalize_bool(imaged, default=True)
    executed = normalize_bool(executed, default=False)
    use_ica = normalize_bool(use_ica, default=False)
    use_autoreject = normalize_bool(use_autoreject, default=False)
    output_dtype = normalize_float_dtype(output_dtype, default="float32")

    frequencies = []
    labels = None
    subject_labels = None
    baseline_correction = normalize_baseline_correction_mode(baseline_correction)


    for filter in filter_bank:

        # 1. Filter continuous Raw first, then extract epochs.
        dataset = build_dataset(
            root_dir,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            subjects=subjects,
            imaged=imaged,
            task_types=task_types,
            executed=executed,
            low_freq=filter[0],
            high_freq=filter[1],
            channels=channels,
            reject_threshold_uv=reject_threshold_uv,
            use_ica=use_ica,
            ica_n_components=ica_n_components,
            ica_random_state=ica_random_state,
            ica_eog_channels=ica_eog_channels,
            use_autoreject=use_autoreject,
            autoreject_random_state=autoreject_random_state,
            autoreject_n_jobs=autoreject_n_jobs,
            autoreject_cv=autoreject_cv,
        )
        temp_x = dataset['X']   #[n_epochs, n_channels, n_samples_per_epoch]
        print(f"Band {filter}: raw-filtered epoch shape {temp_x.shape}")

        if labels is None:
            labels = dataset['y']
            subject_labels = dataset['subject']
        else:
            if not np.array_equal(labels, dataset['y']):
                raise RuntimeError(
                    "Labels changed across frequency bands. "
                    "Check raw filtering and epoch extraction settings."
                )
            if not np.array_equal(subject_labels, dataset['subject']):
                raise RuntimeError(
                    "Subject order changed across frequency bands. "
                    "Check raw filtering and epoch extraction settings."
                )

        temp_x = segment_epochs(
            temp_x,
            sfreq=sfreq,
            segment_duration=segment_duration,
            stride_duration=stride_duration,
        )
        n_epochs, n_segments, n_channels, segment_samples = temp_x.shape
        print(
            f"Band {filter}: segmented shape {temp_x.shape} "
            f"trials: {n_epochs}, n_segments: {n_segments}, n_channels: {n_channels}, in_segment_samples: {segment_samples}"
        )

        # 3. Compute covariance matrices using pyriemann.
        # MNE EEG data are in Volts, so raw covariance values are in V^2
        # and can be around 1e-12..1e-8. Scaling to microvolts keeps the
        # covariance estimator and diagonal regularization away from tiny
        # floating-point magnitudes; trace normalization below removes this
        # global scale before the model sees the matrices.
        covariance_input = temp_x * float(covariance_signal_scale)
        cov_x = Covariances(estimator=estimator).fit_transform(
            covariance_input.reshape(
                n_epochs * n_segments,
                n_channels,
                segment_samples,
            )
        )
        cov_x = cov_x.reshape(n_epochs, n_segments, n_channels, n_channels)

        # 4. Normalize SPD scale
        print(f"Covariance matrix min: {np.min(np.abs(cov_x))}")
        print(f"Covariance matrix max: {np.max(cov_x)}")

        eigvals = np.linalg.eigvalsh(0.5 * (cov_x + np.swapaxes(cov_x, -1, -2)))
        print("eig min:", eigvals.min())
        print("eig p1:", np.percentile(eigvals, 1))
        print("eig median:", np.median(eigvals))
        print("eig max:", eigvals.max())
        print("cond p99:", np.percentile(eigvals[..., -1] / np.maximum(eigvals[..., 0], eps), 99))

        cov_x = trace_normalize(cov_x, eps=eps)

        print(f"norm matrix min: {np.min(np.abs(cov_x))}")
        print(f"norm matrix max: {np.max(cov_x)}")



        # 5. Make sure matrices are strictly SPD
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
        raise RuntimeError("No data was loaded. Check root_dir, subjects, and task settings.")

    # 6. Encode labels
    y, class_names = encode_labels(labels)

    # Output shape: (n_trials, segment, frequency, n_channels, n_channels)
    X_spd = np.stack(frequencies, axis=2)

    if return_subjects:
        return X_spd, y, class_names, np.asarray(subject_labels)

    return X_spd, y, class_names
