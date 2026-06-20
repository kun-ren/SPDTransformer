import re

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


def extract_transition_epochs(
    edf_file,
    tmin=-3.0,
    tmax=3.8,
    low_freq=None,
    high_freq=None,
    channels=None,
    reject_threshold_uv=None,
):
    """
    take T1/T2 onset as the anchor

    [-tmin, +tmax]

    label = task type
    """

    raw = mne.io.read_raw_edf(
        edf_file,
        preload=True,
        verbose=False
    )
    # Re-reference before filtering and epoching
    raw.set_eeg_reference("average", projection=False)
    raw = pick_raw_channels(raw, channels=channels)
    raw.filter(l_freq=0.5, h_freq=None)
    raw.filter(l_freq=None, h_freq=40)
    raw.notch_filter(freqs=60)
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
            f"Rejected {rejected} bad epoch(s) from {edf_file} "
            f"(threshold={float(reject_threshold_uv):.1f} uV, "
            f"ptp_uv median={np.median(ptp_uv):.1f}, "
            f"p90={np.percentile(ptp_uv, 90):.1f}, "
            f"max={np.max(ptp_uv):.1f})."
        )

    return list(X), list(y)

from pathlib import Path


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
):
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
):
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
):
    from pyriemann.estimation import Covariances

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

        # 3. Compute covariance matrices using pyriemann
        cov_x = Covariances(estimator=estimator).fit_transform(
            temp_x.reshape(n_epochs * n_segments, n_channels, segment_samples)
        )
        cov_x = cov_x.reshape(n_epochs, n_segments, n_channels, n_channels)

        # 4. Normalize SPD scale
        cov_x = trace_normalize(cov_x, eps=eps)

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

        frequencies.append(cov_x.astype(np.float32))

    if labels is None:
        raise RuntimeError("No data was loaded. Check root_dir, subjects, and task settings.")

    # 6. Encode labels
    y, class_names = encode_labels(labels)

    # Output shape: (n_trials, segment, frequency, n_channels, n_channels)
    X_spd = np.stack(frequencies, axis=2)

    return X_spd, y, class_names
