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
    raw = pick_raw_channels(raw, channels=channels)
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
        y.append(label)

    return X, y

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
        )
        if X and y:
            X_all.append(X)
            y_all.append(y)

    return (
        np.concatenate(X_all) if X_all else None,
        np.concatenate(y_all) if X_all else None
    )

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
        )
        if X is None:
            continue
        X = np.array(X)
        y = np.array(y)

        X_all.append(X)
        y_all.append(y)

        print(f"Subject {subject_dir}: {X.shape}")

        subject_labels.extend(
            [subject_dir.name] * len(y)
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
):
    from pyriemann.estimation import Covariances

    frequencies = []
    labels = None
    subject_labels = None


    for filter in filter_bank:

        # 1. Filter continuous Raw first, then extract epochs.
        dataset = build_dataset(
            root_dir,
            tmin=-2.0,
            tmax=4.0,
            subjects=subjects,
            imaged=imaged,
            task_types=task_types,
            executed=executed,
            low_freq=filter[0],
            high_freq=filter[1],
            channels=channels,
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

        frequencies.append(cov_x.astype(np.float32))

    if labels is None:
        raise RuntimeError("No data was loaded. Check root_dir, subjects, and task settings.")

    # 6. Encode labels
    y, class_names = encode_labels(labels)

    # Output shape: (n_trials, segment, frequency, n_channels, n_channels)
    X_spd = np.stack(frequencies, axis=2)

    return X_spd, y, class_names
