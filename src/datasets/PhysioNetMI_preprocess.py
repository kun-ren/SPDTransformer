import re

import mne

import numpy as np
from pyriemann.estimation import Covariances
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import LabelEncoder


def bandpass_filter(X, sfreq=160.0, low_freq=8.0, high_freq=30.0, order=4):
    """
    X shape: (n_trials, n_channels, n_times)
    """
    nyquist = sfreq / 2.0

    sos = butter(
        order,
        [low_freq / nyquist, high_freq / nyquist],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, X, axis=-1)


def filter_classes(X, labels, metadata=None, keep_classes=None):
    """
    keep_classes example:
        ["left_hand", "right_hand"]
        ["left_hand", "right_hand", "feet", "hands"]
    """
    if keep_classes is None:
        keep_classes = ["left_hand", "right_hand", "feet", "hands"]

    labels = np.asarray(labels).astype(str)

    keep = np.isin(labels, keep_classes)

    X = X[keep]
    labels = labels[keep]

    if metadata is not None:
        metadata = metadata.iloc[keep].reset_index(drop=True)

    return X, labels, metadata



def trace_normalize(covs, eps=1e-10):
    """
    covs: shape (..., n_channels, n_channels)
    return: each matrix divided by its own trace
    """
    traces = np.trace(covs, axis1=-2, axis2=-1)
    traces = np.maximum(traces, eps)
    return covs / traces[..., None, None]


def regularize_spd(covs, eps=1e-10):
    n_channels = covs.shape[-1]
    eye = np.eye(n_channels, dtype=covs.dtype)
    return covs + eps * eye


def encode_labels(labels):
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
    tmin=-2.0,
    tmax=4.0,
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

    sfreq = raw.info["sfreq"]

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

    return np.array(X), np.array(y)

from pathlib import Path


def load_subject(
    subject_dir,
    tmin=-2.0,
    tmax=4.0
):
    X_all = []
    y_all = []

    for edf_file in Path(subject_dir).glob("*.edf"):

        run_id = int(
            re.search(
                r"R(\d+)",
                edf_file.name
            ).group(1)
        )

        if run_id not in {4, 6, 8, 10, 12, 14}:
            continue

        X, y = extract_transition_epochs(
            str(edf_file),
            tmin,
            tmax
        )

        X_all.append(X)
        y_all.append(y)

    return (
        np.concatenate(X_all),
        np.concatenate(y_all)
    )

def build_dataset(
    root_dir,
    tmin=-2.0,
    tmax=4.0,
):
    X_all = []
    y_all = []
    subjects = []

    for subject_dir in sorted(
        Path(root_dir).glob("S*")
    ):

        X, y = load_subject(
            subject_dir,
            tmin,
            tmax
        )

        X_all.append(X)
        y_all.append(y)

        subjects.extend(
            [subject_dir.name] * len(y)
        )

    return {
        "X": np.concatenate(X_all),
        "y": np.concatenate(y_all),
        "subject": np.array(subjects),
    }


def preprocess_spd(
    filter_bank,
    root_dir="data/MNE-eegbci-data/files/eegmmidb/1.0.0",
    estimator="cov",
    eps=1e-10,
    sfreq=160,
    segment_duration=0.75,
    stride_duration=None,
):

    dataset = build_dataset(root_dir, tmin=-2.0, tmax=4.0)
    X = dataset['X']   #[n_epochs, n_channels, n_samples_per_epoch]
    print(f"raw X shape: {X.shape}")
    labels = dataset['y']

    frequencies = []


    for filter in filter_bank:

        # 2. Band-pass filter
        temp_x = bandpass_filter(
            X,
            sfreq=sfreq,
            low_freq=filter[0],
            high_freq=filter[1],
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
            f"(segment_samples={segment_samples})"
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

        sample_epoch = min(100, cov_x.shape[0] - 1)
        sample_segment = 0
        sample_cov = cov_x[sample_epoch, sample_segment]
        frequencies.append(cov_x.astype(np.float32))

    # 6. Encode labels
    y, class_names = encode_labels(labels)

    X_spd = np.stack(frequencies, axis=2)

    return X_spd, y, class_names
