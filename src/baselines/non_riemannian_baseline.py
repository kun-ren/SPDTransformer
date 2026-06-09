from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FILTER_BANK = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
)
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "experiments" / "splits" / "physionet_mi_seed42.json"

from src.training.shared_split import load_or_create_split_indices


def parse_filter_bank(raw: str) -> list[tuple[float, float]]:
    bands = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        low_raw, high_raw = part.split("-", 1)
        bands.append((float(low_raw), float(high_raw)))
    if not bands:
        raise argparse.ArgumentTypeError("filter bank cannot be empty")
    return bands


def build_log_variance_features(
    X: np.ndarray,
    filter_bank: list[tuple[float, float]],
    sfreq: float,
    segment_duration: float,
    stride_duration: float | None,
    eps: float,
) -> np.ndarray:
    from src.datasets.PhysioNetMI_preprocess import bandpass_filter, segment_epochs

    """
    Build Euclidean features from raw epochs.

    For each frequency band and temporal segment, compute channel-wise
    log-variance. The final feature vector is:
        frequency x segment x channel
    """
    feature_blocks = []
    for low_freq, high_freq in filter_bank:
        filtered = bandpass_filter(
            X,
            sfreq=sfreq,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        segments = segment_epochs(
            filtered,
            sfreq=sfreq,
            segment_duration=segment_duration,
            stride_duration=stride_duration,
        )
        variances = np.var(segments, axis=-1)
        log_variances = np.log(np.maximum(variances, eps))
        feature_blocks.append(log_variances)

    features = np.stack(feature_blocks, axis=1)
    return features.reshape(features.shape[0], -1).astype(np.float64)


def resolve_split_file(split_file: str | None) -> Path | None:
    if split_file in {None, ""}:
        return None
    path = Path(split_file)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_classifier(name: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced"),
        )
    raise ValueError("classifier must be 'lr' or 'svm'")


def evaluate_split(model, X: np.ndarray, y: np.ndarray, indices: np.ndarray, split: str):
    from sklearn.metrics import accuracy_score, f1_score

    prediction = model.predict(X[indices])
    return {
        "split": split,
        "n_samples": int(len(indices)),
        "accuracy": float(accuracy_score(y[indices], prediction)),
        "macro_f1": float(f1_score(y[indices], prediction, average="macro", zero_division=0)),
    }


def save_results(output_dir: Path, rows: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Non-Riemannian baseline on raw PhysioNetMI epochs.")
    parser.add_argument("--root-dir", default="data/MNE-eegbci-data/files/eegmmidb/1.0.0")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--sfreq", type=float, default=160.0)
    parser.add_argument(
        "--filter-bank",
        type=parse_filter_bank,
        default=list(DEFAULT_FILTER_BANK),
        help="Comma-separated bands, for example: 8-12,12-16,16-20,20-24,24-28,28-32",
    )
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--stride-duration", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--classifier", choices=("lr", "svm"), default="lr")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/results/non_riemannian_baseline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    from src.datasets.PhysioNetMI_preprocess import build_dataset, encode_labels

    dataset = build_dataset(args.root_dir, tmin=args.tmin, tmax=args.tmax)
    X = dataset["X"]
    y, class_names = encode_labels(dataset["y"])
    print(f"Raw X shape: {X.shape}")
    print(f"Classes: {class_names.tolist()}")

    features = build_log_variance_features(
        X=X,
        filter_bank=args.filter_bank,
        sfreq=args.sfreq,
        segment_duration=args.segment_duration,
        stride_duration=args.stride_duration,
        eps=args.eps,
    )
    print(f"Feature shape: {features.shape}")

    train_idx, val_idx, test_idx = load_or_create_split_indices(
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        split_file=resolve_split_file(args.split_file),
    )

    model = build_classifier(args.classifier)
    model.fit(features[train_idx], y[train_idx])

    rows = [
        evaluate_split(model, features, y, train_idx, "train"),
        evaluate_split(model, features, y, val_idx, "val"),
        evaluate_split(model, features, y, test_idx, "test"),
    ]
    summary = {
        "baseline": "non_riemannian",
        "classifier": args.classifier,
        "feature": "filterbank_segment_channel_log_variance",
        "class_names": class_names.tolist(),
        "raw_shape": list(X.shape),
        "feature_shape": list(features.shape),
        "filter_bank": [list(band) for band in args.filter_bank],
        "segment_duration": args.segment_duration,
        "stride_duration": args.stride_duration,
        "splits": rows,
    }

    output_dir = PROJECT_ROOT / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(output_dir, rows, summary)

    print("\nResults")
    for row in rows:
        print(row)
    print(f"\nSaved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
