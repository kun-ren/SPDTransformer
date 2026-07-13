from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    compute_metrics,
    config_hash,
    expand_data_training_experiments,
    load_spd_like_train,
    load_yaml,
    log_euclidean_token_mean,
    parse_bool,
    resolve_split_file,
    save_json,
)


def _labels_hash(y: np.ndarray) -> str:
    labels = np.asarray(y, dtype=np.int64)
    return hashlib.sha1(labels.tobytes()).hexdigest()


def _subjects_hash(subjects: np.ndarray | None) -> str | None:
    if subjects is None:
        return None
    subject_labels = np.asarray(subjects, dtype=np.str_)
    return hashlib.sha1("\n".join(subject_labels.tolist()).encode("utf-8")).hexdigest()


def create_train_test_indices(
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    return train_idx, test_idx


def create_subject_train_test_indices(
    y: np.ndarray,
    subjects: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.str_)
    if len(subjects) != len(y):
        raise ValueError(
            f"subjects length ({len(subjects)}) must match labels length ({len(y)})."
        )

    indices = np.arange(len(y))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(indices, y, groups=subjects))
    return train_idx, test_idx


def get_train_test_indices(
    y: np.ndarray,
    training_cfg: dict,
    subject_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    test_size = float(training_cfg.get("test_size", 0.2))
    seed = int(training_cfg.get("seed", 42))
    split_file = resolve_split_file(training_cfg.get("split_file"))
    allow_subject_overlap = parse_bool(
        training_cfg.get("allow_subject_overlap", True),
        default=True,
    )
    split_strategy = "epoch" if allow_subject_overlap else "subject"
    if not allow_subject_overlap and subject_labels is None:
        raise ValueError("subject_labels must be provided for subject-level splits.")

    if split_file is not None and split_file.exists():
        with split_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if payload.get("split_kind") != "train_test":
            raise ValueError(
                f"Split file {split_file} is not a train/test MDM split. "
                "Use a different split_file or delete the old split file."
            )
        expected_subjects_hash = _subjects_hash(subject_labels)
        if payload.get("n_samples") != len(y):
            raise ValueError(
                f"Split file {split_file} was built for {payload.get('n_samples')} "
                f"samples, but current data has {len(y)} samples."
            )
        if payload.get("labels_hash") != _labels_hash(y):
            raise ValueError(
                f"Split file {split_file} does not match the current label order."
            )
        if payload.get("split_strategy") != split_strategy:
            raise ValueError(
                f"Split file {split_file} uses "
                f"split_strategy={payload.get('split_strategy')!r}, but current "
                f"config requests {split_strategy!r}."
            )
        if (
            split_strategy == "subject"
            and payload.get("subjects_hash") != expected_subjects_hash
        ):
            raise ValueError(
                f"Split file {split_file} does not match the current subject order."
            )
        if abs(float(payload.get("test_size", test_size)) - test_size) > 1e-12:
            raise ValueError(
                f"Split file {split_file} uses test_size={payload.get('test_size')}, "
                f"but current config requests test_size={test_size}."
            )
        return (
            np.asarray(payload["train_idx"], dtype=np.int64),
            np.asarray(payload["test_idx"], dtype=np.int64),
        )

    if allow_subject_overlap:
        train_idx, test_idx = create_train_test_indices(
            y=y,
            test_size=test_size,
            seed=seed,
        )
    else:
        train_idx, test_idx = create_subject_train_test_indices(
            y=y,
            subjects=subject_labels,
            test_size=test_size,
            seed=seed,
        )

    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split_kind": "train_test",
            "n_samples": int(len(y)),
            "labels_hash": _labels_hash(y),
            "subjects_hash": _subjects_hash(subject_labels),
            "split_strategy": split_strategy,
            "allow_subject_overlap": bool(allow_subject_overlap),
            "seed": seed,
            "test_size": test_size,
            "train_idx": train_idx.astype(int).tolist(),
            "test_idx": test_idx.astype(int).tolist(),
        }
        with split_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return train_idx, test_idx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MDM baseline using the same SPD preprocessing config as train.py."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metric", default="riemann")
    parser.add_argument("--output-dir", default=None)
    return parser


def run_experiment(
    run_index: int,
    experiment_cfg: dict,
    metric: str,
    base_output_dir: Path,
) -> dict:
    from pyriemann.classification import MDM

    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]

    x_spd, y, subject_labels, class_names = load_spd_like_train(data_cfg)
    x_trial_spd = log_euclidean_token_mean(x_spd)
    train_idx, test_idx = get_train_test_indices(
        y,
        training_cfg,
        subject_labels=subject_labels,
    )

    classifier = MDM(metric=metric)
    classifier.fit(x_trial_spd[train_idx], y[train_idx])

    split_indices = {
        "train": train_idx,
        "test": test_idx,
    }
    rows = []
    for split_name, split_idx in split_indices.items():
        prediction = classifier.predict(x_trial_spd[split_idx])
        row = {
            "split": split_name,
            "n_samples": int(len(split_idx)),
        }
        row.update(compute_metrics(y[split_idx], prediction))
        rows.append(row)

    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "baseline": "mdm",
        "metric": metric,
        "config": experiment_cfg,
        "class_names": class_names,
        "x_spd_shape": list(x_spd.shape),
        "x_trial_spd_shape": list(x_trial_spd.shape),
        "token_pooling": "log_euclidean_mean_over_segment_and_frequency",
        "splits": rows,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[MDM run {run_index}] saved {run_dir}")
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "test_accuracy": rows[-1]["accuracy"],
        "test_macro_f1": rows[-1]["macro_f1"],
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/mdm_baseline",
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = [
        run_experiment(index, experiment, args.metric, base_output_dir)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All MDM runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
