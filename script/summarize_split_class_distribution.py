from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    build_dataset,
    config_hash,
    encode_labels,
    expand_data_training_experiments,
    get_split_indices,
    load_yaml,
    normalize_filter_bank,
    parse_subjects,
    parse_task_types,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize class counts for train/val/test splits using the same "
            "data and split configuration as train.py."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        default="experiments/results/split_class_distribution",
    )
    return parser


def load_labels_like_train(
    data_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    first_band = filter_bank[0]

    dataset = build_dataset(
        root_dir=str(
            data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")
        ),
        tmin=-2.0,
        tmax=4.0,
        subjects=parse_subjects(data_cfg.get("subjects")),
        imaged=data_cfg.get("imaged", True),
        executed=data_cfg.get("executed", False),
        task_types=parse_task_types(data_cfg.get("task_types")),
        low_freq=first_band[0],
        high_freq=first_band[1],
        channels=data_cfg.get("channels"),
    )
    y, class_names = encode_labels(dataset["y"])
    return y.astype(np.int64), list(class_names), np.asarray(dataset["subject"])


def split_distribution_rows(
    y: np.ndarray,
    class_names: list[str],
    split_indices: dict[str, np.ndarray],
    run_index: int,
    experiment_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split_name, indices in split_indices.items():
        split_y = y[indices]
        counts = np.bincount(split_y, minlength=len(class_names))
        split_total = int(len(indices))
        for class_index, class_name in enumerate(class_names):
            count = int(counts[class_index])
            rows.append(
                {
                    "run_index": run_index,
                    "config_hash": config_hash(experiment_cfg),
                    "split": split_name,
                    "class": class_name,
                    "count": count,
                    "split_total": split_total,
                    "fraction": count / split_total if split_total else 0.0,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    if not experiments:
        raise ValueError("No data/training configurations were generated.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cache: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    all_rows = []
    summaries = []
    for run_index, experiment_cfg in enumerate(experiments, start=1):
        data_cfg = experiment_cfg["data"]
        training_cfg = experiment_cfg["training"]
        data_key = config_hash({"data": data_cfg})
        if data_key not in data_cache:
            data_cache[data_key] = load_labels_like_train(data_cfg)

        y, class_names, subject_labels = data_cache[data_key]
        train_idx, val_idx, test_idx = get_split_indices(
            y,
            training_cfg,
            subject_labels=subject_labels,
        )
        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        rows = split_distribution_rows(
            y=y,
            class_names=class_names,
            split_indices=split_indices,
            run_index=run_index,
            experiment_cfg=experiment_cfg,
        )
        all_rows.extend(rows)
        summaries.append(
            {
                "run_index": run_index,
                "config_hash": config_hash(experiment_cfg),
                "class_names": class_names,
                "n_samples": int(len(y)),
                "n_subjects": int(len(set(subject_labels.tolist()))),
                "split_sizes": {
                    split_name: int(len(indices))
                    for split_name, indices in split_indices.items()
                },
                "config": experiment_cfg,
            }
        )

    write_csv(output_dir / "split_class_distribution.csv", all_rows)
    save_json(output_dir / "summary.json", summaries)
    print(f"Saved split class distribution to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
