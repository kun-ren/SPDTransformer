from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    compute_metrics,
    config_hash,
    expand_data_training_experiments,
    get_split_indices,
    load_spd_like_train,
    load_yaml,
    log_euclidean_token_mean,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MDM baseline using the same SPD preprocessing config as train.py."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metric", default="riemann")
    parser.add_argument("--output-dir", default="experiments/results/mdm_baseline")
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
    train_idx, val_idx, test_idx = get_split_indices(
        y,
        training_cfg,
        subject_labels=subject_labels,
    )

    classifier = MDM(metric=metric)
    classifier.fit(x_trial_spd[train_idx], y[train_idx])

    split_indices = {
        "train": train_idx,
        "val": val_idx,
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
    base_output_dir = PROJECT_ROOT / args.output_dir / timestamp
    all_metrics = [
        run_experiment(index, experiment, args.metric, base_output_dir)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All MDM runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
