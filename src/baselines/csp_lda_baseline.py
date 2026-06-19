from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    compute_metrics,
    config_hash,
    expand_data_training_experiments,
    get_split_indices,
    load_segmented_epochs_like_train,
    load_yaml,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSP + LDA baseline using the same data config as train.py."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--output-dir", default="experiments/results/csp_lda_baseline")
    return parser


def csp_features_for_split(
    csp_models,
    x: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    # x shape: (n_trials, n_segments, n_bands, n_channels, n_samples)
    n_trials = len(indices)
    n_segments = x.shape[1]
    features = []
    for band_index, csp in enumerate(csp_models):
        band_windows = x[indices, :, band_index]
        n_channels, n_samples = band_windows.shape[-2:]
        flat_windows = band_windows.reshape(n_trials * n_segments, n_channels, n_samples)
        flat_features = csp.transform(flat_windows)
        trial_features = flat_features.reshape(n_trials, n_segments, -1).mean(axis=1)
        features.append(trial_features)
    return np.concatenate(features, axis=1)


def run_experiment(
    run_index: int,
    experiment_cfg: dict,
    n_components: int,
    base_output_dir: Path,
) -> dict:
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]

    x, y, class_names, filter_bank = load_segmented_epochs_like_train(data_cfg)
    train_idx, val_idx, test_idx = get_split_indices(y, training_cfg)
    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    n_segments = x.shape[1]
    n_bands = x.shape[2]
    n_channels = x.shape[3]
    csp_components = min(int(n_components), n_channels)

    csp_models = []
    y_train_repeated = np.repeat(y[train_idx], n_segments)
    for band_index in range(n_bands):
        train_band = x[train_idx, :, band_index]
        n_train = len(train_idx)
        n_samples = train_band.shape[-1]
        flat_train = train_band.reshape(n_train * n_segments, n_channels, n_samples)
        csp = CSP(
            n_components=csp_components,
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
        )
        csp.fit(flat_train, y_train_repeated)
        csp_models.append(csp)

    x_train_features = csp_features_for_split(csp_models, x, train_idx)
    x_val_features = csp_features_for_split(csp_models, x, val_idx)
    x_test_features = csp_features_for_split(csp_models, x, test_idx)

    classifier = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(),
    )
    classifier.fit(x_train_features, y[train_idx])

    predictions = {
        "train": classifier.predict(x_train_features),
        "val": classifier.predict(x_val_features),
        "test": classifier.predict(x_test_features),
    }
    rows = []
    for split_name, split_idx in split_indices.items():
        row = {
            "split": split_name,
            "n_samples": int(len(split_idx)),
        }
        row.update(compute_metrics(y[split_idx], predictions[split_name]))
        rows.append(row)

    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "baseline": "csp_lda",
        "config": experiment_cfg,
        "class_names": class_names,
        "filter_bank": filter_bank,
        "x_shape": list(x.shape),
        "n_components": csp_components,
        "feature_dim": int(x_train_features.shape[1]),
        "splits": rows,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[CSP+LDA run {run_index}] saved {run_dir}")
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
        run_experiment(index, experiment, args.n_components, base_output_dir)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All CSP+LDA runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
