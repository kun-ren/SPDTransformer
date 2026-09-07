from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    config_hash,
    expand_data_training_experiments,
    load_segmented_epochs_like_train,
    load_yaml,
    make_subject_specific_loro_splits,
    parse_bool,
    save_json,
    save_fold_predictions,
    summarize_subject_fold_metrics,
)


METRIC_NAMES = ("accuracy", "macro_f1", "cohen_kappa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CSP + shrinkage LDA baseline with optional subject-specific "
            "leave-one-run-out train/validation/test evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Override model.n_components from the YAML config.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def compute_csp_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def build_cv_splits(
    y: np.ndarray,
    subject_labels: np.ndarray,
    n_splits: int,
    seed: int,
    allow_subject_overlap: bool,
    subject_fold_method: str = "stratified_group",
    cv_shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create 80/20-style folds without a separate validation dataset."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    indices = np.arange(len(y))
    if not allow_subject_overlap and subject_fold_method == "kfold":
        from src.training.train_global_cross_subject import make_subject_folds
        return [(fold[2], fold[3]) for fold in make_subject_folds(
            subject_labels, n_splits, shuffle=cv_shuffle, seed=seed,
        )]
    if subject_fold_method not in {"kfold", "stratified_group"}:
        raise ValueError(f"Unknown subject_fold_method: {subject_fold_method}")
    if allow_subject_overlap:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(indices, y)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(indices, y, groups=subject_labels)
    return [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )
        for train_idx, test_idx in iterator
    ]


def validate_cv_config(
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
) -> tuple[int, int, bool, float]:
    n_splits = int(training_cfg.get("n_splits", 5))
    seed = int(training_cfg.get("seed", 42))
    test_size = float(data_cfg.get("test_size", training_cfg.get("test_size", 0.2)))
    val_size = float(data_cfg.get("val_size", training_cfg.get("val_size", 0.0)))
    allow_subject_overlap = parse_bool(
        data_cfg.get(
            "allow_subject_overlap",
            training_cfg.get("allow_subject_overlap", True),
        ),
        default=True,
    )

    if n_splits < 2:
        raise ValueError(f"training.n_splits must be at least 2, got {n_splits}.")
    subject_specific = parse_bool(
        training_cfg.get("subject_specific", False), default=False
    )
    held_out_run_validation_size = float(
        training_cfg.get("held_out_run_validation_size", 0.5)
    )
    if subject_specific and not 0.0 < held_out_run_validation_size < 1.0:
        raise ValueError("held_out_run_validation_size must be between 0 and 1.")
    if not subject_specific and not np.isclose(val_size, 0.0):
        raise ValueError(
            "CSP+LDA K-fold evaluation does not use a validation dataset; "
            f"set val_size to 0.0, got {val_size}."
        )
    expected_test_size = 1.0 / n_splits
    if not subject_specific and not np.isclose(test_size, expected_test_size):
        raise ValueError(
            "For K-fold evaluation, test_size must equal 1 / n_splits; "
            f"got test_size={test_size} and n_splits={n_splits} "
            f"(expected {expected_test_size:.6f})."
        )
    return n_splits, seed, allow_subject_overlap, held_out_run_validation_size


class SegmentedFilterBankCSP(TransformerMixin, BaseEstimator):
    """Fit one CSP per band on training windows, then mean window features."""

    def __init__(self, csp):
        self.csp = csp

    def fit(self, x, y):
        if x.ndim != 5 or len(x) != len(y):
            raise ValueError("Expected aligned (trial, segment, band, channel, sample) input.")
        self.n_bands_ = x.shape[2]
        self.csps_ = []
        for band in range(self.n_bands_):
            windows = x[:, :, band].reshape(-1, *x.shape[-2:]).astype(np.float64)
            csp = clone(self.csp)
            csp.fit(windows, np.repeat(y, x.shape[1]))
            self.csps_.append(csp)
        return self

    def transform(self, x):
        if x.ndim != 5 or x.shape[2] != self.n_bands_:
            raise ValueError("Segmented CSP input band count differs from training.")
        features = []
        for band, csp in enumerate(self.csps_):
            windows = x[:, :, band].reshape(-1, *x.shape[-2:]).astype(np.float64)
            features.append(csp.transform(windows).reshape(len(x), x.shape[1], -1).mean(axis=1))
        return np.concatenate(features, axis=-1)


def build_csp_lda_pipeline(
    n_components: int,
    model_cfg: dict[str, Any] | None = None,
):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    model_cfg = model_cfg or {}
    csp_reg = model_cfg.get("csp_reg", "ledoit_wolf")
    if isinstance(csp_reg, str) and csp_reg.strip().lower() in {
        "",
        "none",
        "null",
    }:
        csp_reg = None
    lda_solver = str(model_cfg.get("lda_solver", "svd")).strip().lower()
    lda_shrinkage = model_cfg.get("lda_shrinkage")
    if isinstance(lda_shrinkage, str) and lda_shrinkage.strip().lower() in {
        "",
        "none",
        "null",
    }:
        lda_shrinkage = None
    if lda_solver == "svd" and lda_shrinkage is not None:
        raise ValueError("LDA solver='svd' does not support shrinkage.")
    csp = CSP(
        n_components=n_components,
        reg=csp_reg,
        log=parse_bool(model_cfg.get("csp_log", True), default=True),
        norm_trace=parse_bool(
            model_cfg.get("csp_norm_trace", False), default=False
        ),
    )
    lda = LinearDiscriminantAnalysis(
        solver=lda_solver,
        shrinkage=lda_shrinkage,
    )
    input_mode = model_cfg.get("input_mode", "original")
    if input_mode not in {"original", "segmented_filterbank"}:
        raise ValueError("CSP input_mode must be original or segmented_filterbank.")
    if input_mode == "segmented_filterbank":
        csp = SegmentedFilterBankCSP(csp)
    return Pipeline(
        [
            ("csp", csp),
            ("lda", lda),
        ]
    )


def aggregate_fold_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray([row[metric_name] for row in rows], dtype=float)
        aggregates[metric_name] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
        }
    return aggregates


def print_fold_summary(rows: list[dict[str, Any]], aggregates: dict[str, dict[str, float]]) -> None:
    print("\nCSP + LDA five-fold test results (no validation dataset)")
    print("fold | train | test | accuracy | macro_f1 | Cohen's kappa")
    for row in rows:
        print(
            f"{row['fold']:>4} | {row['n_train']:>5} | {row['n_test']:>4} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['cohen_kappa']:.4f}"
        )
    print("\nFive-fold aggregate (test folds)")
    print("metric        | mean   | max    | min")
    for metric_name, display_name in (
        ("accuracy", "accuracy"),
        ("macro_f1", "macro_f1"),
        ("cohen_kappa", "Cohen's kappa"),
    ):
        stats = aggregates[metric_name]
        print(
            f"{display_name:<13} | {stats['mean']:.4f} | "
            f"{stats['max']:.4f} | {stats['min']:.4f}"
        )


def run_experiment(
    run_index: int,
    experiment_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    n_components_override: int | None,
    base_output_dir: Path,
) -> dict[str, Any]:
    import mne

    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]
    mne.set_log_level(str(model_cfg.get("mne_log_level", "WARNING")))
    (
        n_splits,
        seed,
        allow_subject_overlap,
        held_out_run_validation_size,
    ) = validate_cv_config(training_cfg, data_cfg)

    subject_specific = parse_bool(
        training_cfg.get("subject_specific", False), default=False
    )
    if subject_specific:
        (
            x,
            y,
            subject_labels,
            run_labels,
            class_names,
            filter_bank,
        ) = load_segmented_epochs_like_train(data_cfg, return_runs=True)
        fold_specs = make_subject_specific_loro_splits(
            y,
            subject_labels,
            run_labels,
            held_out_run_validation_size=held_out_run_validation_size,
            seed=seed,
        )
    else:
        x, y, subject_labels, class_names, filter_bank = (
            load_segmented_epochs_like_train(data_cfg)
        )
        run_labels = None
        fold_specs = [
            (
                "all",
                fold_index,
                train_idx,
                np.empty(0, dtype=np.int64),
                test_idx,
            )
            for fold_index, (train_idx, test_idx) in enumerate(
                build_cv_splits(
                    y,
                    subject_labels,
                    n_splits=n_splits,
                    seed=seed,
                    allow_subject_overlap=allow_subject_overlap,
                    subject_fold_method=training_cfg.get("subject_fold_method", "stratified_group"),
                    cv_shuffle=parse_bool(training_cfg.get("cv_shuffle", True)),
                ),
                start=1,
            )
        ]

    n_segments = x.shape[1]
    n_bands = x.shape[2]
    n_channels = x.shape[3]
    segmented_filterbank = model_cfg.get("input_mode", "original") == "segmented_filterbank"
    if not segmented_filterbank and (n_segments != 1 or n_bands != 1):
        raise ValueError(
            "The requested CSP -> LDA Pipeline requires exactly one segment "
            "and one frequency band per trial; got "
            f"n_segments={n_segments}, n_bands={n_bands}."
        )
    trial_data = x if segmented_filterbank else x[:, 0, 0]
    requested_components = (
        n_components_override
        if n_components_override is not None
        else int(model_cfg.get("n_components", 6))
    )
    if requested_components < 1:
        raise ValueError(
            f"model.n_components must be positive, got {requested_components}."
        )
    csp_components = min(requested_components, n_channels)

    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for subject, held_out_run, train_idx, validation_idx, test_idx in fold_specs:
        split_rows.append(
            {
                "subject": subject if subject_specific else None,
                "test_run": int(held_out_run) if subject_specific else None,
                "fold": None if subject_specific else held_out_run,
                "train_indices": train_idx.astype(int).tolist(),
                "validation_indices": validation_idx.astype(int).tolist(),
                "test_indices": test_idx.astype(int).tolist(),
            }
        )
        classifier = build_csp_lda_pipeline(csp_components, model_cfg)
        classifier.fit(trial_data[train_idx], y[train_idx])
        validation_prediction = (
            classifier.predict(trial_data[validation_idx])
            if subject_specific
            else None
        )
        test_prediction = classifier.predict(trial_data[test_idx])

        row = {
            **({"subject": subject} if subject_specific else {}),
            **(
                {"test_run": int(held_out_run)}
                if subject_specific
                else {"fold": held_out_run}
            ),
            "n_train": int(len(train_idx)),
            "n_validation": int(len(validation_idx)),
            "n_test": int(len(test_idx)),
        }
        if validation_prediction is not None:
            validation_metrics = compute_csp_metrics(
                y[validation_idx], validation_prediction
            )
            row.update(
                {
                    f"validation_{name}": value
                    for name, value in validation_metrics.items()
                }
            )
        row.update(compute_csp_metrics(y[test_idx], test_prediction))
        row["_y_true"] = y[test_idx].astype(int).tolist()
        row["_y_pred"] = np.asarray(test_prediction, dtype=int).tolist()
        row["_subject_labels"] = subject_labels[test_idx].astype(str).tolist()
        rows.append(row)
        print(
            f"[CSP+LDA {'subject ' + subject + ' ' if subject_specific else ''}"
            + (
                f"test run R{int(held_out_run):02d}] "
                if subject_specific
                else f"fold {held_out_run}/{n_splits}] "
            )
            + (
                f"validation_accuracy={row['validation_accuracy']:.4f} "
                f"validation_mf1={row['validation_macro_f1']:.4f} | "
                if subject_specific
                else ""
            )
            + f"test_accuracy={row['accuracy']:.4f} "
            f"test_mf1={row['macro_f1']:.4f} "
            f"test_kappa={row['cohen_kappa']:.4f}"
        )

    aggregates = aggregate_fold_metrics(rows)
    subject_rows = summarize_subject_fold_metrics(
        rows,
        (
            "validation_accuracy",
            "validation_macro_f1",
            "validation_cohen_kappa",
            *METRIC_NAMES,
        ),
    )
    if subject_specific:
        print("\nCSP+LDA pooled subject-specific test results")
        for row in subject_rows:
            print(
                f"  {row['Subject']}: trials={row['Trials']} "
                f"accuracy={row['Accuracy (%)'] / 100.0:.4f} "
                f"balanced_accuracy={row['Balanced Accuracy (%)'] / 100.0:.4f} "
                f"macro_f1={row['Macro-F1']:.4f} "
                f"kappa={row['Cohen’s κ']:.4f}"
            )
    else:
        print_fold_summary(rows, aggregates)

    effective_cfg = dict(experiment_cfg)
    effective_cfg["model"] = dict(model_cfg)
    effective_cfg["model"]["n_components"] = csp_components
    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(effective_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    with (run_dir / "fold_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    if subject_specific:
        with (run_dir / "per_run_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
            writer.writeheader()
            writer.writerows(public_rows)
    with (run_dir / "per_subject_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)
    save_json(run_dir / "splits.json", split_rows)
    save_fold_predictions(run_dir / "test_predictions.csv", fold_specs, rows, subject_labels, run_labels)

    summary = {
        "baseline": "csp_lda",
        "input_mode": model_cfg.get("input_mode", "original"),
        "segment_feature_pooling": "mean" if segmented_filterbank else None,
        "evaluation": {
            "strategy": (
                "subject_specific_leave_one_run_out"
                if subject_specific
                else (
                    "stratified_kfold"
                    if allow_subject_overlap
                    else ("subject_kfold" if training_cfg.get("subject_fold_method") == "kfold" else "stratified_group_kfold")
                )
            ),
            "n_splits": None if subject_specific else n_splits,
            "test_size_per_fold": None if subject_specific else 1.0 / n_splits,
            "held_out_run_validation_size": (
                held_out_run_validation_size if subject_specific else None
            ),
            "seed": seed,
        },
        "config": effective_cfg,
        "class_names": class_names,
        "filter_bank": filter_bank,
        "x_shape": list(x.shape),
        "n_components": csp_components,
        "feature_dim": csp_components * n_bands if segmented_filterbank else csp_components,
        "pipeline_steps": ["csp", "lda"],
        "folds": public_rows,
        "runs": public_rows if subject_specific else [],
        "splits_file": "splits.json",
        "subjects": subject_rows,
        "aggregates": aggregates,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[CSP+LDA run {run_index}] saved {run_dir}")
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "aggregates": aggregates,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    model_cfg = dict(config.get("model", {}))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/csp_lda_baseline",
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = [
        run_experiment(
            index,
            experiment,
            model_cfg,
            args.n_components,
            base_output_dir,
        )
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All CSP+LDA runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
