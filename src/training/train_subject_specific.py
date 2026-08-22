"""Train and evaluate one independent SPD Transformer per PhysioNet subject.

Each subject receives an outer stratified cross-validation loop.  A validation
set is drawn only from the outer training fold and controls early stopping and
learning-rate scheduling.  The held-out test fold is evaluated exactly once
after the best validation checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.PhysioNetMI_subject_specific import (  # noqa: E402
    SubjectSpecificDataset,
    load_subject_specific_datasets,
)
from src.models.MotorImageryDataset import MotorImageryDataset  # noqa: E402
from src.training.train import (  # noqa: E402
    build_model,
    evaluate,
    normalize_data_preprocessing_config,
    normalize_precision_name,
    parse_bool,
    predict_loader,
    resolve_precision,
    save_confusion_matrices,
    save_per_class_metrics,
    set_seed,
    split_params,
    train_one_epoch,
)


def parse_subject_spec(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(part) for part in value]
    else:
        if str(value).strip().lower() in {"", "all"}:
            return None
        parts = str(value).split(",")
    subjects: set[int] = set()
    for raw_part in parts:
        part = raw_part.strip().upper().removeprefix("S")
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid subject range: {raw_part!r}.")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(part))
    return sorted(subjects)


def make_subject_folds(
    y: np.ndarray,
    *,
    outer_splits: int,
    validation_size: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create train/validation/test indices for one subject."""

    y = np.asarray(y, dtype=np.int64)
    if outer_splits < 2:
        raise ValueError("outer_splits must be at least 2.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    counts = np.bincount(y)
    if len(counts) == 0 or int(counts.min()) < outer_splits:
        raise ValueError(
            "Every class needs at least outer_splits trials; "
            f"class counts are {counts.tolist()}, outer_splits={outer_splits}."
        )

    outer = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )
    indices = np.arange(len(y))
    folds = []
    for fold_index, (train_val_idx, test_idx) in enumerate(
        outer.split(indices, y),
        start=1,
    ):
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed + fold_index,
        )
        train_rel, val_rel = next(
            inner.split(train_val_idx, y[train_val_idx])
        )
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
        if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
            raise RuntimeError("Subject-specific train/validation/test overlap detected.")
        folds.append(
            (
                train_idx.astype(np.int64),
                val_idx.astype(np.int64),
                test_idx.astype(np.int64),
            )
        )
    return folds


def make_loader(
    dataset: SubjectSpecificDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    dtype: torch.dtype,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        MotorImageryDataset(dataset.x[indices], dataset.y[indices], dtype=dtype),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "std_between_subjects": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "q1": float(np.quantile(data, 0.25)),
        "q3": float(np.quantile(data, 0.75)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def train_subject_fold(
    dataset: SubjectSpecificDataset,
    fold_index: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    fold_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(training_cfg.get("seed", 42)) + fold_index - 1
    set_seed(seed)
    precision = normalize_precision_name(training_cfg.get("precision", "float32"))
    dtype = resolve_precision(precision)
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = parse_bool(
        training_cfg.get("pin_memory", device.type == "cuda"),
        default=device.type == "cuda",
    )
    loaders = {
        "train": make_loader(
            dataset,
            train_idx,
            batch_size=batch_size,
            dtype=dtype,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "validation": make_loader(
            dataset,
            val_idx,
            batch_size=batch_size,
            dtype=dtype,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            dataset,
            test_idx,
            batch_size=batch_size,
            dtype=dtype,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    x = dataset.x
    model = build_model(
        model_cfg=copy.deepcopy(model_cfg),
        spd_in_dim=x.shape[-1],
        num_classes=len(dataset.class_names),
        time_sequence_length=x.shape[1],
        frequency_sequence_length=x.shape[2] if x.ndim >= 5 else 1,
        brain_region_sequence_length=x.shape[3] if x.ndim >= 6 else 1,
    ).to(device=device, dtype=dtype)
    criterion = nn.CrossEntropyLoss()
    stiefel_params, decay_params, no_decay_params = split_params(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
            },
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(training_cfg.get("learning_rate", 1e-3)),
    )
    optimizer_stiefel = None
    if stiefel_params:
        import geoopt

        optimizer_stiefel = geoopt.optim.RiemannianAdam(
            stiefel_params,
            lr=float(training_cfg.get("stiefel_learning_rate", 3e-4)),
            weight_decay=0.0,
            stabilize=10,
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(training_cfg.get("lr_scheduler_patience", 8)),
        threshold=float(training_cfg.get("lr_scheduler_threshold", 1e-4)),
        threshold_mode="abs",
        min_lr=float(training_cfg.get("lr_scheduler_min_lr", 1e-5)),
    )
    stiefel_scheduler = None
    if optimizer_stiefel is not None:
        stiefel_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_stiefel,
            mode="max",
            factor=float(training_cfg.get("lr_scheduler_factor", 0.5)),
            patience=int(training_cfg.get("lr_scheduler_patience", 8)),
            threshold=float(training_cfg.get("lr_scheduler_threshold", 1e-4)),
            threshold_mode="abs",
            min_lr=float(training_cfg.get("stiefel_lr_scheduler_min_lr", 1e-6)),
        )

    epochs = int(training_cfg.get("epochs", 100))
    patience = int(training_cfg.get("early_stopping_patience", 12))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 1e-4))
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    condition_weight = float(training_cfg.get("condition_regularization_weight", 0.0))
    best_val_macro_f1 = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            optimizer_stiefel,
            device,
            gradient_clip_norm=gradient_clip_norm,
            condition_regularization_weight=condition_weight,
        )
        val_metrics = evaluate(
            model,
            loaders["validation"],
            criterion,
            device,
            condition_regularization_weight=condition_weight,
        )
        scheduler.step(val_metrics["macro_f1"])
        if stiefel_scheduler is not None:
            stiefel_scheduler.step(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["macro_f1"] > best_val_macro_f1 + min_delta:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced.")
    model.load_state_dict(best_state)
    validation_predictions = predict_loader(
        model,
        loaders["validation"],
        criterion,
        device,
        condition_regularization_weight=condition_weight,
    )
    test_predictions = predict_loader(
        model,
        loaders["test"],
        criterion,
        device,
        condition_regularization_weight=condition_weight,
    )

    _write_csv(fold_dir / "history.csv", history)
    split_payload = {
        "subject_id": dataset.subject_id,
        "fold": fold_index,
        "seed": seed,
        "train_idx": train_idx.astype(int).tolist(),
        "validation_idx": val_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
    }
    with (fold_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)
    split_predictions = {
        "validation": validation_predictions,
        "test": test_predictions,
    }
    save_per_class_metrics(
        fold_dir / "per_class_metrics.csv",
        split_predictions,
        list(dataset.class_names),
    )
    save_confusion_matrices(
        fold_dir / "confusion_matrices.csv",
        split_predictions,
        list(dataset.class_names),
    )
    if parse_bool(training_cfg.get("save_checkpoints", False), default=False):
        torch.save(
            {
                "model_state_dict": best_state,
                "subject_id": dataset.subject_id,
                "fold": fold_index,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_val_macro_f1,
                "class_names": list(dataset.class_names),
            },
            fold_dir / "best_model.pt",
        )

    row = {
        "subject_id": dataset.subject_id,
        "fold": fold_index,
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "best_epoch": int(best_epoch),
        "validation_accuracy": float(validation_predictions["accuracy"]),
        "validation_macro_f1": float(validation_predictions["macro_f1"]),
        "test_accuracy": float(test_predictions["accuracy"]),
        "test_macro_f1": float(test_predictions["macro_f1"]),
        "test_cohen_kappa": float(test_predictions["cohen_kappa"]),
    }
    return row, test_predictions


def summarize_subject_rows(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    subject_ids = sorted({str(row["subject_id"]) for row in fold_rows})
    for subject_id in subject_ids:
        rows = [row for row in fold_rows if row["subject_id"] == subject_id]
        summary: dict[str, Any] = {
            "subject_id": subject_id,
            "n_folds": len(rows),
        }
        for metric in ("test_accuracy", "test_macro_f1", "test_cohen_kappa"):
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_std_across_folds"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    for section in ("data", "model", "training", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section {section!r} in {path}.")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_physionet_subject_specific.yaml",
    )
    parser.add_argument("--device", help="Training device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--subjects",
        help="Optional override such as 1-3,5 or S001,S002.",
    )
    parser.add_argument(
        "--outer-splits",
        type=int,
        help="Optional outer-fold override, useful for an end-to-end smoke test.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Optional epoch override, useful for an end-to-end smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the planned workload without loading data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    data_cfg = copy.deepcopy(config["data"])
    if str(data_cfg.get("dataset", "physionet_mi")).lower() != "physionet_mi":
        raise ValueError("The subject-specific runner currently supports physionet_mi only.")
    override_subjects = parse_subject_spec(args.subjects)
    if override_subjects is not None:
        data_cfg["subjects"] = override_subjects
    else:
        data_cfg["subjects"] = parse_subject_spec(data_cfg.get("subjects"))
    normalize_data_preprocessing_config(data_cfg)
    training_cfg = copy.deepcopy(config["training"])
    if args.outer_splits is not None:
        if args.outer_splits < 2:
            raise ValueError("--outer-splits must be at least 2.")
        training_cfg["outer_splits"] = args.outer_splits
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        training_cfg["epochs"] = args.epochs
    model_cfg = copy.deepcopy(config["model"])
    outer_splits = int(training_cfg.get("outer_splits", 5))
    validation_size = float(training_cfg.get("validation_size", 0.2))
    planned_subjects = data_cfg.get("subjects")
    planned_count = len(planned_subjects) if planned_subjects is not None else "all"
    print(
        "Subject-specific protocol: one independent model per subject and fold; "
        "early stopping uses validation only."
    )
    print(
        f"Planned subjects={planned_count}, outer_splits={outer_splits}, "
        f"fits={planned_count if isinstance(planned_count, str) else planned_count * outer_splits}."
    )
    if args.dry_run:
        return 0

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_cfg = config["output"]
    output_root = Path(str(output_cfg.get("dir", "experiments/results/subject_specific")))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(
        str(output_cfg.get("dataset_cache_dir", "experiments/cache/preprocessed_datasets"))
    )
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    datasets, skipped = load_subject_specific_datasets(data_cfg, cache_dir)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {**config, "data": data_cfg, "training": training_cfg},
            handle,
            sort_keys=False,
        )
    with (run_dir / "skipped_subjects.json").open("w", encoding="utf-8") as handle:
        json.dump(skipped, handle, indent=2)

    fold_rows: list[dict[str, Any]] = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    for subject_position, dataset in enumerate(datasets, start=1):
        subject_dir = run_dir / dataset.subject_id
        subject_dir.mkdir()
        print(
            f"\n[{subject_position}/{len(datasets)}] {dataset.subject_id}: "
            f"trials={dataset.n_trials}, class_counts={dataset.class_counts}"
        )
        folds = make_subject_folds(
            dataset.y,
            outer_splits=outer_splits,
            validation_size=validation_size,
            seed=int(training_cfg.get("seed", 42)),
        )
        for fold_index, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            fold_dir = subject_dir / f"fold_{fold_index:02d}"
            fold_dir.mkdir()
            row, predictions = train_subject_fold(
                dataset,
                fold_index,
                train_idx,
                val_idx,
                test_idx,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                fold_dir=fold_dir,
                device=device,
            )
            fold_rows.append(row)
            pooled_true.extend(predictions["y_true"].tolist())
            pooled_pred.extend(predictions["y_pred"].tolist())
            print(
                f"  fold {fold_index}/{outer_splits}: "
                f"accuracy={row['test_accuracy']:.4f}, "
                f"macro_f1={row['test_macro_f1']:.4f}, "
                f"kappa={row['test_cohen_kappa']:.4f}, "
                f"best_epoch={row['best_epoch']}"
            )

    _write_csv(run_dir / "per_subject_fold_results.csv", fold_rows)
    subject_rows = summarize_subject_rows(fold_rows)
    _write_csv(run_dir / "per_subject_summary.csv", subject_rows)
    overall = {
        "protocol": "subject-specific nested holdout within outer stratified CV",
        "n_subjects": len(subject_rows),
        "n_skipped_subjects": len(skipped),
        "outer_splits": outer_splits,
        "validation_fraction_of_outer_train": validation_size,
        "chance_accuracy": 1.0 / len(datasets[0].class_names),
        "class_names": list(datasets[0].class_names),
        "accuracy": _metric_summary(
            [float(row["test_accuracy_mean"]) for row in subject_rows]
        ),
        "macro_f1": _metric_summary(
            [float(row["test_macro_f1_mean"]) for row in subject_rows]
        ),
        "cohen_kappa": _metric_summary(
            [float(row["test_cohen_kappa_mean"]) for row in subject_rows]
        ),
        "subjects_above_chance_by_mean_accuracy": sum(
            float(row["test_accuracy_mean"]) > 1.0 / len(datasets[0].class_names)
            for row in subject_rows
        ),
    }
    with (run_dir / "overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    matrix = confusion_matrix(
        pooled_true,
        pooled_pred,
        labels=np.arange(len(datasets[0].class_names)),
    )
    confusion_rows = []
    for true_index, class_name in enumerate(datasets[0].class_names):
        row: dict[str, Any] = {"true_class": class_name}
        row.update(
            {
                f"pred_{predicted_name}": int(matrix[true_index, pred_index])
                for pred_index, predicted_name in enumerate(datasets[0].class_names)
            }
        )
        confusion_rows.append(row)
    _write_csv(run_dir / "pooled_test_confusion_matrix.csv", confusion_rows)
    print(
        "\nSubject-specific accuracy: "
        f"{overall['accuracy']['mean'] * 100:.2f} ± "
        f"{overall['accuracy']['std_between_subjects'] * 100:.2f}% "
        "(mean ± between-subject SD)"
    )
    print(f"Saved results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
