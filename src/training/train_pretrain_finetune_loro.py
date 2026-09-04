"""One global cross-subject model followed by nested subject adaptation.

The subject cohort is split once into configurable global-train and unseen
global-test subjects. One checkpoint is selected using only trials belonging to
the global-train subjects, evaluated on every global-test subject, and then
reused as the starting point for one trial-level fine-tune/test split per unseen
subject. Every subject adaptation restarts from the unchanged global checkpoint.
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

import geoopt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.PhysioNetMI_pretrain_finetune_preprocess import (  # noqa: E402
    load_or_preprocess_spd_with_runs,
    normalize_data_config,
    normalize_dataset_name,
    parse_subjects,
    selected_run_ids,
)
from src.models.MotorImageryDataset import MotorImageryDataset  # noqa: E402
from src.training.config_grid import expand_data_grid, expand_grid  # noqa: E402
from src.training.losses import prototype_loss_settings  # noqa: E402
from src.training.train import (  # noqa: E402
    build_lr_schedulers,
    build_model,
    evaluate,
    normalize_precision_name,
    optimizer_lr_values,
    parse_bool,
    predict_loader,
    resolve_precision,
    save_confusion_matrices,
    save_per_class_metrics,
    set_seed,
    split_params,
    train_one_epoch,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_physionet_pretrain_finetune_loro.yaml"


def format_subject_id(subject_number: int, dataset_name: str) -> str:
    if normalize_dataset_name(dataset_name) == "bnci2014_001":
        return f"A{int(subject_number):02d}"
    return f"S{int(subject_number):03d}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    for section in ("data", "model", "pretrain", "fine_tune", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section {section!r} in {path}.")
    return config


def resolve_single_grid_config(config: dict[str, Any]) -> dict[str, Any]:
    """Accept the same singleton-list syntax used by ``train_grid.yaml``."""

    expanded_sections = {
        "data": expand_data_grid(config["data"]),
        "model": expand_grid(config["model"]),
        "pretrain": expand_grid(config["pretrain"]),
        "fine_tune": expand_grid(config["fine_tune"]),
    }
    multiple = {
        name: len(values)
        for name, values in expanded_sections.items()
        if len(values) != 1
    }
    if multiple:
        raise ValueError(
            "This target-subject runner accepts train_grid-style singleton "
            f"values but not hyperparameter grids; expanded counts: {multiple}."
        )
    return {
        "data": expanded_sections["data"][0],
        "model": expanded_sections["model"][0],
        "pretrain": expanded_sections["pretrain"][0],
        "fine_tune": expanded_sections["fine_tune"][0],
        "output": copy.deepcopy(config["output"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def make_loader(
    full_dataset: MotorImageryDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if len(indices) == 0:
        raise ValueError("Cannot create a loader with zero trials.")
    return DataLoader(
        Subset(full_dataset, indices.astype(int).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def make_model(
    model_cfg: dict[str, Any],
    x: np.ndarray,
    num_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model = build_model(
        model_cfg=copy.deepcopy(model_cfg),
        spd_in_dim=int(x.shape[-1]),
        num_classes=num_classes,
        time_sequence_length=int(x.shape[1]),
        frequency_sequence_length=int(x.shape[2]) if x.ndim >= 5 else 1,
        brain_region_sequence_length=int(x.shape[3]) if x.ndim >= 6 else 1,
    )
    return model.to(device=device, dtype=dtype)


def make_optimizers(
    model: nn.Module,
    cfg: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    stiefel_params, decay_params, no_decay_params = split_params(model)
    euclidean_groups = []
    if decay_params:
        euclidean_groups.append(
            {
                "params": decay_params,
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
            }
        )
    if no_decay_params:
        euclidean_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not euclidean_groups:
        raise RuntimeError("Model has no Euclidean trainable parameters.")
    optimizer = torch.optim.AdamW(
        euclidean_groups,
        lr=float(cfg.get("learning_rate", 1e-3)),
    )
    optimizer_stiefel = None
    if stiefel_params:
        optimizer_stiefel = geoopt.optim.RiemannianAdam(
            stiefel_params,
            lr=float(cfg.get("stiefel_learning_rate", 3e-4)),
            weight_decay=0.0,
            stabilize=10,
        )
    return optimizer, optimizer_stiefel


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    history_path: Path,
    stage_name: str,
) -> tuple[dict[str, torch.Tensor], int, float]:
    """Match ``train.py`` optimization while selecting on validation only."""

    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError(f"{stage_name}.epochs must be at least 1.")
    optimizer, optimizer_stiefel = make_optimizers(model, cfg)
    (
        scheduler_name,
        scheduler_metric,
        scheduler,
        stiefel_scheduler,
    ) = build_lr_schedulers(cfg, optimizer, optimizer_stiefel)
    criterion = nn.CrossEntropyLoss()
    gradient_clip_norm = cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    condition_weight = float(cfg.get("condition_regularization_weight", 0.0))
    prototype_intra_weight, prototype_inter_weight, prototype_margin = (
        prototype_loss_settings(cfg)
    )
    patience = int(cfg.get("early_stopping_patience", 10))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    if patience < 1:
        raise ValueError(f"{stage_name}.early_stopping_patience must be positive.")
    best_validation_macro_f1 = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    print(
        f"    {stage_name} objective=cross_entropy "
        f"+ {prototype_intra_weight:g}*prototype_intra "
        f"+ {prototype_inter_weight:g}*prototype_inter_margin "
        f"(margin={prototype_margin:g})."
    )
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            optimizer_stiefel,
            device,
            gradient_clip_norm=gradient_clip_norm,
            condition_regularization_weight=condition_weight,
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            condition_regularization_weight=condition_weight,
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        )
        if scheduler_name is not None:
            metric_name = str(scheduler_metric or "validation_macro_f1")
            scheduler_values = {
                "validation_loss": float(validation_metrics["loss"]),
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
            }
            if metric_name not in scheduler_values:
                raise ValueError(
                    f"{stage_name}.lr_scheduler_metric must be one of "
                    f"{sorted(scheduler_values)}, got {metric_name!r}."
                )
            scheduler_value = scheduler_values[metric_name]
            scheduler.step(scheduler_value)
            if stiefel_scheduler is not None:
                stiefel_scheduler.step(scheduler_value)
        row = {
            "epoch": epoch,
            "train_loss": float(metrics["loss"]),
            "train_cross_entropy": float(metrics["cross_entropy"]),
            "train_prototype_intra_loss": float(
                metrics["prototype_intra_loss"]
            ),
            "train_prototype_inter_loss": float(
                metrics["prototype_inter_loss"]
            ),
            "train_accuracy": float(metrics["accuracy"]),
            "train_macro_f1": float(metrics["macro_f1"]),
            "validation_loss": float(validation_metrics["loss"]),
            "validation_cross_entropy": float(
                validation_metrics["cross_entropy"]
            ),
            "validation_prototype_intra_loss": float(
                validation_metrics["prototype_intra_loss"]
            ),
            "validation_prototype_inter_loss": float(
                validation_metrics["prototype_inter_loss"]
            ),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "euclid_lr": optimizer_lr_values(optimizer)[0],
            "stiefel_lr": (
                optimizer_lr_values(optimizer_stiefel)[0]
                if optimizer_stiefel is not None
                else None
            ),
        }
        history.append(row)
        improved = (
            validation_metrics["macro_f1"]
            > best_validation_macro_f1 + min_delta
        )
        if improved:
            best_validation_macro_f1 = float(validation_metrics["macro_f1"])
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
        lr_text = f"lr={row['euclid_lr']:.3e}"
        if row["stiefel_lr"] is not None:
            lr_text += f", stiefel_lr={row['stiefel_lr']:.3e}"
        print(
            f"    {stage_name} epoch {epoch:03d}/{epochs}: "
            f"train loss={row['train_loss']:.4f}, "
            f"ce={row['train_cross_entropy']:.4f}, "
            f"intra={row['train_prototype_intra_loss']:.4f}, "
            f"inter={row['train_prototype_inter_loss']:.4f}, "
            f"accuracy={row['train_accuracy']:.4f}, "
            f"mf1={row['train_macro_f1']:.4f} | "
            f"validation loss={row['validation_loss']:.4f}, "
            f"accuracy={row['validation_accuracy']:.4f}, "
            f"mf1={row['validation_macro_f1']:.4f} | "
            f"{lr_text}"
            f"{' [best]' if improved else ''}"
        )
        if epoch - best_epoch >= patience:
            print(
                f"    {stage_name} early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"validation_mf1={best_validation_macro_f1:.4f}."
            )
            break
    _write_csv(history_path, history)
    if best_state is None:
        raise RuntimeError(f"{stage_name} did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    return best_state, best_epoch, best_validation_macro_f1


def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    history_path: Path,
    stage_name: str,
) -> dict[str, torch.Tensor]:
    """Train without validation or test-time model selection."""

    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError(f"{stage_name}.epochs must be at least 1.")
    scheduler_name = str(cfg.get("lr_scheduler", "none")).strip().lower()
    if scheduler_name not in {"", "none", "null", "off", "false"}:
        raise ValueError(
            f"{stage_name}.lr_scheduler must be 'none' without validation; "
            f"got {scheduler_name!r}."
        )
    optimizer, optimizer_stiefel = make_optimizers(model, cfg)
    criterion = nn.CrossEntropyLoss()
    gradient_clip_norm = cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    condition_weight = float(cfg.get("condition_regularization_weight", 0.0))
    prototype_intra_weight, prototype_inter_weight, prototype_margin = (
        prototype_loss_settings(cfg)
    )
    history: list[dict[str, Any]] = []
    print(
        f"    {stage_name} objective=cross_entropy "
        f"+ {prototype_intra_weight:g}*prototype_intra "
        f"+ {prototype_inter_weight:g}*prototype_inter_margin "
        f"(margin={prototype_margin:g})."
    )
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            optimizer_stiefel,
            device,
            gradient_clip_norm=gradient_clip_norm,
            condition_regularization_weight=condition_weight,
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(metrics["loss"]),
            "train_cross_entropy": float(metrics["cross_entropy"]),
            "train_prototype_intra_loss": float(
                metrics["prototype_intra_loss"]
            ),
            "train_prototype_inter_loss": float(
                metrics["prototype_inter_loss"]
            ),
            "train_accuracy": float(metrics["accuracy"]),
            "train_macro_f1": float(metrics["macro_f1"]),
            "euclid_lr": optimizer_lr_values(optimizer)[0],
            "stiefel_lr": (
                optimizer_lr_values(optimizer_stiefel)[0]
                if optimizer_stiefel is not None
                else None
            ),
        }
        history.append(row)
        lr_text = f"lr={row['euclid_lr']:.3e}"
        if row["stiefel_lr"] is not None:
            lr_text += f", stiefel_lr={row['stiefel_lr']:.3e}"
        print(
            f"    {stage_name} epoch {epoch:03d}/{epochs}: "
            f"train loss={row['train_loss']:.4f}, "
            f"ce={row['train_cross_entropy']:.4f}, "
            f"intra={row['train_prototype_intra_loss']:.4f}, "
            f"inter={row['train_prototype_inter_loss']:.4f}, "
            f"accuracy={row['train_accuracy']:.4f}, "
            f"mf1={row['train_macro_f1']:.4f} | {lr_text}"
        )
    _write_csv(history_path, history)
    return _cpu_state_dict(model)


def make_pretrain_split(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    validation_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy trial split retained for reproducibility helpers and old tests."""

    if not 0.0 < validation_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("Pretrain validation_size and test_size must be in (0, 1).")
    if validation_size + test_size >= 1.0:
        raise ValueError("Pretrain validation_size + test_size must be below 1.")
    train_val_idx, test_idx = train_test_split(
        np.asarray(indices, dtype=np.int64),
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y[indices],
    )
    validation_fraction_of_remainder = validation_size / (1.0 - test_size)
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=validation_fraction_of_remainder,
        random_state=seed + 1,
        shuffle=True,
        stratify=y[train_val_idx],
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(validation_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def split_target_train_test(
    target_indices: np.ndarray,
    y: np.ndarray,
    *,
    train_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool all target runs and make one stratified trial-level split."""

    if not 0.0 < train_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("fine_tune train_size and test_size must be in (0, 1).")
    if not np.isclose(train_size + test_size, 1.0):
        raise ValueError(
            "fine_tune train_size and test_size must sum to 1.0; "
            f"got {train_size} + {test_size}."
        )
    target_indices = np.asarray(target_indices, dtype=np.int64)
    train_idx, test_idx = train_test_split(
        target_indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y[target_indices],
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def split_global_subjects(
    subjects: list[str] | np.ndarray,
    *,
    train_size: float,
    test_size: float,
    seed: int,
    shuffle: bool = True,
) -> tuple[list[str], list[str]]:
    """Split subject IDs once; no subject can appear on both sides."""

    subjects_array = np.asarray(sorted(set(np.asarray(subjects).astype(str).tolist())))
    if len(subjects_array) < 2:
        raise ValueError("Global cross-subject training requires at least two subjects.")
    if not 0.0 < train_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("Subject train_size and test_size must be in (0, 1).")
    if not np.isclose(train_size + test_size, 1.0):
        raise ValueError("Subject train_size and test_size must sum to 1.0.")
    train_subjects, test_subjects = train_test_split(
        subjects_array,
        train_size=train_size,
        test_size=test_size,
        random_state=seed if shuffle else None,
        shuffle=shuffle,
    )
    train_result = sorted(np.asarray(train_subjects).astype(str).tolist())
    test_result = sorted(np.asarray(test_subjects).astype(str).tolist())
    if set(train_result) & set(test_result):
        raise RuntimeError("Global subject train/test split overlaps.")
    if set(train_result) | set(test_result) != set(subjects_array.tolist()):
        raise RuntimeError("Global subject train/test split omitted subjects.")
    return train_result, test_result


def make_global_train_validation_split(
    indices: np.ndarray,
    y: np.ndarray,
    *,
    validation_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified validation split only inside global-train subjects."""

    if not 0.0 < validation_size < 1.0:
        raise ValueError("pretrain.validation_size must be in (0, 1).")
    indices = np.asarray(indices, dtype=np.int64)
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=y[indices],
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(validation_idx, dtype=np.int64),
    )


def validate_global_protocol(
    y: np.ndarray,
    subject_labels: np.ndarray,
    global_train_subjects: list[str],
    global_test_subjects: list[str],
    *,
    num_classes: int,
) -> None:
    """Audit the nested subject/trial protocol before model training."""

    train_subject_set = set(global_train_subjects)
    test_subject_set = set(global_test_subjects)
    available_subjects = set(subject_labels.astype(str).tolist())
    if train_subject_set & test_subject_set:
        raise ValueError("Global train and test subject sets overlap.")
    if train_subject_set | test_subject_set != available_subjects:
        missing = sorted(available_subjects - train_subject_set - test_subject_set)
        extra = sorted((train_subject_set | test_subject_set) - available_subjects)
        raise ValueError(
            f"Global subject split does not cover the loaded cohort; missing={missing}, "
            f"extra={extra}."
        )
    expected_classes = set(range(num_classes))
    global_train_mask = np.isin(subject_labels, global_train_subjects)
    global_test_mask = np.isin(subject_labels, global_test_subjects)
    if set(np.unique(y[global_train_mask]).astype(int).tolist()) != expected_classes:
        raise ValueError("Global training subjects do not contain every class.")
    if set(np.unique(y[global_test_mask]).astype(int).tolist()) != expected_classes:
        raise ValueError("Global test subjects do not contain every class.")
    for subject in global_test_subjects:
        target_mask = subject_labels == subject
        counts = np.bincount(y[target_mask], minlength=num_classes)
        if set(np.unique(y[target_mask]).astype(int).tolist()) != expected_classes:
            raise ValueError(f"Global test subject {subject} lacks classes.")
        if int(counts[counts > 0].min()) < 2:
            raise ValueError(
                f"Subject {subject} cannot be split into fine-tune/test; "
                f"class counts are {counts.tolist()}."
            )


def metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--target-subjects",
        help=(
            "Optional explicit global-test subjects, for example 1-10. "
            "When omitted, subjects are split by the configured ratio."
        ),
    )
    parser.add_argument(
        "--data-subjects",
        help="Override the complete data.subjects cohort, mainly for a smoke test.",
    )
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu.")
    parser.add_argument("--pretrain-epochs", type=int, help="Smoke-test override.")
    parser.add_argument("--fine-tune-epochs", type=int, help="Smoke-test override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the protocol without loading EEG data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = resolve_single_grid_config(load_config(config_path))
    pretrain_cfg = copy.deepcopy(config["pretrain"])
    fine_tune_cfg = copy.deepcopy(config["fine_tune"])
    if args.pretrain_epochs is not None:
        pretrain_cfg["epochs"] = args.pretrain_epochs
    if args.fine_tune_epochs is not None:
        fine_tune_cfg["epochs"] = args.fine_tune_epochs
    for name, cfg in (("pretrain", pretrain_cfg), ("fine_tune", fine_tune_cfg)):
        if int(cfg.get("epochs", 0)) < 1:
            raise ValueError(f"{name}.epochs must be at least 1.")

    preprocessing_data_cfg = copy.deepcopy(config["data"])
    legacy_pretrain_subjects = preprocessing_data_cfg.pop("pretrain_subjects", None)
    if args.data_subjects is not None:
        preprocessing_data_cfg["subjects"] = args.data_subjects
    elif preprocessing_data_cfg.get("subjects") is None:
        preprocessing_data_cfg["subjects"] = legacy_pretrain_subjects
    data_cfg = normalize_data_config(preprocessing_data_cfg)
    configured_subject_numbers = data_cfg.get("subjects")
    if not configured_subject_numbers:
        raise ValueError("data.subjects must define the complete subject cohort.")

    subject_train_size = float(pretrain_cfg.get("subject_train_size", 0.8))
    subject_test_size = float(pretrain_cfg.get("subject_test_size", 0.2))
    pretrain_seed = int(pretrain_cfg.get("seed", 42))
    subject_split_seed = int(
        pretrain_cfg.get("subject_split_seed", pretrain_seed)
    )
    subject_split_shuffle = parse_bool(
        pretrain_cfg.get("subject_split_shuffle", True),
        default=True,
    )
    fine_tune_train_size = float(fine_tune_cfg.get("train_size", 0.7))
    fine_tune_test_size = float(fine_tune_cfg.get("test_size", 0.3))
    fine_tune_seed = int(fine_tune_cfg.get("seed", pretrain_seed))
    if not np.isclose(subject_train_size + subject_test_size, 1.0):
        raise ValueError("pretrain subject_train_size and subject_test_size must sum to 1.")
    if not np.isclose(fine_tune_train_size + fine_tune_test_size, 1.0):
        raise ValueError("fine_tune train_size and test_size must sum to 1.")

    dataset_name = normalize_dataset_name(data_cfg.get("dataset"))
    explicit_test_numbers = None
    if args.target_subjects is not None:
        if args.target_subjects.strip().lower() == "all":
            raise ValueError("--target-subjects all would leave no Global-train subjects.")
        explicit_test_numbers = parse_subjects(args.target_subjects)
        if not explicit_test_numbers:
            raise ValueError("--target-subjects did not contain any subject IDs.")

    configured_labels = [
        format_subject_id(number, dataset_name)
        for number in configured_subject_numbers
    ]
    if explicit_test_numbers is None:
        dry_train_subjects, dry_test_subjects = split_global_subjects(
            configured_labels,
            train_size=subject_train_size,
            test_size=subject_test_size,
            seed=subject_split_seed,
            shuffle=subject_split_shuffle,
        )
    else:
        dry_test_subjects = sorted(
            format_subject_id(number, dataset_name)
            for number in explicit_test_numbers
        )
        missing = sorted(set(dry_test_subjects) - set(configured_labels))
        if missing:
            raise ValueError(f"Explicit Global-test subjects are outside data.subjects: {missing}.")
        dry_train_subjects = sorted(set(configured_labels) - set(dry_test_subjects))
        if not dry_train_subjects:
            raise ValueError("Explicit Global-test subjects leave no training subjects.")

    expected_runs: list[int] | str
    if dataset_name == "physionet_mi":
        expected_runs = selected_run_ids(
            imaged=bool(data_cfg["imaged"]),
            executed=bool(data_cfg["executed"]),
            task_types=tuple(data_cfg["task_types"]),
        )
    else:
        expected_runs = "MOABB session/run pairs (normally 12 per subject)"
    print("Protocol: one Global checkpoint -> cross-subject test -> subject adaptation.")
    print(
        f"Subjects: global train={len(dry_train_subjects)}, "
        f"global test={len(dry_test_subjects)} "
        f"({subject_train_size:.2f}/{subject_test_size:.2f}, "
        f"shuffle={subject_split_shuffle}, seed={subject_split_seed})."
    )
    print(
        "Global-train trials use a stratified validation split of "
        f"{float(pretrain_cfg.get('validation_size', 0.15)):.2f}; "
        "all Global-test trials are held out until cross-subject evaluation."
    )
    print(
        "Each Global-test subject then pools all runs and uses one stratified "
        f"trial split ({fine_tune_train_size:.2f}/{fine_tune_test_size:.2f}); "
        f"expected runs={expected_runs}."
    )
    if args.dry_run:
        print(f"Global-train subjects={','.join(dry_train_subjects)}")
        print(f"Global-test subjects={','.join(dry_test_subjects)}")
        return 0

    output_cfg = config["output"]
    output_root = Path(
        str(output_cfg.get("dir", "experiments/results/global_subject_finetune"))
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    global_dir = run_dir / "global"
    global_dir.mkdir()
    ss_tl_dir = run_dir / "ss_tl"
    ss_tl_dir.mkdir()
    cache_dir = Path(
        str(output_cfg.get("dataset_cache_dir", "experiments/cache/preprocessed_datasets"))
    )
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    x, y, subject_labels, run_labels, class_names = (
        load_or_preprocess_spd_with_runs(data_cfg, cache_dir)
    )
    subject_labels = np.asarray(subject_labels, dtype=np.str_)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"This experiment needs at least two classes: {class_names}.")
    available_subjects = sorted(np.unique(subject_labels).astype(str).tolist())
    if explicit_test_numbers is None:
        global_train_subjects, global_test_subjects = split_global_subjects(
            available_subjects,
            train_size=subject_train_size,
            test_size=subject_test_size,
            seed=subject_split_seed,
            shuffle=subject_split_shuffle,
        )
    else:
        global_test_subjects = sorted(
            format_subject_id(number, dataset_name)
            for number in explicit_test_numbers
        )
        missing = sorted(set(global_test_subjects) - set(available_subjects))
        if missing:
            raise ValueError(f"Explicit Global-test subjects were not loaded: {missing}.")
        global_train_subjects = sorted(
            set(available_subjects) - set(global_test_subjects)
        )
    validate_global_protocol(
        y,
        subject_labels,
        global_train_subjects,
        global_test_subjects,
        num_classes=num_classes,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = normalize_precision_name(pretrain_cfg.get("precision", "float32"))
    fine_tune_precision = normalize_precision_name(
        fine_tune_cfg.get("precision", precision)
    )
    if fine_tune_precision != precision:
        raise ValueError("pretrain and fine_tune precision must match.")
    dtype = resolve_precision(precision)
    pretrain_allow_tf32 = parse_bool(
        pretrain_cfg.get("allow_tf32", False), default=False
    )
    fine_tune_allow_tf32 = parse_bool(
        fine_tune_cfg.get("allow_tf32", pretrain_allow_tf32),
        default=pretrain_allow_tf32,
    )
    if fine_tune_allow_tf32 != pretrain_allow_tf32:
        raise ValueError("pretrain and fine_tune allow_tf32 must match.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = pretrain_allow_tf32
        torch.backends.cudnn.allow_tf32 = pretrain_allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(
                "high" if pretrain_allow_tf32 else "highest"
            )

    full_dataset = MotorImageryDataset(x, y, dtype=dtype)
    num_workers = int(pretrain_cfg.get("num_workers", 0))
    pin_memory = parse_bool(
        pretrain_cfg.get("pin_memory", device.type == "cuda"),
        default=device.type == "cuda",
    )
    global_train_all_idx = np.flatnonzero(
        np.isin(subject_labels, global_train_subjects)
    ).astype(np.int64)
    global_test_idx = np.flatnonzero(
        np.isin(subject_labels, global_test_subjects)
    ).astype(np.int64)
    global_train_idx, global_validation_idx = make_global_train_validation_split(
        global_train_all_idx,
        y,
        validation_size=float(pretrain_cfg.get("validation_size", 0.15)),
        seed=pretrain_seed,
    )

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                **config,
                "data": {**data_cfg, "subjects": configured_subject_numbers},
                "pretrain": pretrain_cfg,
                "fine_tune": fine_tune_cfg,
                "resolved_protocol": {
                    "global_train_subjects": global_train_subjects,
                    "global_test_subjects": global_test_subjects,
                },
            },
            handle,
            sort_keys=False,
        )

    print(
        f"\nGlobal training: {len(global_train_subjects)} subjects, "
        f"train/validation trials={len(global_train_idx)}/{len(global_validation_idx)}; "
        f"held-out cross-subject trials={len(global_test_idx)}."
    )
    set_seed(pretrain_seed)
    global_model = make_model(
        config["model"],
        x,
        num_classes,
        device=device,
        dtype=dtype,
    )
    global_train_loader = make_loader(
        full_dataset,
        global_train_idx,
        batch_size=int(pretrain_cfg.get("batch_size", 128)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    global_validation_loader = make_loader(
        full_dataset,
        global_validation_idx,
        batch_size=int(pretrain_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    global_state, global_best_epoch, global_best_validation_mf1 = (
        train_with_early_stopping(
            global_model,
            global_train_loader,
            global_validation_loader,
            pretrain_cfg,
            device=device,
            history_path=global_dir / "train_history.csv",
            stage_name="global",
        )
    )
    criterion = nn.CrossEntropyLoss()
    condition_weight = float(
        pretrain_cfg.get("condition_regularization_weight", 0.0)
    )
    pretrain_intra_weight, pretrain_inter_weight, pretrain_margin = (
        prototype_loss_settings(pretrain_cfg)
    )
    validation_predictions = predict_loader(
        global_model,
        global_validation_loader,
        criterion,
        device,
        condition_regularization_weight=condition_weight,
        prototype_intra_weight=pretrain_intra_weight,
        prototype_inter_weight=pretrain_inter_weight,
        prototype_margin=pretrain_margin,
    )
    global_test_loader = make_loader(
        full_dataset,
        global_test_idx,
        batch_size=int(pretrain_cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    global_predictions = predict_loader(
        global_model,
        global_test_loader,
        criterion,
        device,
        condition_regularization_weight=condition_weight,
        prototype_intra_weight=pretrain_intra_weight,
        prototype_inter_weight=pretrain_inter_weight,
        prototype_margin=pretrain_margin,
    )
    global_metrics = metrics_from_arrays(
        global_predictions["y_true"],
        global_predictions["y_pred"],
        num_classes=num_classes,
    )
    global_prediction_by_index = np.full(len(y), -1, dtype=np.int64)
    global_prediction_by_index[global_test_idx] = np.asarray(
        global_predictions["y_pred"], dtype=np.int64
    )
    global_subject_rows = []
    for subject in global_test_subjects:
        subject_idx = np.flatnonzero(subject_labels == subject).astype(np.int64)
        subject_metrics = metrics_from_arrays(
            y[subject_idx],
            global_prediction_by_index[subject_idx],
            num_classes=num_classes,
        )
        global_subject_rows.append(
            {
                "subject": subject,
                "n_trials": int(len(subject_idx)),
                **subject_metrics,
            }
        )
    global_subject_accuracies = [row["accuracy"] for row in global_subject_rows]
    _write_csv(global_dir / "cross_subject_per_subject.csv", global_subject_rows)
    save_per_class_metrics(
        global_dir / "cross_subject_per_class_metrics.csv",
        {"global_cross_subject": global_predictions},
        class_names,
    )
    save_confusion_matrices(
        global_dir / "cross_subject_confusion_matrix.csv",
        {"global_cross_subject": global_predictions},
        class_names,
    )
    save_per_class_metrics(
        global_dir / "validation_per_class_metrics.csv",
        {"validation": validation_predictions},
        class_names,
    )
    with (global_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "strategy": "single_subject_holdout_ratio_then_trial_validation",
                "subject_train_size": subject_train_size,
                "subject_test_size": subject_test_size,
                "subject_split_seed": subject_split_seed,
                "subject_split_shuffle": subject_split_shuffle,
                "global_train_subjects": global_train_subjects,
                "global_test_subjects": global_test_subjects,
                "global_train_indices": global_train_idx.astype(int).tolist(),
                "global_validation_indices": global_validation_idx.astype(int).tolist(),
                "global_test_indices": global_test_idx.astype(int).tolist(),
            },
            handle,
            indent=2,
        )
    global_summary = {
        "best_epoch": global_best_epoch,
        "best_validation_macro_f1": global_best_validation_mf1,
        "validation_accuracy": float(validation_predictions["accuracy"]),
        "validation_macro_f1": float(validation_predictions["macro_f1"]),
        "n_global_train_subjects": len(global_train_subjects),
        "n_global_test_subjects": len(global_test_subjects),
        "n_global_train_trials": int(len(global_train_idx)),
        "n_global_validation_trials": int(len(global_validation_idx)),
        "n_global_cross_subject_test_trials": int(len(global_test_idx)),
        "mean_subject_accuracy": statistics.fmean(global_subject_accuracies),
        "between_subject_accuracy_sd": (
            statistics.stdev(global_subject_accuracies)
            if len(global_subject_accuracies) > 1
            else 0.0
        ),
        **global_metrics,
    }
    with (global_dir / "cross_subject_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(global_summary, handle, indent=2)
    save_global_checkpoint = parse_bool(
        output_cfg.get("save_pretrained_checkpoints", True), default=True
    )
    if save_global_checkpoint:
        torch.save(
            {
                "model_state_dict": global_state,
                "class_names": class_names,
                "best_epoch": global_best_epoch,
                "best_validation_macro_f1": global_best_validation_mf1,
                "global_train_subjects": global_train_subjects,
                "global_test_subjects": global_test_subjects,
            },
            global_dir / "global_model.pt",
        )
    print(
        "Global cross-subject accuracy="
        f"{global_metrics['accuracy']:.4f}, macro-F1={global_metrics['macro_f1']:.4f}."
    )
    del global_model, global_train_loader, global_validation_loader, global_test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    run_rows: list[dict[str, Any]] = []
    fine_tune_num_workers = int(fine_tune_cfg.get("num_workers", num_workers))
    fine_tune_pin_memory = parse_bool(
        fine_tune_cfg.get("pin_memory", pin_memory), default=pin_memory
    )
    for subject_position, subject in enumerate(global_test_subjects, start=1):
        subject_number_text = "".join(character for character in subject if character.isdigit())
        subject_number = int(subject_number_text) if subject_number_text else subject_position
        subject_seed = fine_tune_seed + subject_number * 100
        target_indices = np.flatnonzero(subject_labels == subject).astype(np.int64)
        fine_tune_indices, test_indices = split_target_train_test(
            target_indices,
            y,
            train_size=fine_tune_train_size,
            test_size=fine_tune_test_size,
            seed=subject_seed,
        )
        subject_dir = ss_tl_dir / subject / "trial_random_split"
        subject_dir.mkdir(parents=True)
        before_metrics = metrics_from_arrays(
            y[test_indices],
            global_prediction_by_index[test_indices],
            num_classes=num_classes,
        )
        print(
            f"\n[{subject_position}/{len(global_test_subjects)}] {subject}: "
            f"fine-tune/test trials={len(fine_tune_indices)}/{len(test_indices)}, "
            f"global-before accuracy={before_metrics['accuracy']:.4f}."
        )
        set_seed(subject_seed)
        model = make_model(
            config["model"],
            x,
            num_classes,
            device=device,
            dtype=dtype,
        )
        model.load_state_dict(global_state)
        fine_tune_loader = make_loader(
            full_dataset,
            fine_tune_indices,
            batch_size=int(fine_tune_cfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=fine_tune_num_workers,
            pin_memory=fine_tune_pin_memory,
        )
        final_state = train_fixed_epochs(
            model,
            fine_tune_loader,
            fine_tune_cfg,
            device=device,
            history_path=subject_dir / "fine_tune_history.csv",
            stage_name=f"fine-tune {subject}",
        )
        test_loader = make_loader(
            full_dataset,
            test_indices,
            batch_size=int(fine_tune_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=fine_tune_num_workers,
            pin_memory=fine_tune_pin_memory,
        )
        fine_tune_intra_weight, fine_tune_inter_weight, fine_tune_margin = (
            prototype_loss_settings(fine_tune_cfg)
        )
        predictions = predict_loader(
            model,
            test_loader,
            criterion,
            device,
            condition_regularization_weight=float(
                fine_tune_cfg.get("condition_regularization_weight", 0.0)
            ),
            prototype_intra_weight=fine_tune_intra_weight,
            prototype_inter_weight=fine_tune_inter_weight,
            prototype_margin=fine_tune_margin,
        )
        after_metrics = metrics_from_arrays(
            predictions["y_true"],
            predictions["y_pred"],
            num_classes=num_classes,
        )
        row = {
            "target_subject": subject,
            "fine_tune_runs": ",".join(
                str(value) for value in sorted(np.unique(run_labels[fine_tune_indices]))
            ),
            "test_runs": ",".join(
                str(value) for value in sorted(np.unique(run_labels[test_indices]))
            ),
            "n_global_train_subjects": len(global_train_subjects),
            "n_global_train_trials": int(len(global_train_idx)),
            "n_fine_tune_trials": int(len(fine_tune_indices)),
            "n_test_trials": int(len(test_indices)),
            "global_before_accuracy": before_metrics["accuracy"],
            "global_before_balanced_accuracy": before_metrics["balanced_accuracy"],
            "global_before_macro_f1": before_metrics["macro_f1"],
            "global_before_cohen_kappa": before_metrics["cohen_kappa"],
            "fine_tuned_accuracy": after_metrics["accuracy"],
            "fine_tuned_balanced_accuracy": after_metrics["balanced_accuracy"],
            "fine_tuned_macro_f1": after_metrics["macro_f1"],
            "fine_tuned_cohen_kappa": after_metrics["cohen_kappa"],
            "accuracy_gain": after_metrics["accuracy"] - before_metrics["accuracy"],
            "epochs_completed": int(fine_tune_cfg["epochs"]),
            "_y_true": np.asarray(predictions["y_true"], dtype=np.int64),
            "_y_pred": np.asarray(predictions["y_pred"], dtype=np.int64),
        }
        run_rows.append(row)
        save_per_class_metrics(
            subject_dir / "per_class_metrics.csv", {"test": predictions}, class_names
        )
        save_confusion_matrices(
            subject_dir / "confusion_matrix.csv", {"test": predictions}, class_names
        )
        with (subject_dir / "split.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_subject": subject,
                    "strategy": "pooled_runs_stratified_trial_train_test",
                    "train_size": fine_tune_train_size,
                    "test_size": fine_tune_test_size,
                    "seed": subject_seed,
                    "source_global_checkpoint": (
                        "global/global_model.pt"
                        if save_global_checkpoint
                        else "in_memory_global_best_state"
                    ),
                    "fine_tune_indices": fine_tune_indices.astype(int).tolist(),
                    "test_indices": test_indices.astype(int).tolist(),
                },
                handle,
                indent=2,
            )
        if parse_bool(
            output_cfg.get("save_fine_tuned_checkpoints", False), default=False
        ):
            torch.save(
                {
                    "model_state_dict": final_state,
                    "target_subject": subject,
                    "class_names": class_names,
                    "source_global_checkpoint": (
                        "global/global_model.pt"
                        if save_global_checkpoint
                        else "in_memory_global_best_state"
                    ),
                },
                subject_dir / "fine_tuned_model.pt",
            )
        print(
            f"    fine-tuned accuracy={after_metrics['accuracy']:.4f}, "
            f"gain={row['accuracy_gain']:+.4f}."
        )
        del model, fine_tune_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in run_rows
    ]
    _write_csv(run_dir / "per_subject_results.csv", public_rows)
    _write_csv(
        run_dir / "per_subject_summary.csv",
        [
            {
                "Subject": row["target_subject"],
                "Trials": row["n_test_trials"],
                "Accuracy (%)": float(row["fine_tuned_accuracy"]) * 100.0,
                "Balanced Accuracy (%)": (
                    float(row["fine_tuned_balanced_accuracy"]) * 100.0
                ),
                "Macro-F1": row["fine_tuned_macro_f1"],
                "Cohen’s κ": row["fine_tuned_cohen_kappa"],
            }
            for row in public_rows
        ],
    )
    pooled_true = np.concatenate([row["_y_true"] for row in run_rows])
    pooled_pred = np.concatenate([row["_y_pred"] for row in run_rows])
    fine_tuned_metrics = metrics_from_arrays(
        pooled_true,
        pooled_pred,
        num_classes=num_classes,
    )
    subject_accuracies = [float(row["fine_tuned_accuracy"]) for row in run_rows]
    overall = {
        "protocol": "single_global_subject_split_then_nested_subject_trial_adaptation",
        "dataset": dataset_name,
        "class_names": class_names,
        "chance_accuracy": 1.0 / num_classes,
        "subject_train_fraction": subject_train_size,
        "subject_test_fraction": subject_test_size,
        "global_train_subjects": global_train_subjects,
        "global_test_subjects": global_test_subjects,
        "global_cross_subject": global_summary,
        "fine_tune_train_fraction": fine_tune_train_size,
        "fine_tune_test_fraction": fine_tune_test_size,
        "fine_tune_seed": fine_tune_seed,
        "fine_tune_uses_validation": False,
        "fine_tuned_mean_subject_accuracy": statistics.fmean(subject_accuracies),
        "fine_tuned_between_subject_accuracy_sd": (
            statistics.stdev(subject_accuracies)
            if len(subject_accuracies) > 1
            else 0.0
        ),
        "fine_tuned_pooled": fine_tuned_metrics,
    }
    with (run_dir / "overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    matrix = confusion_matrix(
        pooled_true,
        pooled_pred,
        labels=np.arange(num_classes),
    )
    confusion_rows = []
    for true_index, class_name in enumerate(class_names):
        confusion_rows.append(
            {
                "true_class": class_name,
                **{
                    f"pred_{predicted_name}": int(matrix[true_index, pred_index])
                    for pred_index, predicted_name in enumerate(class_names)
                },
            }
        )
    _write_csv(run_dir / "pooled_fine_tuned_confusion_matrix.csv", confusion_rows)
    print(
        "\nFine-tuned mean subject accuracy: "
        f"{overall['fine_tuned_mean_subject_accuracy'] * 100:.2f} +/- "
        f"{overall['fine_tuned_between_subject_accuracy_sd'] * 100:.2f}%"
    )
    print(f"Saved results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
