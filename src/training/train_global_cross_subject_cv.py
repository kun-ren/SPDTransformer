"""Select global-model hyperparameters with subject-wise cross-validation.

Each configuration is trained independently in every fold. Fold train and test
subjects are disjoint, and each subject appears in test exactly once. There is
no validation split, early stopping, final holdout evaluation, or fine-tuning.
The arithmetic mean of fold test accuracies is the selection score.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import random
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
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.MotorImageryDataset import MotorImageryDataset  # noqa: E402
from src.training.config_grid import expand_data_grid, expand_grid  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_physionet_global_cv_hparam.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def normalize_precision_name(value: Any) -> str:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"float32", "float", "single", "fp32"}:
        return "float32"
    if normalized in {"float64", "double", "fp64"}:
        return "float64"
    raise ValueError(f"Unsupported training precision: {value!r}")


def resolve_precision(value: Any) -> torch.dtype:
    return (
        torch.float64
        if normalize_precision_name(value) == "float64"
        else torch.float32
    )


def load_dataset(data_cfg: dict[str, Any], cache_dir: Path):
    """Delay MNE imports so grid inspection does not initialize EEG tooling."""

    from src.datasets.PhysioNetMI_pretrain_finetune_preprocess import (
        load_or_preprocess_spd_with_runs,
        normalize_data_config,
    )

    return load_or_preprocess_spd_with_runs(normalize_data_config(data_cfg), cache_dir)


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
        Subset(full_dataset, np.asarray(indices).astype(int).tolist()),
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
    from src.training.train import build_model

    model = build_model(
        model_cfg=copy.deepcopy(model_cfg),
        spd_in_dim=int(x.shape[-1]),
        num_classes=num_classes,
        time_sequence_length=int(x.shape[1]),
        frequency_sequence_length=int(x.shape[2]) if x.ndim >= 5 else 1,
        brain_region_sequence_length=int(x.shape[3]) if x.ndim >= 6 else 1,
    )
    return model.to(device=device, dtype=dtype)


def split_params(model: nn.Module):
    stiefel_params = []
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if isinstance(parameter, geoopt.ManifoldParameter):
            stiefel_params.append(parameter)
        elif (
            "norm" in name.lower()
            or "metric_low_rank" in name.lower()
            or "metric_matrix" in name.lower()
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return stiefel_params, decay_params, no_decay_params


def optimizer_lr_values(
    optimizer: torch.optim.Optimizer | None,
) -> list[float]:
    if optimizer is None:
        return []
    return [float(group["lr"]) for group in optimizer.param_groups]


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


def make_lr_schedulers(
    optimizer: torch.optim.Optimizer,
    optimizer_stiefel: torch.optim.Optimizer | None,
    cfg: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    """Build epoch-based schedulers that never inspect a test-fold metric."""

    scheduler_name = str(cfg.get("lr_scheduler", "none")).strip().lower()
    if scheduler_name in {"", "none", "null", "off", "false"}:
        return None, None

    epochs = int(cfg["epochs"])

    def build_scheduler(
        target_optimizer: torch.optim.Optimizer,
        *,
        stiefel: bool,
    ) -> Any:
        if scheduler_name in {"cosine", "cosine_annealing"}:
            min_lr_key = (
                "stiefel_lr_scheduler_min_lr" if stiefel else "lr_scheduler_min_lr"
            )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                target_optimizer,
                T_max=int(cfg.get("lr_scheduler_t_max", epochs)),
                eta_min=float(cfg.get(min_lr_key, 0.0)),
            )
        if scheduler_name in {"step", "step_lr"}:
            return torch.optim.lr_scheduler.StepLR(
                target_optimizer,
                step_size=int(cfg.get("lr_scheduler_step_size", 30)),
                gamma=float(cfg.get("lr_scheduler_gamma", 0.1)),
            )
        if scheduler_name in {"multistep", "multi_step", "multistep_lr"}:
            return torch.optim.lr_scheduler.MultiStepLR(
                target_optimizer,
                milestones=[int(value) for value in cfg["lr_scheduler_milestones"]],
                gamma=float(cfg.get("lr_scheduler_gamma", 0.1)),
            )
        if scheduler_name in {"exponential", "exponential_lr"}:
            return torch.optim.lr_scheduler.ExponentialLR(
                target_optimizer,
                gamma=float(cfg.get("lr_scheduler_gamma", 0.99)),
            )
        raise AssertionError(f"Unvalidated scheduler: {scheduler_name}")

    scheduler = build_scheduler(optimizer, stiefel=False)
    stiefel_scheduler = (
        build_scheduler(optimizer_stiefel, stiefel=True)
        if optimizer_stiefel is not None
        else None
    )
    return scheduler, stiefel_scheduler


def condition_regularization(matrix: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    eigenvalues = torch.clamp(torch.linalg.eigvalsh(matrix), min=eps)
    return torch.log(eigenvalues[..., -1] / eigenvalues[..., 0]).mean()


def assert_model_finite(model: nn.Module, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite parameter after {context}: {name}")
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient after {context}: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer_euclid: torch.optim.Optimizer,
    optimizer_stiefel: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    gradient_clip_norm: float | None,
    condition_regularization_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []
    optimizers = [optimizer_euclid]
    if optimizer_stiefel is not None:
        optimizers.append(optimizer_stiefel)
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
        y_batch = y_batch.to(device, non_blocking=device.type == "cuda")
        use_condition_regularization = condition_regularization_weight > 0
        logits, aux = model(
            x_batch,
            return_aux=use_condition_regularization,
        )
        condition_loss = logits.new_tensor(0.0)
        if use_condition_regularization and aux:
            condition_loss = torch.stack(
                [condition_regularization(matrix) for matrix in aux.values()]
            ).mean()
        loss = criterion(logits, y_batch) + (
            condition_regularization_weight * condition_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item():.6e}")
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        assert_model_finite(model, "backward")
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            parameters = [
                parameter
                for optimizer in optimizers
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=gradient_clip_norm)
        for optimizer in optimizers:
            optimizer.step()
        assert_model_finite(model, "optimizer step")
        total_loss += loss.item() * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": classification_metrics(
            np.asarray(y_true),
            np.asarray(y_pred),
            num_classes=int(max(y_true) + 1),
        )["macro_f1"],
    }


def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    history_path: Path,
    stage_name: str,
) -> dict[str, torch.Tensor]:
    optimizer, optimizer_stiefel = make_optimizers(model, cfg)
    scheduler, stiefel_scheduler = make_lr_schedulers(
        optimizer,
        optimizer_stiefel,
        cfg,
    )
    criterion = nn.CrossEntropyLoss()
    gradient_clip_norm = cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    condition_weight = float(cfg.get("condition_regularization_weight", 0.0))
    history = []
    scheduler_name = str(cfg.get("lr_scheduler", "none")).strip().lower()
    print(
        f"    {stage_name} lr_scheduler={scheduler_name}, "
        f"euclid_lr={optimizer_lr_values(optimizer)[0]:.6g}, "
        f"stiefel_lr="
        f"{optimizer_lr_values(optimizer_stiefel)[0] if optimizer_stiefel is not None else None}."
    )
    for epoch in range(1, int(cfg["epochs"]) + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            optimizer_stiefel,
            device,
            gradient_clip_norm=gradient_clip_norm,
            condition_regularization_weight=condition_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(metrics["loss"]),
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
        if scheduler is not None:
            scheduler.step()
        if stiefel_scheduler is not None:
            stiefel_scheduler.step()
        print(
            f"    {stage_name} epoch {epoch:03d}/{int(cfg['epochs'])}: "
            f"loss={row['train_loss']:.4f}, accuracy={row['train_accuracy']:.4f}, "
            f"macro-F1={row['train_macro_f1']:.4f}, "
            f"euclid_lr={row['euclid_lr']:.6g}, "
            f"stiefel_lr={row['stiefel_lr']}."
        )
    write_csv(history_path, history)
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    condition_regularization_weight: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=device.type == "cuda")
            y_batch = y_batch.to(device, non_blocking=device.type == "cuda")
            use_condition_regularization = condition_regularization_weight > 0
            logits, aux = model(
                x_batch,
                return_aux=use_condition_regularization,
            )
            condition_loss = logits.new_tensor(0.0)
            if use_condition_regularization and aux:
                condition_loss = torch.stack(
                    [condition_regularization(matrix) for matrix in aux.values()]
                ).mean()
            loss = criterion(logits, y_batch) + (
                condition_regularization_weight * condition_loss
            )
            total_loss += loss.item() * y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    true_array = np.asarray(y_true, dtype=np.int64)
    pred_array = np.asarray(y_pred, dtype=np.int64)
    return {
        "loss": total_loss / len(true_array),
        "accuracy": float((true_array == pred_array).mean()),
        "macro_f1": classification_metrics(
            true_array,
            pred_array,
            num_classes=int(max(true_array.max(), pred_array.max()) + 1),
        )["macro_f1"],
        "cohen_kappa": classification_metrics(
            true_array,
            pred_array,
            num_classes=int(max(true_array.max(), pred_array.max()) + 1),
        )["cohen_kappa"],
        "y_true": true_array,
        "y_pred": pred_array,
    }


def metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return classification_metrics(y_true, y_pred, num_classes=num_classes)


def confusion_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    matrix = confusion_array(y_true, y_pred, num_classes)
    true_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(float)
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros(num_classes, dtype=float),
        where=true_support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(num_classes, dtype=float),
        where=predicted_support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=float),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total)
    expected_agreement = float(
        np.dot(true_support, predicted_support) / float(total * total)
    )
    kappa = (
        (accuracy - expected_agreement) / (1.0 - expected_agreement)
        if expected_agreement < 1.0
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "cohen_kappa": float(kappa),
    }


def save_per_class_metrics(
    path: Path,
    split_predictions: dict[str, dict[str, Any]],
    class_names: list[str],
) -> None:
    rows = []
    for split_name, prediction in split_predictions.items():
        matrix = confusion_array(
            np.asarray(prediction["y_true"]),
            np.asarray(prediction["y_pred"]),
            len(class_names),
        )
        support = matrix.sum(axis=1)
        predicted_support = matrix.sum(axis=0)
        true_positive = np.diag(matrix).astype(float)
        precision = np.divide(
            true_positive,
            predicted_support,
            out=np.zeros(len(class_names), dtype=float),
            where=predicted_support > 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros(len(class_names), dtype=float),
            where=support > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(len(class_names), dtype=float),
            where=(precision + recall) > 0,
        )
        for class_index, class_name in enumerate(class_names):
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "precision": float(precision[class_index]),
                    "recall": float(recall[class_index]),
                    "f1": float(f1[class_index]),
                    "support": int(support[class_index]),
                }
            )
    write_csv(path, rows)


def save_confusion_matrices(
    path: Path,
    split_predictions: dict[str, dict[str, Any]],
    class_names: list[str],
) -> None:
    rows = []
    for split_name, prediction in split_predictions.items():
        matrix = confusion_array(
            np.asarray(prediction["y_true"]),
            np.asarray(prediction["y_pred"]),
            len(class_names),
        )
        for true_index, class_name in enumerate(class_names):
            rows.append(
                {
                    "split": split_name,
                    "true_class": class_name,
                    **{
                        f"pred_{predicted_name}": int(matrix[true_index, pred_index])
                        for pred_index, predicted_name in enumerate(class_names)
                    },
                }
            )
    write_csv(path, rows)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    for section in ("data", "model", "training", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section {section!r} in {path}.")
    return config


def build_search_grid(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    data_grid = expand_data_grid(config["data"])
    if len(data_grid) != 1:
        raise ValueError(
            "The hyperparameter runner requires one preprocessing configuration; "
            f"data expanded to {len(data_grid)} configurations."
        )
    model_grid = expand_grid(config["model"])
    training_grid = expand_grid(config["training"])
    combinations = [
        (model_cfg, training_cfg)
        for model_cfg, training_cfg in itertools.product(model_grid, training_grid)
    ]
    if not combinations:
        raise ValueError("The model/training search grid is empty.")
    return data_grid[0], combinations


def make_subject_folds(
    subjects: list[str] | np.ndarray,
    *,
    n_splits: int,
    shuffle: bool,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    subject_array = np.asarray(
        sorted(set(np.asarray(subjects).astype(str).tolist())),
        dtype=np.str_,
    )
    if n_splits < 2 or n_splits > len(subject_array):
        raise ValueError(
            f"cv_n_splits must be between 2 and {len(subject_array)}, got {n_splits}."
        )
    positions = np.arange(len(subject_array))
    if shuffle:
        positions = np.random.default_rng(seed).permutation(positions)
    folds = []
    for test_positions in np.array_split(positions, n_splits):
        train_positions = np.setdiff1d(positions, test_positions, assume_unique=True)
        train_subjects = sorted(subject_array[train_positions].tolist())
        test_subjects = sorted(subject_array[test_positions].tolist())
        if set(train_subjects) & set(test_subjects):
            raise RuntimeError("A CV fold contains overlapping train/test subjects.")
        folds.append((train_subjects, test_subjects))
    tested_subjects = [subject for _, fold_test in folds for subject in fold_test]
    if sorted(tested_subjects) != sorted(subject_array.tolist()):
        raise RuntimeError("Every subject must occur in exactly one CV test fold.")
    return folds


def config_hash(model_cfg: dict[str, Any], training_cfg: dict[str, Any]) -> str:
    payload = json.dumps(
        {"model": model_cfg, "training": training_cfg},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def flatten_config(prefix: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}.{key}": (
            json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        )
        for key, value in config.items()
    }


def validate_training_config(training_cfg: dict[str, Any]) -> None:
    if int(training_cfg.get("epochs", 0)) < 1:
        raise ValueError("training.epochs must be at least 1.")
    scheduler = str(training_cfg.get("lr_scheduler", "none")).strip().lower()
    supported_schedulers = {
        "",
        "none",
        "null",
        "off",
        "false",
        "cosine",
        "cosine_annealing",
        "step",
        "step_lr",
        "multistep",
        "multi_step",
        "multistep_lr",
        "exponential",
        "exponential_lr",
    }
    if scheduler not in supported_schedulers:
        raise ValueError(
            "training.lr_scheduler must be one of none, cosine, step, "
            f"multistep, or exponential; got {scheduler!r}."
        )
    if scheduler in {"cosine", "cosine_annealing"}:
        t_max = int(
            training_cfg.get("lr_scheduler_t_max", training_cfg.get("epochs", 0))
        )
        if t_max < 1:
            raise ValueError("training.lr_scheduler_t_max must be at least 1.")
        for key in ("lr_scheduler_min_lr", "stiefel_lr_scheduler_min_lr"):
            if float(training_cfg.get(key, 0.0)) < 0.0:
                raise ValueError(f"training.{key} must be non-negative.")
        min_lr_pairs = (
            ("lr_scheduler_min_lr", "learning_rate"),
            ("stiefel_lr_scheduler_min_lr", "stiefel_learning_rate"),
        )
        for min_lr_key, initial_lr_key in min_lr_pairs:
            if float(training_cfg.get(min_lr_key, 0.0)) > float(
                training_cfg.get(initial_lr_key, 0.0)
            ):
                raise ValueError(
                    f"training.{min_lr_key} must not exceed "
                    f"training.{initial_lr_key}."
                )
    if scheduler in {"step", "step_lr"}:
        if int(training_cfg.get("lr_scheduler_step_size", 30)) < 1:
            raise ValueError("training.lr_scheduler_step_size must be at least 1.")
    if scheduler in {"multistep", "multi_step", "multistep_lr"}:
        milestones = training_cfg.get("lr_scheduler_milestones")
        if not isinstance(milestones, (list, tuple)) or not milestones:
            raise ValueError(
                "training.lr_scheduler_milestones must be a non-empty list."
            )
        if any(int(value) < 1 for value in milestones):
            raise ValueError(
                "Every training.lr_scheduler_milestones value must be positive."
            )
    if scheduler in {
        "step",
        "step_lr",
        "multistep",
        "multi_step",
        "multistep_lr",
        "exponential",
        "exponential_lr",
    }:
        gamma = float(training_cfg.get("lr_scheduler_gamma", 0.1))
        if not 0.0 < gamma <= 1.0:
            raise ValueError("training.lr_scheduler_gamma must be in (0, 1].")


def class_counts(y: np.ndarray, indices: np.ndarray, num_classes: int) -> list[int]:
    return np.bincount(y[indices], minlength=num_classes).astype(int).tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu.")
    parser.add_argument("--epochs", type=int, help="Override epochs in every grid item.")
    parser.add_argument(
        "--max-configs",
        type=int,
        help="Run only the first N configurations, mainly for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded hyperparameter grid without loading EEG data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    raw_data_cfg, combinations = build_search_grid(config)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        combinations = [
            (model_cfg, {**training_cfg, "epochs": args.epochs})
            for model_cfg, training_cfg in combinations
        ]
    if args.max_configs is not None:
        if args.max_configs < 1:
            raise ValueError("--max-configs must be at least 1.")
        combinations = combinations[: args.max_configs]
    for _, training_cfg in combinations:
        validate_training_config(training_cfg)

    print(
        "Protocol: subject-wise Global cross-subject CV; train/test only, "
        "no validation, early stopping, final holdout, or fine-tuning."
    )
    print(f"Expanded hyperparameter configurations: {len(combinations)}")
    for index, (model_cfg, training_cfg) in enumerate(combinations, start=1):
        print(
            f"  [{index:03d}] model={json.dumps(model_cfg, sort_keys=True)} | "
            f"training={json.dumps(training_cfg, sort_keys=True)}"
        )
    if args.dry_run:
        return 0

    data_cfg = copy.deepcopy(raw_data_cfg)
    output_cfg = config["output"]
    output_root = Path(
        str(output_cfg.get("dir", "experiments/results/global_cross_subject_cv"))
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(
        str(output_cfg.get("dataset_cache_dir", "experiments/cache/preprocessed_datasets"))
    )
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    x, y, subject_labels, _run_labels, class_names = load_dataset(data_cfg, cache_dir)
    y = np.asarray(y, dtype=np.int64)
    subject_labels = np.asarray(subject_labels, dtype=np.str_)
    subjects = sorted(np.unique(subject_labels).astype(str).tolist())
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError(f"Cross-subject CV needs at least two classes: {class_names}")
    print(
        f"Loaded data: shape={tuple(x.shape)}, subjects={len(subjects)}, "
        f"classes={class_names}, counts="
        f"{np.bincount(y, minlength=num_classes).tolist()}."
    )
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    search_rows: list[dict[str, Any]] = []
    for config_index, (model_cfg, training_cfg) in enumerate(combinations, start=1):
        config_id = f"config_{config_index:03d}_{config_hash(model_cfg, training_cfg)}"
        config_dir = run_dir / config_id
        config_dir.mkdir()
        with (config_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"data": data_cfg, "model": model_cfg, "training": training_cfg},
                handle,
                sort_keys=False,
            )

        n_splits = int(training_cfg.get("cv_n_splits", 5))
        cv_shuffle = parse_bool(training_cfg.get("cv_shuffle", False), default=False)
        cv_seed = int(training_cfg.get("cv_seed", 42))
        folds = make_subject_folds(
            subjects,
            n_splits=n_splits,
            shuffle=cv_shuffle,
            seed=cv_seed,
        )
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        precision = normalize_precision_name(training_cfg.get("precision", "float32"))
        dtype = resolve_precision(precision)
        allow_tf32 = parse_bool(training_cfg.get("allow_tf32", False), default=False)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        full_dataset = MotorImageryDataset(x, y, dtype=dtype)
        num_workers = int(training_cfg.get("num_workers", 0))
        pin_memory = parse_bool(
            training_cfg.get("pin_memory", device.type == "cuda"),
            default=device.type == "cuda",
        )
        batch_size = int(training_cfg.get("batch_size", 128))
        condition_weight = float(
            training_cfg.get("condition_regularization_weight", 0.0)
        )
        criterion = nn.CrossEntropyLoss()
        print(
            f"\n[{config_index}/{len(combinations)}] {config_id}\n"
            f"  model={json.dumps(model_cfg, sort_keys=True)}\n"
            f"  training={json.dumps(training_cfg, sort_keys=True)}"
        )

        fold_rows: list[dict[str, Any]] = []
        fold_predictions: dict[str, dict[str, Any]] = {}
        subject_rows: list[dict[str, Any]] = []
        for fold_number, (train_subjects, test_subjects) in enumerate(folds, start=1):
            fold_dir = config_dir / f"fold_{fold_number:02d}"
            fold_dir.mkdir()
            train_idx = np.flatnonzero(np.isin(subject_labels, train_subjects)).astype(
                np.int64
            )
            test_idx = np.flatnonzero(np.isin(subject_labels, test_subjects)).astype(
                np.int64
            )
            if set(np.unique(y[train_idx]).tolist()) != set(range(num_classes)):
                raise ValueError(f"{config_id} fold {fold_number} train lacks classes.")
            if set(np.unique(y[test_idx]).tolist()) != set(range(num_classes)):
                raise ValueError(f"{config_id} fold {fold_number} test lacks classes.")
            fold_seed = int(training_cfg.get("seed", 42)) + fold_number
            print(
                f"  Fold {fold_number}/{n_splits}: train subjects/trials="
                f"{len(train_subjects)}/{len(train_idx)}, test subjects/trials="
                f"{len(test_subjects)}/{len(test_idx)}, seed={fold_seed}"
            )
            print(
                f"    test subjects={','.join(test_subjects)} | "
                f"train counts={class_counts(y, train_idx, num_classes)} | "
                f"test counts={class_counts(y, test_idx, num_classes)}"
            )
            set_seed(fold_seed)
            model = make_model(
                model_cfg,
                x,
                num_classes,
                device=device,
                dtype=dtype,
            )
            train_loader = make_loader(
                full_dataset,
                train_idx,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            train_eval_loader = make_loader(
                full_dataset,
                train_idx,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            test_loader = make_loader(
                full_dataset,
                test_idx,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            final_state = train_fixed_epochs(
                model,
                train_loader,
                training_cfg,
                device=device,
                history_path=fold_dir / "train_history.csv",
                stage_name=f"{config_id} fold {fold_number}",
            )
            train_predictions = predict_loader(
                model,
                train_eval_loader,
                criterion,
                device,
                condition_regularization_weight=condition_weight,
            )
            test_predictions = predict_loader(
                model,
                test_loader,
                criterion,
                device,
                condition_regularization_weight=condition_weight,
            )
            train_metrics = metrics_from_arrays(
                train_predictions["y_true"],
                train_predictions["y_pred"],
                num_classes=num_classes,
            )
            test_metrics = metrics_from_arrays(
                test_predictions["y_true"],
                test_predictions["y_pred"],
                num_classes=num_classes,
            )
            fold_rows.append(
                {
                    "fold": fold_number,
                    "seed": fold_seed,
                    "n_train_subjects": len(train_subjects),
                    "n_test_subjects": len(test_subjects),
                    "n_train_trials": int(len(train_idx)),
                    "n_test_trials": int(len(test_idx)),
                    "train_accuracy": train_metrics["accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_cohen_kappa": test_metrics["cohen_kappa"],
                    "train_subjects": ",".join(train_subjects),
                    "test_subjects": ",".join(test_subjects),
                }
            )
            fold_predictions[f"fold_{fold_number:02d}_test"] = test_predictions
            prediction_by_index = np.full(len(y), -1, dtype=np.int64)
            prediction_by_index[test_idx] = np.asarray(
                test_predictions["y_pred"], dtype=np.int64
            )
            for subject in test_subjects:
                subject_idx = np.flatnonzero(subject_labels == subject).astype(np.int64)
                metrics = metrics_from_arrays(
                    y[subject_idx],
                    prediction_by_index[subject_idx],
                    num_classes=num_classes,
                )
                subject_rows.append(
                    {
                        "fold": fold_number,
                        "subject": subject,
                        "n_trials": int(len(subject_idx)),
                        **metrics,
                    }
                )
            with (fold_dir / "split.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "fold": fold_number,
                        "train_subjects": train_subjects,
                        "test_subjects": test_subjects,
                        "train_indices": train_idx.astype(int).tolist(),
                        "test_indices": test_idx.astype(int).tolist(),
                    },
                    handle,
                    indent=2,
                )
            save_per_class_metrics(
                fold_dir / "per_class_metrics.csv",
                {"train": train_predictions, "test": test_predictions},
                class_names,
            )
            save_confusion_matrices(
                fold_dir / "confusion_matrices.csv",
                {"train": train_predictions, "test": test_predictions},
                class_names,
            )
            if parse_bool(output_cfg.get("save_fold_checkpoints", False), default=False):
                torch.save(
                    {
                        "model_state_dict": final_state,
                        "class_names": class_names,
                        "model": model_cfg,
                        "training": training_cfg,
                        "fold": fold_number,
                    },
                    fold_dir / "model.pt",
                )
            print(
                f"    train accuracy={train_metrics['accuracy']:.4f}, "
                f"test accuracy={test_metrics['accuracy']:.4f}, "
                f"test macro-F1={test_metrics['macro_f1']:.4f}."
            )
            del model, train_loader, train_eval_loader, test_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_mean, train_sd = mean_sd(
            [float(row["train_accuracy"]) for row in fold_rows]
        )
        test_mean, test_sd = mean_sd(
            [float(row["test_accuracy"]) for row in fold_rows]
        )
        balanced_mean, balanced_sd = mean_sd(
            [float(row["test_balanced_accuracy"]) for row in fold_rows]
        )
        macro_f1_mean, macro_f1_sd = mean_sd(
            [float(row["test_macro_f1"]) for row in fold_rows]
        )
        summary = {
            "config_id": config_id,
            "selection_metric": "arithmetic_mean_fold_test_accuracy",
            "train_accuracy_mean": train_mean,
            "train_accuracy_sd": train_sd,
            "test_accuracy_mean": test_mean,
            "test_accuracy_sd": test_sd,
            "test_balanced_accuracy_mean": balanced_mean,
            "test_balanced_accuracy_sd": balanced_sd,
            "test_macro_f1_mean": macro_f1_mean,
            "test_macro_f1_sd": macro_f1_sd,
            "model": model_cfg,
            "training": training_cfg,
            "folds": fold_rows,
        }
        write_csv(config_dir / "fold_results.csv", fold_rows)
        write_csv(config_dir / "per_subject_results.csv", subject_rows)
        save_per_class_metrics(
            config_dir / "fold_test_per_class_metrics.csv",
            fold_predictions,
            class_names,
        )
        save_confusion_matrices(
            config_dir / "fold_test_confusion_matrices.csv",
            fold_predictions,
            class_names,
        )
        with (config_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        search_rows.append(
            {
                "config_id": config_id,
                "train_accuracy_mean": train_mean,
                "train_accuracy_sd": train_sd,
                "test_accuracy_mean": test_mean,
                "test_accuracy_sd": test_sd,
                "test_balanced_accuracy_mean": balanced_mean,
                "test_macro_f1_mean": macro_f1_mean,
                **flatten_config("model", model_cfg),
                **flatten_config("training", training_cfg),
            }
        )
        print(
            f"  {n_splits}-FOLD RESULT: train accuracy="
            f"{train_mean:.4f} +/- {train_sd:.4f}; "
            f"test accuracy={test_mean:.4f} +/- {test_sd:.4f}."
        )

    ranked_rows = sorted(
        search_rows,
        key=lambda row: float(row["test_accuracy_mean"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
    ranked_rows = [
        {"rank": row.pop("rank"), **row}
        for row in ranked_rows
    ]
    write_csv(run_dir / "search_results.csv", ranked_rows)
    best = ranked_rows[0]
    with (run_dir / "best_hyperparameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "selection_metric": "arithmetic_mean_fold_test_accuracy",
                "important_note": (
                    "These test folds were used for hyperparameter selection and "
                    "are not an independent final test set."
                ),
                "best": best,
            },
            handle,
            indent=2,
        )
    print(
        f"\nBEST CONFIG: {best['config_id']} | mean fold test accuracy="
        f"{float(best['test_accuracy_mean']):.4f} +/- "
        f"{float(best['test_accuracy_sd']):.4f}."
    )
    print(f"Saved hyperparameter ranking: {run_dir / 'search_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
