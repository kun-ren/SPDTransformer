from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import geoopt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from script.load_moabb_dataset import parse_int_list
from src.datasets.PhysioNetMI_preprocess import preprocess_spd
from src.models.MotorImageryDataset import MotorImageryDataset
from src.models.SPDTransformerClassifier import SPDTransformerClassifier

from src.training.shared_split import load_or_create_split_indices

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_grid.yaml"
DATASET_CACHE_VERSION = 3
DEFAULT_DATASET_CACHE_DIR = PROJECT_ROOT / "experiments" / "cache" / "preprocessed_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-train SPDTransformerClassifier from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config. List-valued keys in data/model/training are grid values.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def normalize_filter_bank(filter_bank: Any) -> list[list[float]]:
    if not isinstance(filter_bank, list) or not filter_bank:
        raise ValueError("data.filter_bank must be a non-empty list.")

    normalized = []
    for band in filter_bank:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError(
                "Each filter bank item must be [low_freq, high_freq], "
                f"got {band!r}."
            )
        normalized.append([float(band[0]), float(band[1])])
    return normalized


def is_filter_bank_value(key: str, value: Any) -> bool:
    if key != "filter_bank" or not isinstance(value, list):
        return False
    return bool(value) and all(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and all(isinstance(number, (int, float)) for number in item)
        for item in value
    )


def grid_values(key: str, value: Any) -> list[Any]:
    if is_filter_bank_value(key, value):
        return [normalize_filter_bank(value)]
    if isinstance(value, list):
        return value
    return [value]


def expand_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    if not section:
        return [{}]

    keys = list(section)
    value_lists = [grid_values(key, section[key]) for key in keys]
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    return combinations


def expand_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_grid(config.get("data", {}))
    model_grid = expand_grid(config.get("model", {}))
    training_grid = expand_grid(config.get("training", {}))

    experiments = []
    for data_cfg, model_cfg, training_cfg in itertools.product(
            data_grid,
            model_grid,
            training_grid,
    ):
        experiments.append(
            {
                "data": deepcopy(data_cfg),
                "model": deepcopy(model_cfg),
                "training": deepcopy(training_cfg),
                "output": deepcopy(config.get("output", {})),
            }
        )
    return experiments


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def dataset_cache_key(data_cfg: dict[str, Any]) -> str:
    return config_hash({"data": data_cfg})


def resolve_project_path(path_value: Any, default: Path) -> Path:
    if path_value in {None, ""}:
        return default
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dataset_cache_path(cache_dir: Path, data_key: str) -> Path:
    return cache_dir / f"spd_dataset_{data_key}.npz"


def make_run_dir(base_dir: Path, run_index: int, config: dict[str, Any]) -> Path:
    run_id = f"run_{run_index:03d}_{config_hash(config)}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_split_file(split_file: Any) -> Path | None:
    if split_file in {None, ""}:
        return None
    path = Path(str(split_file))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_precision(precision: Any) -> torch.dtype:
    precision = str(precision or "float64").lower()
    if precision in {"float64", "double", "fp64"}:
        return torch.float64
    if precision in {"float32", "float", "single", "fp32"}:
        return torch.float32
    raise ValueError(
        "training.precision must be one of: float64, double, fp64, "
        "float32, float, single, fp32."
    )


def parse_task_types(task_types: Any) -> tuple[str, ...]:
    if task_types is None:
        return ("unilateral_fist", "both")
    if isinstance(task_types, str):
        return tuple(part.strip() for part in task_types.split(",") if part.strip())
    return tuple(str(part).strip() for part in task_types if str(part).strip())


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


def make_loaders(
        x: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        batch_size: int,
        num_workers: int,
        dtype: torch.dtype,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        MotorImageryDataset(x[train_idx], y[train_idx], dtype=dtype),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        MotorImageryDataset(x[val_idx], y[val_idx], dtype=dtype),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        MotorImageryDataset(x[test_idx], y[test_idx], dtype=dtype),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def build_model(
        model_cfg: dict[str, Any],
        spd_in_dim: int,
        num_classes: int,
        time_sequence_length,
        frequency_sequence_length,
) -> SPDTransformerClassifier:
    depth = int(model_cfg.get("depth", 1))
    attention_dim = parse_attention_dims(
        model_cfg.get("attention_dim", spd_in_dim),
        depth=depth,
    )
    return SPDTransformerClassifier(
        num_heads=int(model_cfg.get("head_nums", 1)),
        spd_in_dim=spd_in_dim,
        attention_dim=attention_dim,
        stage_transition=bool(model_cfg.get("stage_transition", True)),
        time_sequence_length=time_sequence_length,
        frequency_sequence_length=frequency_sequence_length,
        tau=model_cfg.get("tau", 1.0),
        num_classes=num_classes,
        ffn_hidden_spd_dim=model_cfg.get("ffn_hidden_spd_dim"),
        metric=str(model_cfg.get("metric", "log-euclidean")),
        depth=depth,
        classifier_type=str(model_cfg.get("classifier_type", "pooling")),
        pooling=str(model_cfg.get("pooling", "attention")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
        debug_attention_dropout=bool(model_cfg.get("debug_attention_dropout", False)),
        debug_attention_shape=bool(model_cfg.get("debug_attention_shape", False)),
        debug_tensor_stats=bool(model_cfg.get("debug_tensor_stats", False)),
        learnable_metric_mode=str(model_cfg.get("learnable_metric_mode", "low-rank")),
        learnable_metric_rank=model_cfg.get("learnable_metric_rank"),
        eps=float(model_cfg.get("eps", 1e-6)),
        use_position_bias=bool(model_cfg.get("use_position_bias", True)),
        layer_norm_affine=bool(model_cfg.get("layer_norm_affine", True)),
        stage_projection_init=str(model_cfg.get("stage_projection_init", "identity")),
    )


def parse_attention_dims(value: Any, depth: int) -> list[int]:
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}.")

    if isinstance(value, str):
        dims = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        dims = [int(item) for item in value]
    else:
        dims = [int(value)]

    if not dims:
        raise ValueError("model.attention_dim must contain at least one dimension.")

    if len(dims) < depth:
        dims.extend([dims[-1]] * (depth - len(dims)))
    elif len(dims) > depth:
        dims = dims[:depth]

    return dims


def condition_regularization(P, eps=1e-5):
    """
    P: (..., d, d), SPD matrix after BiMap
    """
    eigvals = torch.linalg.eigvalsh(P)
    eigvals = torch.clamp(eigvals, min=eps)

    cond = eigvals[..., -1] / eigvals[..., 0]
    return torch.log(cond).mean()


def split_params(model: nn.Module):
    stiefel_params = []
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Geoopt manifold parameters: BiMap W
        if isinstance(param, geoopt.ManifoldParameter):
            stiefel_params.append(param)
            continue

        # Bias and normalization layers: no weight decay
        if (
                "norm" in name.lower() or "metric_low_rank" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return stiefel_params, decay_params, no_decay_params


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        condition_regularization_weight: float = 1e-3,
) -> dict[str, float]:
    predictions = predict_loader(
        model,
        loader,
        criterion,
        device,
        condition_regularization_weight=condition_regularization_weight,
    )
    return {
        "loss": predictions["loss"],
        "accuracy": predictions["accuracy"],
        "macro_f1": predictions["macro_f1"],
    }


def predict_loader(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        condition_regularization_weight: float = 1e-3,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, aux = model(x_batch)
            cond_loss = logits.new_tensor(0.0)
            if condition_regularization_weight > 0:
                for name, P_bimap in aux.items():
                    cond_loss = cond_loss + condition_regularization(P_bimap)
                cond_loss = cond_loss / len(aux)

            loss = (
                    criterion(logits, y_batch)
                    + condition_regularization_weight * cond_loss
            )

            total_loss += loss.item() * y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_per_class_metrics(
        path: Path,
        split_predictions: dict[str, dict[str, Any]],
        class_names: list[str],
) -> None:
    labels = np.arange(len(class_names))
    rows = []
    for split_name, prediction in split_predictions.items():
        precision, recall, f1, support = precision_recall_fscore_support(
            prediction["y_true"],
            prediction["y_pred"],
            labels=labels,
            zero_division=0,
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

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrices(
        path: Path,
        split_predictions: dict[str, dict[str, Any]],
        class_names: list[str],
) -> None:
    labels = np.arange(len(class_names))
    rows = []
    pred_columns = [f"pred_{class_name}" for class_name in class_names]
    for split_name, prediction in split_predictions.items():
        matrix = confusion_matrix(
            prediction["y_true"],
            prediction["y_pred"],
            labels=labels,
        )
        for true_index, class_name in enumerate(class_names):
            row = {
                "split": split_name,
                "true_class": class_name,
            }
            row.update(
                {
                    column: int(matrix[true_index, pred_index])
                    for pred_index, column in enumerate(pred_columns)
                }
            )
            rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "true_class", *pred_columns],
        )
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer_euclid: torch.optim.Optimizer,
        optimizer_stiefel: geoopt.optim.RiemannianAdam,
        device: torch.device,
        gradient_clip_norm: float | None = None,
        debug_anomaly: bool = False,
        condition_regularization_weight: float = 1e-3,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits, aux = model(x_batch)
        cls_loss = criterion(logits, y_batch)

        cond_loss = logits.new_tensor(0.0)
        if condition_regularization_weight > 0:
            for name, P_bimap in aux.items():
                cond_loss = cond_loss + condition_regularization(P_bimap)
            cond_loss = cond_loss / len(aux)

        loss = cls_loss + condition_regularization_weight * cond_loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss detected: "
                f"cls_loss={cls_loss.item():.6e} "
                f"cond_loss={cond_loss.item():.6e} "
                f"loss={loss.item():.6e}. "
                "Check input SPD matrices, learning rate, and model numerical stability."
            )

        optimizer_euclid.zero_grad()
        optimizer_stiefel.zero_grad()

        if debug_anomaly:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
        assert_model_finite(model, "backward")

        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            params_to_clip = [
                p
                for optimizer in (optimizer_euclid, optimizer_stiefel)
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]

            torch.nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=gradient_clip_norm,
            )

        optimizer_euclid.step()
        optimizer_stiefel.step()
        assert_model_finite(model, "optimizer step")

        total_loss += loss.item() * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def assert_model_finite(model: nn.Module, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                "Non-finite parameter detected after "
                f"{context}: {name} | {_tensor_finite_summary(parameter)}"
            )
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                "Non-finite gradient detected after "
                f"{context}: {name} | {_tensor_finite_summary(parameter.grad)}"
            )


def _tensor_finite_summary(tensor: torch.Tensor) -> str:
    with torch.no_grad():
        finite = torch.isfinite(tensor)
        finite_count = int(finite.sum().item())
        total_count = tensor.numel()
        nan_count = int(torch.isnan(tensor).sum().item())
        posinf_count = int(torch.isposinf(tensor).sum().item())
        neginf_count = int(torch.isneginf(tensor).sum().item())

        if finite_count == 0:
            finite_range = "finite_min=NA finite_max=NA"
        else:
            finite_values = tensor.detach()[finite]
            finite_range = (
                f"finite_min={finite_values.min().item():.6e} "
                f"finite_max={finite_values.max().item():.6e}"
            )

        return (
            f"finite={finite_count}/{total_count} "
            f"nan={nan_count} +inf={posinf_count} -inf={neginf_count} "
            f"{finite_range}"
        )


def append_history(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def optimizer_lr_values(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def format_lr_values(values: list[float]) -> str:
    return ",".join(f"{value:.3e}" for value in values)


def resolve_scheduler_metric(metric_name: str, val_metrics: dict[str, float]) -> float:
    metric_map = {
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
    }
    if metric_name not in metric_map:
        raise ValueError(
            "lr_scheduler_metric must be one of "
            "'val_loss', 'val_accuracy', or 'val_macro_f1', "
            f"got {metric_name!r}."
        )
    return float(metric_map[metric_name])


def build_lr_schedulers(
        training_cfg: dict[str, Any],
        optimizer_euclid: torch.optim.Optimizer,
        optimizer_stiefel: torch.optim.Optimizer,
) -> tuple[str | None, str | None, Any, Any]:
    scheduler_name = str(training_cfg.get("lr_scheduler", "none")).lower()
    if scheduler_name in {"", "none", "null", "false", "off"}:
        return None, None, None, None

    if scheduler_name not in {
        "plateau",
        "reduce_on_plateau",
        "reduce_lr_on_plateau",
    }:
        raise ValueError(
            "Only lr_scheduler='plateau' is supported, "
            f"got {scheduler_name!r}."
        )

    metric_name = str(training_cfg.get("lr_scheduler_metric", "val_macro_f1"))
    mode = training_cfg.get("lr_scheduler_mode")
    if mode is None:
        mode = "min" if metric_name == "val_loss" else "max"
    mode = str(mode).lower()
    if mode not in {"min", "max"}:
        raise ValueError("lr_scheduler_mode must be 'min' or 'max'.")

    factor = float(training_cfg.get("lr_scheduler_factor", 0.5))
    if not 0.0 < factor < 1.0:
        raise ValueError("lr_scheduler_factor must be between 0 and 1.")

    scheduler_kwargs = {
        "mode": mode,
        "factor": factor,
        "patience": int(training_cfg.get("lr_scheduler_patience", 4)),
        "threshold": float(training_cfg.get("lr_scheduler_threshold", 1e-4)),
        "threshold_mode": str(
            training_cfg.get("lr_scheduler_threshold_mode", "rel")
        ),
        "cooldown": int(training_cfg.get("lr_scheduler_cooldown", 0)),
        "eps": float(training_cfg.get("lr_scheduler_eps", 1e-8)),
    }
    euclid_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_euclid,
        min_lr=float(training_cfg.get("lr_scheduler_min_lr", 1e-6)),
        **scheduler_kwargs,
    )
    stiefel_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_stiefel,
        min_lr=float(
            training_cfg.get(
                "stiefel_lr_scheduler_min_lr",
                training_cfg.get("lr_scheduler_min_lr", 1e-6),
            )
        ),
        **scheduler_kwargs,
    )
    return scheduler_name, metric_name, euclid_scheduler, stiefel_scheduler


def train_experiment(
        run_index: int,
        experiment_cfg: dict[str, Any],
        x: np.ndarray,
        y: np.ndarray,
        subject_labels: np.ndarray,
        class_names: list[str],
        base_output_dir: Path,
        device: torch.device,
) -> dict[str, Any]:
    training_cfg = experiment_cfg["training"]
    model_cfg = experiment_cfg["model"]
    seed = int(training_cfg.get("seed", 42))
    dtype = resolve_precision(training_cfg.get("precision", "float64"))
    set_seed(seed)

    run_dir = make_run_dir(base_output_dir, run_index, experiment_cfg)
    save_yaml(run_dir / "config.yaml", experiment_cfg)

    split_file = resolve_split_file(training_cfg.get("split_file"))
    allow_subject_overlap = parse_bool(
        training_cfg.get("allow_subject_overlap", True),
        default=True,
    )
    global_max = np.max(x)
    global_min = np.min(np.abs(x))
    print(f"max: {global_max}")
    print(f"min: {global_min}")

    train_idx, val_idx, test_idx = load_or_create_split_indices(
        y=y,
        test_size=float(training_cfg.get("test_size", 0.15)),
        val_size=float(training_cfg.get("val_size", 0.15)),
        seed=seed,
        split_file=split_file,
        subjects=subject_labels,
        allow_subject_overlap=allow_subject_overlap,
    )
    train_subjects = set(subject_labels[train_idx].tolist())
    val_subjects = set(subject_labels[val_idx].tolist())
    test_subjects = set(subject_labels[test_idx].tolist())
    if not allow_subject_overlap:
        train_test_overlap = train_subjects & test_subjects
        train_val_overlap = train_subjects & val_subjects
        val_test_overlap = val_subjects & test_subjects
        if train_test_overlap or train_val_overlap or val_test_overlap:
            raise RuntimeError(
                "Subject-level split failed: subject overlap detected between "
                f"splits. train-test={sorted(train_test_overlap)}, "
                f"train-val={sorted(train_val_overlap)}, "
                f"val-test={sorted(val_test_overlap)}."
            )

    train_loader, val_loader, test_loader = make_loaders(
        x=x,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(training_cfg.get("batch_size", 16)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        dtype=dtype,
    )

    time_sequence_length = x.shape[-4]
    frequency_sequence_length = x.shape[-3]
    model = build_model(
        model_cfg=model_cfg,
        spd_in_dim=x.shape[-1],
        time_sequence_length=time_sequence_length,
        frequency_sequence_length=frequency_sequence_length,
        num_classes=len(class_names),
    ).to(device=device, dtype=dtype)

    criterion = nn.CrossEntropyLoss()
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    # apply_weight_decay_to_special_parameters = bool(
    #    training_cfg.get("apply_weight_decay_to_special_parameters", False)
    # )

    stiefel_params, decay_params, no_decay_params = split_params(model)

    optimizer_euclid = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=float(training_cfg.get("learning_rate", 1e-3)),
    )

    optimizer_stiefel = geoopt.optim.RiemannianAdam(
        stiefel_params,
        lr=float(training_cfg.get("stiefel_learning_rate", 1e-3)),
        weight_decay=0.0,
        stabilize=10,
    )
    (
        lr_scheduler_name,
        lr_scheduler_metric,
        lr_scheduler_euclid,
        lr_scheduler_stiefel,
    ) = build_lr_schedulers(
        training_cfg,
        optimizer_euclid,
        optimizer_stiefel,
    )

    best_val_macro_f1 = -1.0
    best_epoch = 0
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_model.pt"
    epochs = int(training_cfg.get("epochs", 50))
    condition_regularization_weight = float(
        training_cfg.get("condition_regularization_weight", 1e-3)
    )
    early_stopping_patience = training_cfg.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_delta = float(
        training_cfg.get("early_stopping_min_delta", 0.0)
    )
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    debug_anomaly = parse_bool(
        training_cfg.get("debug_anomaly", False),
        default=False,
    )

    print(f"\n[Run {run_index}] {run_dir.name}")
    print(f"  model={model_cfg}")
    print(f"  training={training_cfg}")
    if lr_scheduler_name is not None:
        print(
            "  lr_scheduler="
            f"{lr_scheduler_name} metric={lr_scheduler_metric} "
            f"euclid_lr={format_lr_values(optimizer_lr_values(optimizer_euclid))} "
            f"stiefel_lr={format_lr_values(optimizer_lr_values(optimizer_stiefel))}"
        )
    print(
        "  split="
        f"{'epoch-level' if allow_subject_overlap else 'subject-level'} "
        f"subjects train/val/test="
        f"{len(train_subjects)}/{len(val_subjects)}/{len(test_subjects)}"
    )

    print(f"thread: {torch.get_num_threads()}")
    print(torch.get_num_interop_threads())

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer_euclid,
            optimizer_stiefel,
            device,
            gradient_clip_norm=gradient_clip_norm,
            debug_anomaly=debug_anomaly,
            condition_regularization_weight=condition_regularization_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            condition_regularization_weight=condition_regularization_weight,
        )

        if lr_scheduler_name is not None:
            old_euclid_lrs = optimizer_lr_values(optimizer_euclid)
            old_stiefel_lrs = optimizer_lr_values(optimizer_stiefel)
            scheduler_value = resolve_scheduler_metric(
                lr_scheduler_metric,
                val_metrics,
            )
            lr_scheduler_euclid.step(scheduler_value)
            lr_scheduler_stiefel.step(scheduler_value)
            new_euclid_lrs = optimizer_lr_values(optimizer_euclid)
            new_stiefel_lrs = optimizer_lr_values(optimizer_stiefel)
            if (
                    new_euclid_lrs != old_euclid_lrs
                    or new_stiefel_lrs != old_stiefel_lrs
            ):
                print(
                    "  lr scheduler step | "
                    f"{lr_scheduler_metric}={scheduler_value:.6f} "
                    f"euclid_lr {format_lr_values(old_euclid_lrs)}"
                    f" -> {format_lr_values(new_euclid_lrs)} "
                    f"stiefel_lr {format_lr_values(old_stiefel_lrs)}"
                    f" -> {format_lr_values(new_stiefel_lrs)}"
                )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "euclid_lr": optimizer_lr_values(optimizer_euclid)[0],
            "stiefel_lr": optimizer_lr_values(optimizer_stiefel)[0],
        }
        append_history(history_path, row)

        if val_metrics["macro_f1"] > best_val_macro_f1 + early_stopping_min_delta:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": experiment_cfg,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                checkpoint_path,
            )

        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"mf1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"mf1={val_metrics['macro_f1']:.4f}"
        )

        if (
                early_stopping_patience is not None
                and early_stopping_patience > 0
                and epoch - best_epoch >= early_stopping_patience
        ):
            print(
                f"  early stopping at epoch {epoch:03d}/{epochs} | "
                f"best_epoch={best_epoch} "
                f"best_val_mf1={best_val_macro_f1:.4f}"
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        # This checkpoint is written by this training run and contains
        # metadata such as config/class names in addition to tensor weights.
        # weights_only=True rejects those numpy/Python objects.
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    split_predictions = {
        "train": predict_loader(
            model,
            train_loader,
            criterion,
            device,
            condition_regularization_weight=condition_regularization_weight,
        ),
        "val": predict_loader(
            model,
            val_loader,
            criterion,
            device,
            condition_regularization_weight=condition_regularization_weight,
        ),
        "test": predict_loader(
            model,
            test_loader,
            criterion,
            device,
            condition_regularization_weight=condition_regularization_weight,
        ),
    }
    save_per_class_metrics(
        run_dir / "per_class_metrics.csv",
        split_predictions,
        class_names,
    )
    save_confusion_matrices(
        run_dir / "confusion_matrix.csv",
        split_predictions,
        class_names,
    )
    test_metrics = split_predictions["test"]

    metrics = {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "class_names": class_names,
        "precision": str(dtype).replace("torch.", ""),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "allow_subject_overlap": allow_subject_overlap,
        "n_train_subjects": int(len(train_subjects)),
        "n_val_subjects": int(len(val_subjects)),
        "n_test_subjects": int(len(test_subjects)),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        f"[Run {run_index}] done | best_epoch={best_epoch} "
        f"best_val_mf1={best_val_macro_f1:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_mf1={test_metrics['macro_f1']:.4f}"
    )
    return metrics


def preprocess_dataset(
        data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])

    subjects = parse_int_list(data_cfg["subjects"])
    # for dataset in data_cfg["datasets"]:
    #     dataset["filter_bank"] = filter_bank
    task_types = parse_task_types(data_cfg.get("task_types"))
    x, y, class_names, subject_labels = preprocess_spd(
        filter_bank=filter_bank,
        root_dir=str(data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")),
        subjects=subjects,
        channels=data_cfg.get("channels"),
        estimator=str(data_cfg.get("estimator", "lwf")),
        sfreq=float(data_cfg.get("sfreq", 160)),
        eps=float(data_cfg.get("eps", 1e-8)),
        segment_duration=float(data_cfg.get("segment_duration", 1.0)),
        stride_duration=data_cfg.get("stride_duration", 0.5),
        imaged=parse_bool(data_cfg.get("imaged", True), default=True),
        executed=parse_bool(data_cfg.get("executed", False), default=False),
        task_types=task_types,
        reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
        baseline_correction=data_cfg.get("baseline_correction"),
        baseline_window=data_cfg.get("baseline_window"),
        epoch_tmin=float(data_cfg.get("epoch_tmin", -2.0)),
        epoch_tmax=float(data_cfg.get("epoch_tmax", 4.0)),
        use_ica=parse_bool(data_cfg.get("use_ica", False), default=False),
        ica_n_components=data_cfg.get("ica_n_components", 20),
        ica_random_state=int(data_cfg.get("ica_random_state", 42)),
        ica_eog_channels=data_cfg.get("ica_eog_channels"),
        use_autoreject=parse_bool(
            data_cfg.get("use_autoreject", False),
            default=False,
        ),
        autoreject_random_state=int(data_cfg.get("autoreject_random_state", 42)),
        autoreject_n_jobs=int(data_cfg.get("autoreject_n_jobs", 1)),
        autoreject_cv=int(data_cfg.get("autoreject_cv", 10)),
        return_subjects=True,
        covariance_signal_scale=float(
            data_cfg.get("covariance_signal_scale", 1e6)
        ),
    )
    if not np.isfinite(x).all():
        bad_count = int((~np.isfinite(x)).sum())
        raise ValueError(
            f"Preprocessed SPD dataset contains {bad_count} NaN or Inf values."
        )
    return (
        x.astype(np.float32),
        y.astype(np.int64),
        np.asarray(subject_labels, dtype=np.str_),
        list(class_names),
    )


def load_cached_dataset(
        cache_path: Path,
        data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata_json = str(payload["metadata_json"].item())
            metadata = json.loads(metadata_json)
            if metadata.get("cache_version") != DATASET_CACHE_VERSION:
                print(
                    f"  Dataset cache format changed, rebuilding: {cache_path}"
                )
                return None
            if metadata.get("data_config") != data_cfg:
                print(
                    f"  Dataset cache key collision or config mismatch, rebuilding: {cache_path}"
                )
                return None

            x = np.asarray(payload["x"], dtype=np.float32)
            y = np.asarray(payload["y"], dtype=np.int64)
            subject_labels = np.asarray(payload["subject_labels"], dtype=np.str_)
            class_names = [str(name) for name in payload["class_names"].tolist()]

        if not np.isfinite(x).all():
            bad_count = int((~np.isfinite(x)).sum())
            print(
                f"  Cached SPD dataset contains {bad_count} NaN or Inf values, rebuilding."
            )
            return None

        if len(subject_labels) != len(y):
            print(
                "  Cached subject label count does not match y length, rebuilding."
            )
            return None

        return x, y, subject_labels, class_names
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"  Failed to read dataset cache {cache_path}: {error}. Rebuilding.")
        return None


def save_cached_dataset(
        cache_path: Path,
        data_cfg: dict[str, Any],
        x: np.ndarray,
        y: np.ndarray,
        subject_labels: np.ndarray,
        class_names: list[str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": DATASET_CACHE_VERSION,
        "data_config": data_cfg,
    }
    tmp_path = cache_path.with_name(
        f"{cache_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.tmp"
    )
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            x=x.astype(np.float32, copy=False),
            y=y.astype(np.int64, copy=False),
            subject_labels=np.asarray(subject_labels, dtype=np.str_),
            class_names=np.asarray(class_names, dtype=np.str_),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, default=str),
                dtype=np.str_,
            ),
        )
    tmp_path.replace(cache_path)


def load_or_preprocess_dataset(
        data_cfg: dict[str, Any],
        cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data_key = dataset_cache_key(data_cfg)
    cache_path = dataset_cache_path(cache_dir, data_key)
    cached_dataset = load_cached_dataset(cache_path, data_cfg)
    if cached_dataset is not None:
        print(f"\nLoaded preprocessed data from cache {data_key}: {cache_path}")
        x_cached, y_cached, subjects_cached, names_cached = cached_dataset
        print(
            f"  X.shape={x_cached.shape}, y.shape={y_cached.shape}, "
            f"subjects={len(set(subjects_cached.tolist()))}, classes={names_cached}"
        )
        return cached_dataset

    print(f"\nPreprocessing data config {data_key}: {data_cfg}")
    dataset = preprocess_dataset(data_cfg)
    x_cached, y_cached, subjects_cached, names_cached = dataset
    print(
        f"  X.shape={x_cached.shape}, y.shape={y_cached.shape}, "
        f"subjects={len(set(subjects_cached.tolist()))}, classes={names_cached}"
    )
    save_cached_dataset(
        cache_path,
        data_cfg,
        x_cached,
        y_cached,
        subjects_cached,
        names_cached,
    )
    print(f"  Saved preprocessed data cache: {cache_path}")
    return dataset


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    experiments = expand_experiments(config)
    if not experiments:
        raise ValueError("No experiment configurations were generated.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_cfg = config.get("output", {})
    base_output_dir = PROJECT_ROOT / str(output_cfg.get("dir", "experiments/results")) / timestamp
    base_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = resolve_project_path(
        output_cfg.get("dataset_cache_dir"),
        DEFAULT_DATASET_CACHE_DIR,
    )

    # Preprocess once from the first expanded data config. If you grid data
    # preprocessing parameters, each distinct data config will be handled below.

    # cache loaded dataset
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    all_metrics = []

    print(f"Generated {len(experiments)} experiment(s)")
    print(f"Saving runs under: {base_output_dir}")
    print(f"Dataset cache: {dataset_cache_dir}")
    print(f"Device: {device}")

    for run_index, experiment_cfg in enumerate(experiments, start=1):
        data_key = dataset_cache_key(experiment_cfg["data"])
        if data_key not in data_cache:
            data_cache[data_key] = load_or_preprocess_dataset(
                experiment_cfg["data"],
                dataset_cache_dir,
            )

        x, y, subject_labels, class_names = data_cache[data_key]

        print(f"y: {y[:10]}")
        print(f"subjects labels: {subject_labels}")
        print(f"class names: {class_names}")
        print(f"y types: {set(y)}")
        metrics = train_experiment(
            run_index=run_index,
            experiment_cfg=experiment_cfg,
            x=x,
            y=y,
            subject_labels=subject_labels,
            class_names=class_names,
            base_output_dir=base_output_dir,
            device=device,
        )
        all_metrics.append(metrics)

    summary_path = base_output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2)

    print(f"\nAll runs complete. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
