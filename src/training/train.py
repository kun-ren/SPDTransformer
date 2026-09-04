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
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.BCICompetitionIV2a_preprocess import preprocess_bci_iv_2a_spd
from src.datasets.PhysioNetMI_preprocess import (
    normalize_float_dtype,
    preprocess_spd,
)
from src.models.MotorImageryDataset import MotorImageryDataset
from src.models.SPDTransformerClassifier import SPDTransformerClassifier
from src.training.config_grid import (
    expand_data_grid,
    expand_grid,
    normalize_data_time_config,
    normalize_filter_bank,
)
from src.training.losses import (
    compute_training_objective,
    prototype_loss_settings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_grid.yaml"
DATASET_CACHE_VERSION = 6
DEFAULT_DATASET_CACHE_DIR = PROJECT_ROOT / "experiments" / "cache" / "preprocessed_datasets"
CV_METRIC_NAMES = ("accuracy", "macro_f1", "cohen_kappa")


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
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help=(
            "Override training precision for all expanded runs. "
            "Accepted values: float32/fp32/single or float64/fp64/double."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return config


def expand_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_data_grid(config.get("data", {}))
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
    return config_hash({"data": dataset_preprocessing_config(data_cfg)})


def dataset_preprocessing_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that can change the preprocessed tensors."""
    canonical = deepcopy(data_cfg)
    normalize_data_preprocessing_config(canonical)
    split_only_keys = {
        "seed",
        "test_size",
        "val_size",
        "allow_subject_overlap",
        "split_file",
    }
    return {
        key: deepcopy(value)
        for key, value in canonical.items()
        if key not in split_only_keys
    }


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


def normalize_precision_name(precision: Any) -> str:
    precision = str(precision or "float64").strip().lower()
    if precision in {"float64", "double", "fp64"}:
        return "float64"
    if precision in {"float32", "float", "single", "fp32"}:
        return "float32"
    raise ValueError(
        "training.precision must be one of: float64, double, fp64, "
        "float32, float, single, fp32."
    )


def resolve_precision(precision: Any) -> torch.dtype:
    precision_name = normalize_precision_name(precision)
    if precision_name == "float64":
        return torch.float64
    return torch.float32


def first_floating_parameter_dtype(model: nn.Module) -> torch.dtype | None:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return None


def parse_task_types(task_types: Any) -> tuple[str, ...]:
    if task_types is None:
        return ("unilateral_fist", "both")
    if isinstance(task_types, str):
        return tuple(part.strip() for part in task_types.split(",") if part.strip())
    return tuple(str(part).strip() for part in task_types if str(part).strip())


def parse_subjects(subjects: Any) -> list[int] | None:
    if subjects is None:
        return None
    if isinstance(subjects, str):
        cleaned = subjects.strip()
        if cleaned.lower() in {"", "none", "null", "all"}:
            return None
        values: list[int] = []
        for part in cleaned.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if start > end:
                    raise argparse.ArgumentTypeError(f"invalid range: {part}")
                values.extend(range(start, end + 1))
            else:
                values.append(int(part))
        return sorted(set(values))
    if isinstance(subjects, (int, np.integer)):
        return [int(subjects)]

    values = []
    for subject in subjects:
        parsed = parse_subjects(subject)
        if parsed is not None:
            values.extend(parsed)
    return sorted(set(values)) or None


def normalize_dataset_name(name: Any) -> str:
    normalized = str(name or "physionet_mi").strip().lower().replace("-", "_")
    aliases = {
        "physionet": "physionet_mi",
        "physionetmi": "physionet_mi",
        "physionet_mi": "physionet_mi",
        "eegbci": "physionet_mi",
        "eeg_bci": "physionet_mi",
        "bci_iv_2a": "bnci2014_001",
        "bci_competition_iv_2a": "bnci2014_001",
        "bciciv_2a": "bnci2014_001",
        "bcic_iv_2a": "bnci2014_001",
        "bnci2014_001": "bnci2014_001",
        "bnci2014001": "bnci2014_001",
    }
    if normalized not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unknown data.dataset {name!r}. Valid datasets: {valid}.")
    return aliases[normalized]


def resolve_covariance_signal_scale(value: Any, dataset_name: str) -> float:
    if value is None:
        value = "auto"
    if isinstance(value, str) and value.strip().lower() in {"auto", ""}:
        # PhysioNet MNE EDF epochs are in Volts; MOABB array output is already
        # scaled to microvolts through the dataset unit_factor.
        return 1.0e6 if dataset_name == "physionet_mi" else 1.0
    return float(value)


def normalize_data_preprocessing_config(data_cfg: dict[str, Any]) -> None:
    canonical = normalize_data_time_config(data_cfg)
    data_cfg.clear()
    data_cfg.update(canonical)
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "physionet_mi"))
    data_cfg["dataset"] = dataset_name
    if "filter_bank" in data_cfg:
        data_cfg["filter_bank"] = normalize_filter_bank(data_cfg["filter_bank"])
    data_cfg["covariance_signal_scale"] = resolve_covariance_signal_scale(
        data_cfg.get("covariance_signal_scale", "auto"),
        dataset_name=dataset_name,
    )


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


def normalize_data_split_config(
        data_cfg: dict[str, Any],
        training_cfg: dict[str, Any],
) -> tuple[int, float, bool]:
    _ = training_cfg
    seed = int(data_cfg.get("seed", 42))
    test_size = float(data_cfg.get("test_size", 0.2))
    allow_subject_overlap = parse_bool(
        data_cfg.get("allow_subject_overlap", True),
        default=True,
    )
    data_cfg["seed"] = seed
    data_cfg["test_size"] = test_size
    data_cfg["allow_subject_overlap"] = allow_subject_overlap
    return seed, test_size, allow_subject_overlap


def make_loaders(
        x: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        batch_size: int,
        num_workers: int,
        dtype: torch.dtype,
        pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        MotorImageryDataset(x[train_idx], y[train_idx], dtype=dtype),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        MotorImageryDataset(x[test_idx], y[test_idx], dtype=dtype),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


def make_train_test_split_indices(
        y: np.ndarray,
        test_size: float,
        seed: int,
        subjects: np.ndarray | None = None,
        allow_subject_overlap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    indices = np.arange(len(y))
    if allow_subject_overlap:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )
        return train_idx.astype(np.int64), test_idx.astype(np.int64)

    if subjects is None:
        raise ValueError(
            "subjects must be provided when allow_subject_overlap is False."
        )
    subjects = np.asarray(subjects, dtype=np.str_)
    if len(subjects) != len(y):
        raise ValueError(
            f"subjects length ({len(subjects)}) must match labels length ({len(y)})."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(indices, y, groups=subjects))
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def make_cross_validation_splits(
        y: np.ndarray,
        subjects: np.ndarray,
        n_splits: int,
        seed: int,
        allow_subject_overlap: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified sample-level or subject-disjoint CV folds."""
    if n_splits < 2:
        raise ValueError(f"training.n_splits must be at least 2, got {n_splits}.")

    y = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.str_)
    indices = np.arange(len(y))
    if allow_subject_overlap:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(indices, y)
    else:
        if len(subjects) != len(y):
            raise ValueError(
                f"subjects length ({len(subjects)}) must match labels length "
                f"({len(y)})."
            )
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(indices, y, groups=subjects)

    return [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )
        for train_idx, test_idx in iterator
    ]


def aggregate_fold_metrics(
        fold_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return mean, maximum, and minimum test metrics across folds."""
    if not fold_metrics:
        raise ValueError("fold_metrics must contain at least one fold.")

    aggregates: dict[str, dict[str, float]] = {}
    for metric_name in CV_METRIC_NAMES:
        key = f"test_{metric_name}"
        values = np.asarray([row[key] for row in fold_metrics], dtype=float)
        aggregates[metric_name] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
        }
    return aggregates


def save_fold_results_csv(
        path: Path,
        fold_metrics: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "fold",
        "n_train",
        "n_test",
        "best_epoch",
        "test_accuracy",
        "test_macro_f1",
        "test_cohen_kappa",
    ]
    rows = [
        {field: metrics[field] for field in fieldnames}
        for metrics in fold_metrics
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_split_metadata(
        path: Path,
        y: np.ndarray,
        subject_labels: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        seed: int,
        test_size: float,
        allow_subject_overlap: bool,
) -> None:
    split_strategy = "epoch" if allow_subject_overlap else "subject"
    payload = {
        "split_strategy": split_strategy,
        "allow_subject_overlap": bool(allow_subject_overlap),
        "seed": int(seed),
        "test_size": float(test_size),
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_idx": train_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
        "train_subjects": sorted(set(subject_labels[train_idx].tolist())),
        "test_subjects": sorted(set(subject_labels[test_idx].tolist())),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_model(
        model_cfg: dict[str, Any],
        spd_in_dim: int,
        num_classes: int,
        time_sequence_length,
        frequency_sequence_length,
        brain_region_sequence_length=1,
) -> SPDTransformerClassifier:
    if "share_metric_across_heads" in model_cfg:
        raise ValueError(
            "model.share_metric_across_heads has been removed because heads "
            "within one axis always share their metric. Use "
            "model.share_metric_across_layers to control sharing between "
            "encoder layers."
        )

    depth = int(model_cfg.get("depth", 1))
    attention_dim = parse_attention_dims(
        model_cfg.get("attention_dim", spd_in_dim),
        depth=depth,
    )

    def optional_int(key: str) -> int | None:
        value = model_cfg.get(key)
        return None if value is None else int(value)

    tangent_use_position_embedding = model_cfg.get(
        "tangent_use_position_embedding"
    )
    if tangent_use_position_embedding is not None:
        tangent_use_position_embedding = parse_bool(
            tangent_use_position_embedding,
            default=True,
        )

    return SPDTransformerClassifier(
        num_heads=int(model_cfg.get("head_nums", 1)),
        spd_in_dim=spd_in_dim,
        attention_dim=attention_dim,
        stage_transition=bool(model_cfg.get("stage_transition", True)),
        time_sequence_length=time_sequence_length,
        frequency_sequence_length=frequency_sequence_length,
        brain_region_sequence_length=brain_region_sequence_length,
        tau=model_cfg.get("tau", 1.0),
        num_classes=num_classes,
        ffn_hidden_spd_dim=model_cfg.get("ffn_hidden_spd_dim", None),
        ffn_tangent_mixer_rank=int(
            model_cfg.get("ffn_tangent_mixer_rank", 0)
        ),
        metric=str(model_cfg.get("metric", "log-euclidean")),
        depth=depth,
        classifier_type=str(model_cfg.get("classifier_type", "pooling")),
        pooling=str(model_cfg.get("pooling", "weighted")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
        debug_attention_dropout=bool(model_cfg.get("debug_attention_dropout", False)),
        debug_attention_shape=bool(model_cfg.get("debug_attention_shape", False)),
        debug_tensor_stats=bool(model_cfg.get("debug_tensor_stats", False)),
        learnable_metric_mode=str(model_cfg.get("learnable_metric_mode", "low-rank")),
        learnable_metric_score=str(model_cfg.get("learnable_metric_score", "qgk")),
        learnable_metric_rank=model_cfg.get("learnable_metric_rank"),
        eps=float(model_cfg.get("eps", 1e-6)),
        use_position_bias=bool(model_cfg.get("use_position_bias", True)),
        layer_norm_affine=bool(model_cfg.get("layer_norm_affine", True)),
        stage_projection_init=str(model_cfg.get("stage_projection_init", "identity")),
        add_norm_type=str(model_cfg.get("add_norm_type", "trace")),
        encoder_type=str(model_cfg.get("encoder_type", "spd")),
        tangent_d_model=optional_int("tangent_d_model"),
        tangent_nhead=optional_int("tangent_nhead"),
        tangent_num_layers=optional_int("tangent_num_layers"),
        tangent_dim_feedforward=optional_int("tangent_dim_feedforward"),
        tangent_activation=str(model_cfg.get("tangent_activation", "gelu")),
        tangent_norm_first=parse_bool(
            model_cfg.get("tangent_norm_first", False),
            default=False,
        ),
        tangent_use_position_embedding=tangent_use_position_embedding,
        position_bias_axes=model_cfg.get("position_bias_axes"),
        position_bias_max=float(model_cfg.get("position_bias_max", 0.5)),
        attention_score_target_rms=float(
            model_cfg.get("attention_score_target_rms", 1.0)
        ),
        attention_score_clip=float(model_cfg.get("attention_score_clip", 5.0)),
        share_metric_across_layers=model_cfg.get(
            "share_metric_across_layers",
            False,
        ),
        independent_metric_per_axis=parse_bool(
            model_cfg.get("independent_metric_per_axis", True),
            default=True,
        ),
        head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        pooling_weight_mode=str(model_cfg.get("pooling_weight_mode", "full")),
        pooling_dropout=float(model_cfg.get("pooling_dropout", 0.0)),
        pooling_uniform_mix=float(model_cfg.get("pooling_uniform_mix", 0.0)),
        pooling_mean_anchor=parse_bool(
            model_cfg.get("pooling_mean_anchor", False),
            default=False,
        ),
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

    if len(dims) != depth:
        raise ValueError(
            "model.attention_dim must provide exactly one value per layer: "
            f"depth={depth}, got {dims}."
        )
    if any(dim < 1 for dim in dims):
        raise ValueError(
            f"Every model.attention_dim value must be positive, got {dims}."
        )

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
                "norm" in name.lower()
                or "metric_low_rank" in name.lower()
                or "metric_matrix" in name.lower()
                or "pooling_gate_logit" in name.lower()
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
        condition_regularization_weight: float = 0.0,
        prototype_intra_weight: float = 0.0,
        prototype_inter_weight: float = 0.0,
        prototype_margin: float = 1.0,
) -> dict[str, float]:
    predictions = predict_loader(
        model,
        loader,
        criterion,
        device,
        condition_regularization_weight=condition_regularization_weight,
        prototype_intra_weight=prototype_intra_weight,
        prototype_inter_weight=prototype_inter_weight,
        prototype_margin=prototype_margin,
    )
    return {
        "loss": predictions["loss"],
        "cross_entropy": predictions["cross_entropy"],
        "condition_loss": predictions["condition_loss"],
        "prototype_intra_loss": predictions["prototype_intra_loss"],
        "prototype_inter_loss": predictions["prototype_inter_loss"],
        "accuracy": predictions["accuracy"],
        "macro_f1": predictions["macro_f1"],
        "cohen_kappa": predictions["cohen_kappa"],
    }


def predict_loader(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        condition_regularization_weight: float = 0.0,
        prototype_intra_weight: float = 0.0,
        prototype_inter_weight: float = 0.0,
        prototype_margin: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    component_totals = {
        "cross_entropy": 0.0,
        "condition_loss": 0.0,
        "prototype_intra_loss": 0.0,
        "prototype_inter_loss": 0.0,
    }
    y_true = []
    y_pred = []

    with torch.no_grad():
        non_blocking = device.type == "cuda"
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)

            logits, _aux, losses = compute_training_objective(
                model,
                x_batch,
                y_batch,
                criterion,
                condition_regularization_weight=condition_regularization_weight,
                condition_regularization_fn=condition_regularization,
                prototype_intra_weight=prototype_intra_weight,
                prototype_inter_weight=prototype_inter_weight,
                prototype_margin=prototype_margin,
            )
            loss = losses["loss"]

            total_loss += loss.item() * y_batch.size(0)
            for name in component_totals:
                component_totals[name] += losses[name].item() * y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "loss": total_loss / len(y_true),
        **{
            name: value / len(y_true)
            for name, value in component_totals.items()
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
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
        optimizer_stiefel: geoopt.optim.RiemannianAdam | None,
        device: torch.device,
        gradient_clip_norm: float | None = None,
        debug_anomaly: bool = False,
        condition_regularization_weight: float = 0.0,
        prototype_intra_weight: float = 0.0,
        prototype_inter_weight: float = 0.0,
        prototype_margin: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    component_totals = {
        "cross_entropy": 0.0,
        "condition_loss": 0.0,
        "prototype_intra_loss": 0.0,
        "prototype_inter_loss": 0.0,
    }
    y_true = []
    y_pred = []
    non_blocking = device.type == "cuda"

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)
        logits, _aux, losses = compute_training_objective(
            model,
            x_batch,
            y_batch,
            criterion,
            condition_regularization_weight=condition_regularization_weight,
            condition_regularization_fn=condition_regularization,
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        )
        loss = losses["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss detected: "
                f"cross_entropy={losses['cross_entropy'].item():.6e} "
                f"cond_loss={losses['condition_loss'].item():.6e} "
                f"prototype_intra={losses['prototype_intra_loss'].item():.6e} "
                f"prototype_inter={losses['prototype_inter_loss'].item():.6e} "
                f"loss={loss.item():.6e}. "
                "Check input SPD matrices, learning rate, and model numerical stability."
            )

        optimizers = [optimizer_euclid]
        if optimizer_stiefel is not None:
            optimizers.append(optimizer_stiefel)
        for optimizer in optimizers:
            optimizer.zero_grad()

        if debug_anomaly:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()
        assert_model_finite(model, "backward")

        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            params_to_clip = [
                p
                for optimizer in optimizers
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]

            torch.nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=gradient_clip_norm,
            )

        for optimizer in optimizers:
            optimizer.step()
        assert_model_finite(model, "optimizer step")

        total_loss += loss.item() * y_batch.size(0)
        for name in component_totals:
            component_totals[name] += losses[name].item() * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(y_true),
        **{
            name: value / len(y_true)
            for name, value in component_totals.items()
        },
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


def optimizer_lr_values(
        optimizer: torch.optim.Optimizer | None,
) -> list[float]:
    if optimizer is None:
        return []
    return [float(group["lr"]) for group in optimizer.param_groups]


def format_lr_values(values: list[float]) -> str:
    if not values:
        return "none"
    return ",".join(f"{value:.3e}" for value in values)


def resolve_scheduler_metric(metric_name: str, test_metrics: dict[str, float]) -> float:
    metric_map = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }
    if metric_name not in metric_map:
        raise ValueError(
            "lr_scheduler_metric must be one of "
            "'test_loss', 'test_accuracy', or 'test_macro_f1', "
            f"got {metric_name!r}."
        )
    return float(metric_map[metric_name])


def build_lr_schedulers(
        training_cfg: dict[str, Any],
        optimizer_euclid: torch.optim.Optimizer,
        optimizer_stiefel: torch.optim.Optimizer | None,
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

    metric_name = str(training_cfg.get("lr_scheduler_metric", "test_macro_f1"))
    mode = training_cfg.get("lr_scheduler_mode")
    if mode is None:
        mode = "min" if metric_name == "test_loss" else "max"
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
    stiefel_scheduler = None
    if optimizer_stiefel is not None:
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


def train_fold(
        run_index: int,
        fold_index: int,
        n_splits: int,
        experiment_cfg: dict[str, Any],
        x: np.ndarray,
        y: np.ndarray,
        subject_labels: np.ndarray,
        class_names: list[str],
        run_dir: Path,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        device: torch.device,
) -> dict[str, Any]:
    training_cfg = experiment_cfg["training"]
    model_cfg = experiment_cfg["model"]
    data_cfg = experiment_cfg["data"]
    precision_name = normalize_precision_name(
        training_cfg.get("precision", "float64")
    )
    training_cfg["precision"] = precision_name
    seed, _test_size, allow_subject_overlap = normalize_data_split_config(
        data_cfg,
        training_cfg,
    )
    dtype = resolve_precision(precision_name)
    set_seed(seed + fold_index - 1)
    if device.type == "cuda":
        allow_tf32 = parse_bool(
            training_cfg.get("allow_tf32", False),
            default=False,
        )
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    global_max = np.max(x)
    global_min = np.min(np.abs(x))
    print(f"max: {global_max}")
    print(f"min: {global_min}")

    train_subjects = set(subject_labels[train_idx].tolist())
    test_subjects = set(subject_labels[test_idx].tolist())
    if not allow_subject_overlap:
        train_test_overlap = train_subjects & test_subjects
        if train_test_overlap:
            raise RuntimeError(
                "Subject-level split failed: subject overlap detected between "
                f"train and test: {sorted(train_test_overlap)}."
            )
    save_split_metadata(
        run_dir / "split.json",
        y=y,
        subject_labels=subject_labels,
        train_idx=train_idx,
        test_idx=test_idx,
        seed=seed,
        test_size=1.0 / n_splits,
        allow_subject_overlap=allow_subject_overlap,
    )

    train_loader, test_loader = make_loaders(
        x=x,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        batch_size=int(training_cfg.get("batch_size", 16)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        dtype=dtype,
        pin_memory=parse_bool(
            training_cfg.get("pin_memory", device.type == "cuda"),
            default=device.type == "cuda",
        ),
    )

    time_sequence_length = x.shape[1]
    frequency_sequence_length = x.shape[2] if x.ndim >= 5 else 1
    brain_region_sequence_length = x.shape[3] if x.ndim >= 6 else 1
    model = build_model(
        model_cfg=model_cfg,
        spd_in_dim=x.shape[-1],
        time_sequence_length=time_sequence_length,
        frequency_sequence_length=frequency_sequence_length,
        brain_region_sequence_length=brain_region_sequence_length,
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

    optimizer_stiefel = None
    if stiefel_params:
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

    best_test_macro_f1 = -1.0
    best_epoch = 0
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_model.pt"
    epochs = int(training_cfg.get("epochs", 50))
    condition_regularization_weight = float(
        training_cfg.get("condition_regularization_weight", 0.0)
    )
    (
        prototype_intra_weight,
        prototype_inter_weight,
        prototype_margin,
    ) = prototype_loss_settings(training_cfg)
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

    print(
        f"\n[Run {run_index} | Fold {fold_index}/{n_splits}] "
        f"{run_dir.name}"
    )
    print(f"  data={data_cfg}")
    print(f"  model={model_cfg}")
    print(f"  training={training_cfg}")
    print(
        "  dtype="
        f"precision={precision_name} "
        f"torch_dtype={str(dtype).replace('torch.', '')} "
        f"cached_x_dtype={x.dtype} "
        f"loader_x_dtype={train_loader.dataset.x.dtype} "
        f"model_param_dtype={first_floating_parameter_dtype(model)}"
    )
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
        f"subjects train/test={len(train_subjects)}/{len(test_subjects)} "
        f"samples train/test={len(train_idx)}/{len(test_idx)}"
    )
    print(
        "  objective="
        f"cross_entropy + {prototype_intra_weight:g}*prototype_intra "
        f"+ {prototype_inter_weight:g}*prototype_inter_margin "
        f"(margin={prototype_margin:g})"
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
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            condition_regularization_weight=condition_regularization_weight,
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        )

        if lr_scheduler_name is not None:
            old_euclid_lrs = optimizer_lr_values(optimizer_euclid)
            old_stiefel_lrs = optimizer_lr_values(optimizer_stiefel)
            scheduler_value = resolve_scheduler_metric(
                lr_scheduler_metric,
                test_metrics,
            )
            lr_scheduler_euclid.step(scheduler_value)
            if lr_scheduler_stiefel is not None:
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
            "train_cross_entropy": train_metrics["cross_entropy"],
            "train_prototype_intra_loss": train_metrics["prototype_intra_loss"],
            "train_prototype_inter_loss": train_metrics["prototype_inter_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "test_loss": test_metrics["loss"],
            "test_cross_entropy": test_metrics["cross_entropy"],
            "test_prototype_intra_loss": test_metrics["prototype_intra_loss"],
            "test_prototype_inter_loss": test_metrics["prototype_inter_loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "euclid_lr": optimizer_lr_values(optimizer_euclid)[0],
            "stiefel_lr": (
                optimizer_lr_values(optimizer_stiefel)[0]
                if optimizer_stiefel is not None
                else None
            ),
        }
        append_history(history_path, row)

        if test_metrics["macro_f1"] > best_test_macro_f1 + early_stopping_min_delta:
            best_test_macro_f1 = test_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": experiment_cfg,
                    "best_epoch": best_epoch,
                    "best_test_macro_f1": best_test_macro_f1,
                },
                checkpoint_path,
            )

        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['cross_entropy']:.4f} "
            f"intra={train_metrics['prototype_intra_loss']:.4f} "
            f"inter={train_metrics['prototype_inter_loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"mf1={train_metrics['macro_f1']:.4f} | "
            f"test loss={test_metrics['loss']:.4f} "
            f"ce={test_metrics['cross_entropy']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"mf1={test_metrics['macro_f1']:.4f}"
        )

        if (
                early_stopping_patience is not None
                and early_stopping_patience > 0
                and epoch - best_epoch >= early_stopping_patience
        ):
            print(
                f"  early stopping at epoch {epoch:03d}/{epochs} | "
                f"best_epoch={best_epoch} "
                f"best_test_mf1={best_test_macro_f1:.4f}"
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
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
        ),
        "test": predict_loader(
            model,
            test_loader,
            criterion,
            device,
            condition_regularization_weight=condition_regularization_weight,
            prototype_intra_weight=prototype_intra_weight,
            prototype_inter_weight=prototype_inter_weight,
            prototype_margin=prototype_margin,
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
        "fold": fold_index,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_test_macro_f1": best_test_macro_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_cohen_kappa": test_metrics["cohen_kappa"],
        "class_names": class_names,
        "precision": precision_name,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "cached_x_dtype": str(x.dtype),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "allow_subject_overlap": allow_subject_overlap,
        "n_train_subjects": int(len(train_subjects)),
        "n_test_subjects": int(len(test_subjects)),
    }

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(
        f"[Run {run_index} | Fold {fold_index}/{n_splits}] done | "
        f"best_epoch={best_epoch} "
        f"best_test_mf1={best_test_macro_f1:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_mf1={test_metrics['macro_f1']:.4f} "
        f"test_kappa={test_metrics['cohen_kappa']:.4f}"
    )
    return metrics


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
    """Train and evaluate one configuration with stratified K-fold CV."""
    training_cfg = experiment_cfg["training"]
    data_cfg = experiment_cfg["data"]
    seed, _test_size, allow_subject_overlap = normalize_data_split_config(
        data_cfg,
        training_cfg,
    )
    n_splits = int(training_cfg.get("n_splits", 5))
    training_cfg["n_splits"] = n_splits

    folds = make_cross_validation_splits(
        y=y,
        subjects=subject_labels,
        n_splits=n_splits,
        seed=seed,
        allow_subject_overlap=allow_subject_overlap,
    )
    run_dir = make_run_dir(base_output_dir, run_index, experiment_cfg)
    save_yaml(run_dir / "config.yaml", experiment_cfg)

    fold_metrics: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=False, exist_ok=False)
        metrics = train_fold(
            run_index=run_index,
            fold_index=fold_index,
            n_splits=n_splits,
            experiment_cfg=experiment_cfg,
            x=x,
            y=y,
            subject_labels=subject_labels,
            class_names=class_names,
            run_dir=fold_dir,
            train_idx=train_idx,
            test_idx=test_idx,
            device=device,
        )
        fold_metrics.append(metrics)

    aggregates = aggregate_fold_metrics(fold_metrics)
    summary = {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "evaluation": {
            "strategy": (
                "stratified_kfold"
                if allow_subject_overlap
                else "stratified_group_kfold"
            ),
            "n_splits": n_splits,
            "seed": seed,
            "test_size_per_fold": 1.0 / n_splits,
            "allow_subject_overlap": allow_subject_overlap,
        },
        "folds": fold_metrics,
        "aggregate": aggregates,
    }
    save_fold_results_csv(run_dir / "fold_results.csv", fold_metrics)
    with (run_dir / "five_fold_summary.json").open(
            "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n[Run {run_index}] {n_splits}-fold aggregate test results")
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
    print(f"Saved fold results: {run_dir / 'fold_results.csv'}")
    print(f"Saved aggregate summary: {run_dir / 'five_fold_summary.json'}")
    return summary


def preprocess_dataset(
        data_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    normalize_data_preprocessing_config(data_cfg)
    dataset_name = normalize_dataset_name(data_cfg.get("dataset", "physionet_mi"))
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    epoch_tmin, epoch_tmax = data_cfg["epoch_slice"]
    segment_duration, stride_duration = data_cfg["segment_slice"]
    subjects = parse_subjects(data_cfg.get("subjects"))
    covariance_signal_scale = resolve_covariance_signal_scale(
        data_cfg.get("covariance_signal_scale", "auto"),
        dataset_name=dataset_name,
    )

    if dataset_name == "physionet_mi":
        task_types = parse_task_types(data_cfg.get("task_types"))
        x, y, class_names, subject_labels = preprocess_spd(
            filter_bank=filter_bank,
            root_dir=str(
                data_cfg.get(
                    "root_dir",
                    "data/MNE-eegbci-data/files/eegmmidb/1.0.0",
                )
            ),
            subjects=subjects,
            channels=data_cfg.get("channels"),
            estimator=str(data_cfg.get("estimator", "lwf")),
            sfreq=float(data_cfg.get("sfreq", 160)),
            eps=float(data_cfg.get("eps", 1e-8)),
            segment_duration=float(segment_duration),
            stride_duration=stride_duration,
            imaged=parse_bool(data_cfg.get("imaged", True), default=True),
            executed=parse_bool(data_cfg.get("executed", False), default=False),
            task_types=task_types,
            reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
            baseline_correction=data_cfg.get("baseline_correction"),
            baseline_window=data_cfg.get("baseline_window"),
            epoch_tmin=float(epoch_tmin),
            epoch_tmax=float(epoch_tmax),
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
            autoreject_cache_dir=data_cfg.get("autoreject_cache_dir"),
            autoreject_force_rebuild=parse_bool(
                data_cfg.get("autoreject_force_rebuild", False),
                default=False,
            ),
            return_subjects=True,
            covariance_signal_scale=covariance_signal_scale,
            replace_covariance_diagonal_with_raw_energy=parse_bool(
                data_cfg.get("replace_covariance_diagonal_with_raw_energy", False),
                default=False,
            ),
            brain_region_mode=data_cfg.get("brain_region_mode"),
            output_dtype=data_cfg.get("covariance_output_dtype", "float32"),
        )
    elif dataset_name == "bnci2014_001":
        x, y, class_names, subject_labels = preprocess_bci_iv_2a_spd(
            filter_bank=filter_bank,
            root_dir=str(data_cfg.get("root_dir", "data")),
            subjects=subjects,
            channels=data_cfg.get("channels"),
            events=data_cfg.get("events", data_cfg.get("bci_iv_2a_events")),
            sessions=data_cfg.get("sessions", data_cfg.get("bci_iv_2a_sessions")),
            estimator=str(data_cfg.get("estimator", "lwf")),
            sfreq=float(data_cfg.get("sfreq", 250)),
            eps=float(data_cfg.get("eps", 1e-8)),
            segment_duration=float(segment_duration),
            stride_duration=stride_duration,
            reject_threshold_uv=data_cfg.get("reject_threshold_uv"),
            baseline_correction=data_cfg.get("baseline_correction"),
            baseline_window=data_cfg.get("baseline_window"),
            epoch_tmin=float(epoch_tmin),
            epoch_tmax=float(epoch_tmax),
            use_ica=parse_bool(data_cfg.get("use_ica", False), default=False),
            use_autoreject=parse_bool(
                data_cfg.get("use_autoreject", False),
                default=False,
            ),
            autoreject_random_state=int(data_cfg.get("autoreject_random_state", 42)),
            autoreject_n_jobs=int(data_cfg.get("autoreject_n_jobs", 1)),
            autoreject_cv=int(data_cfg.get("autoreject_cv", 10)),
            return_subjects=True,
            covariance_signal_scale=covariance_signal_scale,
            replace_covariance_diagonal_with_raw_energy=parse_bool(
                data_cfg.get("replace_covariance_diagonal_with_raw_energy", False),
                default=False,
            ),
            brain_region_mode=data_cfg.get("brain_region_mode"),
            output_dtype=data_cfg.get("covariance_output_dtype", "float32"),
            moabb_accept_terms=parse_bool(
                data_cfg.get("moabb_accept_terms", True),
                default=True,
            ),
            moabb_force_update=parse_bool(
                data_cfg.get("moabb_force_update", False),
                default=False,
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not np.isfinite(x).all():
        bad_count = int((~np.isfinite(x)).sum())
        raise ValueError(
            f"Preprocessed SPD dataset contains {bad_count} NaN or Inf values."
        )
    return (
        x.astype(
            normalize_float_dtype(
                data_cfg.get("covariance_output_dtype", "float32")
            ),
            copy=False,
        ),
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

            x = np.asarray(payload["x"])
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
            x=x,
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
    preprocessing_cfg = dataset_preprocessing_config(data_cfg)
    data_key = dataset_cache_key(data_cfg)
    cache_path = dataset_cache_path(cache_dir, data_key)
    cached_dataset = load_cached_dataset(cache_path, preprocessing_cfg)
    if cached_dataset is not None:
        print(f"\nLoaded preprocessed data from cache {data_key}: {cache_path}")
        x_cached, y_cached, subjects_cached, names_cached = cached_dataset
        print(
            f"  X.shape={x_cached.shape}, y.shape={y_cached.shape}, "
            f"subjects={len(set(subjects_cached.tolist()))}, classes={names_cached}"
        )
        return cached_dataset

    print(f"\nPreprocessing data config {data_key}: {preprocessing_cfg}")
    dataset = preprocess_dataset(preprocessing_cfg)
    x_cached, y_cached, subjects_cached, names_cached = dataset
    print(
        f"  X.shape={x_cached.shape}, y.shape={y_cached.shape}, "
        f"subjects={len(set(subjects_cached.tolist()))}, classes={names_cached}"
    )
    save_cached_dataset(
        cache_path,
        preprocessing_cfg,
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
    if args.precision is not None:
        precision_name = normalize_precision_name(args.precision)
        for experiment_cfg in experiments:
            experiment_cfg.setdefault("training", {})["precision"] = precision_name
        print(f"Precision override: {precision_name}")
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

    # Cache each unique preprocessing configuration in memory. Split-only data
    # grid values (for example allow_subject_overlap) share the same tensors.
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    all_metrics = []

    print(f"Generated {len(experiments)} experiment(s)")
    print(f"Saving runs under: {base_output_dir}")
    print(f"Dataset cache: {dataset_cache_dir}")
    print(f"Device: {device}")

    for run_index, experiment_cfg in enumerate(experiments, start=1):
        normalize_data_split_config(
            experiment_cfg["data"],
            experiment_cfg["training"],
        )
        normalize_data_preprocessing_config(experiment_cfg["data"])
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
