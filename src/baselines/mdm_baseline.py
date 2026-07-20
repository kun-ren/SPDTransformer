from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MDM_DATASET_CACHE_VERSION = 2
DEFAULT_MDM_DATASET_CACHE_DIR = (
    PROJECT_ROOT / "experiments" / "cache" / "mdm_preprocessed_datasets"
)

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    compute_metrics,
    config_hash,
    expand_data_grid,
    expand_grid,
    load_spd_like_train,
    load_yaml,
    matrix_exp,
    matrix_log,
    parse_bool,
    normalize_data_time_config,
    resolve_split_file,
    save_json,
)

# MNE's native runtime must initialize before PyTorch on Windows.
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.SPDMDMClassifier import LogEuclideanPrototypeClassifier


TOKEN_WEIGHT_KEYS = {"token_weight_logits", "token_weights"}
MDM_METRIC_KEYS = {
    "metric",
    "mdm_metric",
    "mean_metric",
    "metric_mean",
    "mdm_mean_metric",
    "distance_metric",
    "metric_distance",
    "mdm_distance_metric",
    "map_metric",
    "metric_map",
    "fgmdm_map_metric",
}


class IncompatibleExperiment(ValueError):
    """Raised when an expanded data/model combination should be skipped."""


def _labels_hash(y: np.ndarray) -> str:
    labels = np.asarray(y, dtype=np.int64)
    return hashlib.sha1(labels.tobytes()).hexdigest()


def _subjects_hash(subjects: np.ndarray | None) -> str | None:
    if subjects is None:
        return None
    subject_labels = np.asarray(subjects, dtype=np.str_)
    return hashlib.sha1("\n".join(subject_labels.tolist()).encode("utf-8")).hexdigest()


def resolve_project_path(path_value: Any, default: Path) -> Path:
    if path_value in {None, ""}:
        return default
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def dataset_cache_key(data_cfg: dict[str, Any]) -> str:
    return config_hash({"data": mdm_preprocessing_data_config(data_cfg)})


def mdm_preprocessing_data_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    canonical = normalize_data_time_config(data_cfg)
    split_only_keys = {
        "seed",
        "test_size",
        "val_size",
        "allow_subject_overlap",
    }
    return {
        key: deepcopy(value)
        for key, value in canonical.items()
        if key not in split_only_keys
    }


def dataset_cache_path(cache_dir: Path, data_key: str) -> Path:
    return cache_dir / f"mdm_spd_dataset_{data_key}.npz"


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "false", "off"}
    return False


def is_band_pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(number, (int, float)) for number in value)
    )


def is_filter_bank_scheme(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        is_band_pair(item) for item in value
    )


def mdm_model_grid_values(key: str, value: Any) -> list[Any]:
    if key in TOKEN_WEIGHT_KEYS:
        return [None if _is_none_like(value) else value]
    if key in MDM_METRIC_KEYS and isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return [value]


def expand_mdm_model_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    if not section:
        return [{}]
    keys = list(section)
    value_lists = [mdm_model_grid_values(key, section[key]) for key in keys]
    expanded = []
    seen: set[str] = set()
    for values in itertools.product(*value_lists):
        model_cfg = dict(zip(keys, values))
        if resolve_classifier_type(model_cfg) == "differentiable":
            model_cfg = {
                key: value
                for key, value in model_cfg.items()
                if key not in MDM_METRIC_KEYS
            }
        model_key = json.dumps(model_cfg, sort_keys=True, default=str)
        if model_key not in seen:
            expanded.append(model_cfg)
            seen.add(model_key)
    return expanded


def expand_mdm_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_data_grid(config.get("data", {}))
    model_grid = expand_mdm_model_grid(config.get("model", {}))
    training_grid = expand_grid(config.get("training", {}))
    output_cfg = deepcopy(config.get("output", {}))
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
                "output": deepcopy(output_cfg),
            }
        )
    return experiments


def override_mdm_classifier(
    experiments: list[dict[str, Any]],
    classifier_type: Any,
) -> list[dict[str, Any]]:
    if _is_none_like(classifier_type):
        return experiments

    normalized = normalize_classifier_type(classifier_type)
    overridden = []
    seen: set[str] = set()
    for experiment in experiments:
        updated = deepcopy(experiment)
        updated["model"]["classifier_type"] = normalized
        if normalized == "differentiable":
            for key in MDM_METRIC_KEYS:
                updated["model"].pop(key, None)
        experiment_key = json.dumps(updated, sort_keys=True, default=str)
        if experiment_key not in seen:
            overridden.append(updated)
            seen.add(experiment_key)
    return overridden


def normalize_token_pooling(pooling: Any) -> str:
    normalized = str(pooling or "mean").strip().lower().replace("-", "_")
    if normalized in {"original", "raw", "none", "identity", "trial", "single"}:
        return "original"
    if normalized in {"mean", "mdm_mean", "log_euclidean_mean"}:
        return "mean"
    if normalized in {"weighted", "weight", "mdm_weighted", "learned_weighted"}:
        return "weighted"
    raise ValueError(
        "MDM token pooling must be 'original', 'mean', or 'weighted', "
        f"got {pooling!r}."
    )


def normalize_classifier_type(value: Any) -> str:
    normalized = str(value or "pyriemann").strip().lower().replace("-", "_")
    if normalized in {"pyriemann", "classic", "classical", "mdm"}:
        return "pyriemann"
    if normalized in {"fgmdm", "fg_mdm", "fisher_geodesic_mdm"}:
        return "fgmdm"
    if normalized in {
        "differentiable",
        "learnable",
        "learned",
        "prototype",
        "log_euclidean_prototype",
    }:
        return "differentiable"
    raise ValueError(
        "MDM classifier_type must be 'pyriemann', 'fgmdm', or "
        "'differentiable', "
        f"got {value!r}."
    )


def resolve_classifier_type(model_cfg: dict[str, Any]) -> str:
    value = model_cfg.get(
        "classifier_type",
        model_cfg.get("classifier", model_cfg.get("mdm_classifier", "pyriemann")),
    )
    return normalize_classifier_type(value)


def obvious_original_mdm_token_count(data_cfg: dict[str, Any]) -> int | None:
    has_explicit_epoch = "epoch_slice" in data_cfg or {
        "epoch_tmin",
        "epoch_tmax",
    }.issubset(data_cfg)
    has_explicit_segment = "segment_slice" in data_cfg or "segment_duration" in data_cfg
    data_cfg = normalize_data_time_config(data_cfg)
    token_count = 1
    filter_bank = data_cfg.get("filter_bank")
    if is_filter_bank_scheme(filter_bank):
        token_count *= len(filter_bank)

    brain_region_mode = data_cfg.get("brain_region_mode")
    if not _is_none_like(brain_region_mode):
        # Exact region count is dataset-specific and resolved during preprocessing.
        # Any enabled region preset creates more than one SPD token per trial.
        token_count *= 2

    if has_explicit_epoch and has_explicit_segment:
        epoch_tmin, epoch_tmax = data_cfg["epoch_slice"]
        segment_duration, _ = data_cfg["segment_slice"]
        epoch_duration = float(epoch_tmax) - float(epoch_tmin)
        if float(segment_duration) < epoch_duration - 1e-9:
            token_count *= 2

    return token_count


def validate_data_model_compatibility(
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> None:
    pooling = normalize_token_pooling(
        model_cfg.get("pooling", model_cfg.get("token_pooling", "mean"))
    )
    if pooling != "original":
        return

    token_count = obvious_original_mdm_token_count(data_cfg)
    if token_count is not None and token_count > 1:
        raise IncompatibleExperiment(
            "pooling='original' requires one SPD matrix per trial, but this "
            "data config obviously creates multiple tokens "
            f"(filter_bank={data_cfg.get('filter_bank')!r}, "
            f"segment_slice={data_cfg.get('segment_slice')!r}, "
            f"epoch_slice={data_cfg.get('epoch_slice')!r}, "
            f"brain_region_mode={data_cfg.get('brain_region_mode')!r})."
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
            if metadata.get("cache_version") != MDM_DATASET_CACHE_VERSION:
                print(f"  MDM dataset cache version changed, rebuilding: {cache_path}")
                return None
            if metadata.get("data_config") != data_cfg:
                print(
                    "  MDM dataset cache key collision or config mismatch, "
                    f"rebuilding: {cache_path}"
                )
                return None

            x_spd = np.asarray(payload["x_spd"])
            y = np.asarray(payload["y"], dtype=np.int64)
            subject_labels = np.asarray(payload["subject_labels"], dtype=np.str_)
            class_names = [str(name) for name in payload["class_names"].tolist()]

        if not np.isfinite(x_spd).all():
            bad_count = int((~np.isfinite(x_spd)).sum())
            print(
                f"  Cached MDM SPD dataset contains {bad_count} NaN/Inf values, "
                "rebuilding."
            )
            return None
        if len(subject_labels) != len(y):
            print("  Cached subject label count does not match y length, rebuilding.")
            return None
        return x_spd, y, subject_labels, class_names
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"  Failed to read MDM dataset cache {cache_path}: {error}. Rebuilding.")
        return None


def save_cached_dataset(
    cache_path: Path,
    data_cfg: dict[str, Any],
    x_spd: np.ndarray,
    y: np.ndarray,
    subject_labels: np.ndarray,
    class_names: list[str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "cache_version": MDM_DATASET_CACHE_VERSION,
        "data_config": data_cfg,
    }
    tmp_path = cache_path.with_name(
        f"{cache_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.tmp"
    )
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            x_spd=x_spd,
            y=y.astype(np.int64, copy=False),
            subject_labels=np.asarray(subject_labels, dtype=np.str_),
            class_names=np.asarray(class_names, dtype=np.str_),
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, default=str),
                dtype=np.str_,
            ),
        )
    tmp_path.replace(cache_path)


def load_or_preprocess_spd(
    data_cfg: dict[str, Any],
    cache_dir: Path,
    memory_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    preprocessing_cfg = mdm_preprocessing_data_config(data_cfg)
    data_key = dataset_cache_key(data_cfg)
    if data_key in memory_cache:
        print(f"\nUsing in-memory MDM data cache {data_key}")
        return memory_cache[data_key]

    cache_path = dataset_cache_path(cache_dir, data_key)
    cached_dataset = load_cached_dataset(cache_path, preprocessing_cfg)
    if cached_dataset is not None:
        print(f"\nLoaded MDM preprocessed data from cache {data_key}: {cache_path}")
        x_cached, y_cached, subjects_cached, class_names_cached = cached_dataset
        print(
            f"  X.shape={x_cached.shape}, y.shape={y_cached.shape}, "
            f"subjects={len(set(subjects_cached.tolist()))}, "
            f"classes={class_names_cached}"
        )
        memory_cache[data_key] = cached_dataset
        return cached_dataset

    print(f"\nPreprocessing MDM data config {data_key}: {preprocessing_cfg}")
    dataset = load_spd_like_train(preprocessing_cfg)
    x_spd, y, subject_labels, class_names = dataset
    print(
        f"  X.shape={x_spd.shape}, y.shape={y.shape}, "
        f"subjects={len(set(subject_labels.tolist()))}, classes={class_names}"
    )
    save_cached_dataset(
        cache_path,
        preprocessing_cfg,
        x_spd,
        y,
        subject_labels,
        class_names,
    )
    print(f"  Saved MDM preprocessed data cache: {cache_path}")
    memory_cache[data_key] = dataset
    return dataset


def normalize_metric_name(value: Any) -> str:
    if _is_none_like(value):
        raise ValueError("MDM metric names cannot be null or empty.")
    return str(value).strip().lower().replace("-", "")


def metric_config_value(
    model_cfg: dict[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in model_cfg and not _is_none_like(model_cfg[key]):
            return model_cfg[key]
    return None


def resolve_mdm_metric(
    model_cfg: dict[str, Any],
    cli_metric: Any = None,
    cli_mean_metric: Any = None,
    cli_distance_metric: Any = None,
) -> str | dict[str, str]:
    configured_metric = metric_config_value(model_cfg, "metric", "mdm_metric")
    base_metric = cli_metric if not _is_none_like(cli_metric) else configured_metric
    if _is_none_like(base_metric):
        base_metric = "riemann"

    if isinstance(base_metric, dict):
        metric_mean = metric_config_value(base_metric, "mean")
        metric_distance = metric_config_value(base_metric, "distance", "dist")
        fallback_metric = None
    else:
        metric_mean = None
        metric_distance = None
        fallback_metric = base_metric

    configured_mean = metric_config_value(
        model_cfg,
        "mean_metric",
        "metric_mean",
        "mdm_mean_metric",
    )
    configured_distance = metric_config_value(
        model_cfg,
        "distance_metric",
        "metric_distance",
        "mdm_distance_metric",
    )
    mean_metric = (
        cli_mean_metric
        if not _is_none_like(cli_mean_metric)
        else configured_mean
        if not _is_none_like(configured_mean)
        else metric_mean
    )
    distance_metric = (
        cli_distance_metric
        if not _is_none_like(cli_distance_metric)
        else configured_distance
        if not _is_none_like(configured_distance)
        else metric_distance
    )

    if _is_none_like(mean_metric) and _is_none_like(distance_metric):
        return normalize_metric_name(fallback_metric or base_metric)

    if _is_none_like(mean_metric):
        mean_metric = fallback_metric
    if _is_none_like(distance_metric):
        distance_metric = fallback_metric
    if _is_none_like(mean_metric) or _is_none_like(distance_metric):
        raise ValueError(
            "Both MDM mean and distance metrics must be set when using a metric dict."
        )
    return {
        "mean": normalize_metric_name(mean_metric),
        "distance": normalize_metric_name(distance_metric),
    }


def resolve_fgmdm_metric(
    model_cfg: dict[str, Any],
    cli_metric: Any = None,
    cli_mean_metric: Any = None,
    cli_distance_metric: Any = None,
) -> str | dict[str, str]:
    metric = resolve_mdm_metric(
        model_cfg,
        cli_metric=cli_metric,
        cli_mean_metric=cli_mean_metric,
        cli_distance_metric=cli_distance_metric,
    )
    configured_map = metric_config_value(
        model_cfg,
        "map_metric",
        "metric_map",
        "fgmdm_map_metric",
    )
    configured_metric = metric_config_value(model_cfg, "metric", "mdm_metric")
    if _is_none_like(configured_map) and isinstance(configured_metric, dict):
        configured_map = metric_config_value(configured_metric, "map")

    if isinstance(metric, str) and _is_none_like(configured_map):
        return metric

    if isinstance(metric, str):
        mean_metric = metric
        distance_metric = metric
    else:
        mean_metric = metric["mean"]
        distance_metric = metric["distance"]
    map_metric = (
        mean_metric
        if _is_none_like(configured_map)
        else normalize_metric_name(configured_map)
    )
    # pyRiemann 0.11 forwards this dict to both FGDA and MDM, requiring the
    # union of their keys even though the FgMDM docstring says "dist".
    return {
        "mean": mean_metric,
        "distance": distance_metric,
        "map": map_metric,
    }


def validate_spd_array(x_spd: np.ndarray) -> None:
    if x_spd.ndim < 3:
        raise ValueError(
            "Expected SPD input shape (trial, ..., channels, channels), "
            f"got {x_spd.shape}."
        )
    if x_spd.shape[-1] != x_spd.shape[-2]:
        raise ValueError(f"Last two SPD dimensions must be square, got {x_spd.shape}.")


def token_shape_of(x_spd: np.ndarray) -> tuple[int, ...]:
    validate_spd_array(x_spd)
    return tuple(int(size) for size in x_spd.shape[1:-2])


def softmax_np(logits: np.ndarray) -> np.ndarray:
    flat_logits = logits.reshape(-1).astype(np.float64)
    flat_logits = flat_logits - np.max(flat_logits)
    flat_weights = np.exp(flat_logits)
    flat_weights = flat_weights / flat_weights.sum()
    return flat_weights.reshape(logits.shape)


def resolve_token_weights(
    model_cfg: dict[str, Any],
    token_shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    if not token_shape:
        raise ValueError("Weighted token pooling requires at least one token dimension.")
    if len(token_shape) > 3:
        raise ValueError(
            "Weighted MDM pooling supports segment, segment/frequency, or "
            f"segment/frequency/region tokens, got token shape {token_shape}."
        )

    raw_logits = model_cfg.get("token_weight_logits")
    raw_weights = model_cfg.get("token_weights")
    if not _is_none_like(raw_logits) and not _is_none_like(raw_weights):
        raise ValueError("Set only one of token_weight_logits or token_weights.")

    if _is_none_like(raw_logits) and _is_none_like(raw_weights):
        uniform = np.full(token_shape, 1.0 / np.prod(token_shape), dtype=np.float64)
        return uniform, "uniform"

    raw_value = raw_weights if not _is_none_like(raw_weights) else raw_logits
    values = np.asarray(raw_value, dtype=np.float64)
    if values.shape == ():
        values = np.full(token_shape, float(values), dtype=np.float64)
    elif values.shape == token_shape:
        values = values.astype(np.float64, copy=False)
    elif values.ndim == 1 and values.size == int(np.prod(token_shape)):
        values = values.reshape(token_shape)
    else:
        raise ValueError(
            "Token weights/logits must be scalar, flat with "
            f"{int(np.prod(token_shape))} values, or shaped as {token_shape}; "
            f"got shape {values.shape}."
        )

    if not np.isfinite(values).all():
        raise ValueError("Token weights/logits contain NaN or Inf.")

    if not _is_none_like(raw_weights):
        if np.any(values < 0):
            raise ValueError("token_weights must be non-negative.")
        total = values.sum()
        if total <= 0:
            raise ValueError("token_weights must sum to a positive value.")
        return values / total, "configured_weights"

    return softmax_np(values), "configured_logits"


def pool_spd_tokens(
    x_spd: np.ndarray,
    model_cfg: dict[str, Any],
    eps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pooling = normalize_token_pooling(
        model_cfg.get("pooling", model_cfg.get("token_pooling", "mean"))
    )
    token_shape = token_shape_of(x_spd)
    token_axes = tuple(range(1, x_spd.ndim - 2))

    if pooling == "original":
        if not token_shape:
            x_trial_spd = x_spd
        elif int(np.prod(token_shape)) == 1:
            x_trial_spd = x_spd.reshape(
                x_spd.shape[0],
                x_spd.shape[-2],
                x_spd.shape[-1],
            )
        else:
            raise IncompatibleExperiment(
                "pooling='original' requires exactly one SPD matrix per trial. "
                f"Current token shape is {token_shape}; use pooling='mean' or "
                "pooling='weighted' for segmented/filter-bank/brain-region inputs."
            )
        return x_trial_spd.astype(np.float64), {
            "mode": "original",
            "token_shape": list(token_shape),
            "has_parameters": False,
        }

    if not token_shape:
        return x_spd.astype(np.float64), {
            "mode": pooling,
            "token_shape": [],
            "has_parameters": False,
        }

    log_x = matrix_log(x_spd, eps=eps)
    if pooling == "mean":
        pooled_log = log_x.mean(axis=token_axes)
        return matrix_exp(pooled_log).astype(np.float64), {
            "mode": "mean",
            "token_shape": list(token_shape),
            "has_parameters": False,
        }

    weights, source = resolve_token_weights(model_cfg, token_shape)
    view_shape = (1, *token_shape, 1, 1)
    pooled_log = (log_x * weights.reshape(view_shape)).sum(axis=token_axes)
    return matrix_exp(pooled_log).astype(np.float64), {
        "mode": "weighted",
        "token_shape": list(token_shape),
        "weight_source": source,
        "has_parameters": True,
        "token_weights": weights.tolist(),
    }


def resolve_precision(value: Any) -> torch.dtype:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"float32", "float", "single", "fp32"}:
        return torch.float32
    if normalized in {"float64", "double", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported precision: {value!r}.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class IndexedLogSPDDataset(Dataset):
    def __init__(
        self,
        x_log: torch.Tensor,
        y: torch.Tensor,
        indices: np.ndarray,
    ) -> None:
        self.x_log = x_log
        self.y = y
        self.indices = torch.from_numpy(indices).long()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_index = self.indices[index]
        return self.x_log[source_index], self.y[source_index]


def make_log_spd_loader(
    x_log: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        IndexedLogSPDDataset(x_log, y, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def train_differentiable_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    non_blocking = device.type == "cuda"

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite differentiable MDM loss detected. "
                "Check covariance scaling and the learning rate."
            )
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += float(loss.item()) * y_batch.size(0)
        total_samples += y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = compute_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


def evaluate_differentiable(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    non_blocking = device.type == "cuda"

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * y_batch.size(0)
            total_samples += y_batch.size(0)
            y_true.extend(y_batch.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


def run_differentiable_mdm(
    x_spd: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    class_names: list[str],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    device: torch.device,
    precision_override: str | None,
    run_dir: Path,
    experiment_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)
    dtype = resolve_precision(
        precision_override or training_cfg.get("precision", "float32")
    )
    eps = float(model_cfg.get("eps", 1e-6))
    x_log_np = matrix_log(x_spd, eps=eps)
    if not np.isfinite(x_log_np).all():
        raise RuntimeError("Matrix logarithm produced NaN or Inf values.")
    x_log = torch.from_numpy(x_log_np).to(dtype=dtype)
    del x_log_np
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64)).long()

    token_shape = token_shape_of(x_spd)
    pooling = normalize_token_pooling(
        model_cfg.get("pooling", model_cfg.get("token_pooling", "mean"))
    )
    model = LogEuclideanPrototypeClassifier(
        spd_dim=int(x_spd.shape[-1]),
        num_classes=len(class_names),
        token_shape=token_shape,
        pooling=pooling,
        eps=eps,
        prototype_init_std=float(model_cfg.get("prototype_init_std", 1e-3)),
    ).to(device=device, dtype=dtype)

    weight_source = None
    if model.token_weight_logits is not None:
        initial_weights, weight_source = resolve_token_weights(model_cfg, token_shape)
        initial_logits = np.log(np.maximum(initial_weights, np.finfo(np.float64).tiny))
        with torch.no_grad():
            model.token_weight_logits.copy_(
                torch.as_tensor(initial_logits, device=device, dtype=dtype)
            )

    batch_size = int(training_cfg.get("batch_size", 64))
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = parse_bool(
        training_cfg.get("pin_memory", device.type == "cuda"),
        default=device.type == "cuda",
    )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": make_log_spd_loader(
            x_log,
            y_tensor,
            train_idx,
            batch_size,
            True,
            num_workers,
            pin_memory,
            generator,
        ),
        "train_eval": make_log_spd_loader(
            x_log,
            y_tensor,
            train_idx,
            batch_size,
            False,
            num_workers,
            pin_memory,
        ),
        "test": make_log_spd_loader(
            x_log,
            y_tensor,
            test_idx,
            batch_size,
            False,
            num_workers,
            pin_memory,
        ),
    }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-2)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    gradient_clip_norm = training_cfg.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    epochs = int(training_cfg.get("epochs", 100))
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}.")

    print(
        f"  classifier=differentiable dtype={dtype} device={device} "
        f"epochs={epochs} batch_size={batch_size} pooling={pooling}",
        flush=True,
    )
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_differentiable_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            gradient_clip_norm,
        )
        test_metrics = evaluate_differentiable(
            model,
            loaders["test"],
            criterion,
            device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
        )
        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"mf1={train_metrics['macro_f1']:.4f} | "
            f"test loss={test_metrics['loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"mf1={test_metrics['macro_f1']:.4f}",
            flush=True,
        )

    write_csv(run_dir / "history.csv", history_rows)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "model_state_dict": state_dict,
            "class_names": class_names,
            "config": experiment_cfg,
            "x_spd_shape": list(x_spd.shape),
            "token_shape": list(token_shape),
        },
        run_dir / "model.pt",
    )

    rows = []
    for split_name, split_idx, loader_name in (
        ("train", train_idx, "train_eval"),
        ("test", test_idx, "test"),
    ):
        metrics = evaluate_differentiable(
            model,
            loaders[loader_name],
            criterion,
            device,
        )
        rows.append(
            {
                "split": split_name,
                "n_samples": int(len(split_idx)),
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    token_weights = model.token_weights()
    pooling_summary = {
        "mode": pooling,
        "token_shape": list(token_shape),
        "has_parameters": token_weights is not None,
    }
    if token_weights is not None:
        pooling_summary.update(
            {
                "weight_source": weight_source,
                "token_weights": token_weights.detach().cpu().tolist(),
            }
        )
    learned_scale = (
        torch.nn.functional.softplus(model.mdm_head.logit_scale_raw) + eps
    )
    details = {
        "metric": {"mean": "logeuclid", "distance": "logeuclid"},
        "x_trial_spd_shape": [int(x_spd.shape[0]), int(x_spd.shape[-1]), int(x_spd.shape[-1])],
        "token_pooling": pooling_summary,
        "training_epochs": epochs,
        "learned_logit_scale": float(learned_scale.detach().cpu()),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return rows, details


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
    data_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    split_cfg = dict(training_cfg)
    if data_cfg is not None:
        for key in ("seed", "test_size", "allow_subject_overlap"):
            if key in data_cfg:
                split_cfg[key] = data_cfg[key]
    test_size = float(split_cfg.get("test_size", 0.2))
    seed = int(split_cfg.get("seed", 42))
    split_file = resolve_split_file(split_cfg.get("split_file"))
    allow_subject_overlap = parse_bool(
        split_cfg.get("allow_subject_overlap", True),
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
    parser.add_argument(
        "--classifier-type",
        default=None,
        help=(
            "Override model.classifier_type: pyriemann, fgmdm, or "
            "differentiable."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--precision",
        default=None,
        help="Override differentiable training precision: float32 or float64.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Use the same pyRiemann metric for class means and distances. "
            "Defaults to config model.metric or riemann."
        ),
    )
    parser.add_argument(
        "--mean-metric",
        default=None,
        help="Override the metric used to compute MDM class centroids.",
    )
    parser.add_argument(
        "--distance-metric",
        default=None,
        help="Override the metric used to classify by distance to centroids.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise incompatible grid combinations instead of recording skips.",
    )
    return parser


def run_experiment(
    run_index: int,
    experiment_cfg: dict,
    cli_classifier_type: str | None,
    cli_metric: str | None,
    cli_mean_metric: str | None,
    cli_distance_metric: str | None,
    base_output_dir: Path,
    dataset_cache_dir: Path,
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]],
    device: torch.device,
    precision_override: str | None,
) -> dict:
    data_cfg = experiment_cfg["data"]
    model_cfg = experiment_cfg.get("model", {})
    training_cfg = experiment_cfg["training"]
    classifier_type = normalize_classifier_type(
        cli_classifier_type or resolve_classifier_type(model_cfg)
    )

    validate_data_model_compatibility(data_cfg, model_cfg)

    x_spd, y, subject_labels, class_names = load_or_preprocess_spd(
        data_cfg,
        dataset_cache_dir,
        data_cache,
    )
    train_idx, test_idx = get_train_test_indices(
        y,
        training_cfg,
        subject_labels=subject_labels,
        data_cfg=data_cfg,
    )

    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"

    if classifier_type in {"pyriemann", "fgmdm"}:
        from pyriemann.classification import FgMDM, MDM

        x_trial_spd, pooling_summary = pool_spd_tokens(
            x_spd,
            model_cfg,
            eps=float(model_cfg.get("eps", data_cfg.get("eps", 1e-8))),
        )
        if classifier_type == "fgmdm":
            metric = resolve_fgmdm_metric(
                model_cfg,
                cli_metric=cli_metric,
                cli_mean_metric=cli_mean_metric,
                cli_distance_metric=cli_distance_metric,
            )
            tsupdate = parse_bool(
                model_cfg.get("fgmdm_tsupdate", model_cfg.get("tsupdate", False)),
                default=False,
            )
            n_jobs = int(model_cfg.get("n_jobs", 1))
            classifier = FgMDM(
                metric=metric,
                tsupdate=tsupdate,
                n_jobs=n_jobs,
            )
        else:
            metric = resolve_mdm_metric(
                model_cfg,
                cli_metric=cli_metric,
                cli_mean_metric=cli_mean_metric,
                cli_distance_metric=cli_distance_metric,
            )
            tsupdate = None
            n_jobs = int(model_cfg.get("n_jobs", 1))
            classifier = MDM(metric=metric, n_jobs=n_jobs)
        classifier.fit(x_trial_spd[train_idx], y[train_idx])

        rows = []
        for split_name, split_idx in {"train": train_idx, "test": test_idx}.items():
            prediction = classifier.predict(x_trial_spd[split_idx])
            row = {
                "split": split_name,
                "n_samples": int(len(split_idx)),
            }
            row.update(compute_metrics(y[split_idx], prediction))
            rows.append(row)
        classifier_details = {
            "metric": metric,
            "x_trial_spd_shape": list(x_trial_spd.shape),
            "token_pooling": pooling_summary,
            "n_jobs": n_jobs,
        }
        if tsupdate is not None:
            classifier_details["tsupdate"] = tsupdate
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "config.json", experiment_cfg)
    else:
        if any(
            not _is_none_like(value)
            for value in (cli_metric, cli_mean_metric, cli_distance_metric)
        ):
            raise ValueError(
                "--metric, --mean-metric, and --distance-metric only apply to "
                "classifier_type='pyriemann' or 'fgmdm'."
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "config.json", experiment_cfg)
        rows, classifier_details = run_differentiable_mdm(
            x_spd=x_spd,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            class_names=class_names,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            device=device,
            precision_override=precision_override,
            run_dir=run_dir,
            experiment_cfg=experiment_cfg,
        )

    write_csv(run_dir / "results.csv", rows)

    summary = {
        "baseline": "mdm",
        "classifier_type": classifier_type,
        "config": experiment_cfg,
        "class_names": class_names,
        "x_spd_shape": list(x_spd.shape),
        "splits": rows,
    }
    summary.update(classifier_details)
    save_json(run_dir / "summary.json", summary)
    print(f"[MDM run {run_index}] {classifier_type} saved {run_dir}")
    return {
        "status": "completed",
        "run_index": run_index,
        "run_dir": str(run_dir),
        "test_accuracy": rows[-1]["accuracy"],
        "test_macro_f1": rows[-1]["macro_f1"],
    }


def skipped_experiment_summary(
    run_index: int,
    experiment_cfg: dict,
    reason: Exception,
) -> dict:
    return {
        "status": "skipped",
        "run_index": run_index,
        "config_hash": config_hash(experiment_cfg),
        "reason": str(reason),
        "config": experiment_cfg,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_mdm_experiments(config)
    experiments = override_mdm_classifier(experiments, args.classifier_type)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/mdm_baseline",
    )
    dataset_cache_dir = resolve_project_path(
        config.get("output", {}).get("dataset_cache_dir"),
        DEFAULT_MDM_DATASET_CACHE_DIR,
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    print(f"MDM dataset cache: {dataset_cache_dir}")
    all_metrics = []
    completed_count = 0
    skipped_count = 0
    for index, experiment in enumerate(experiments, start=1):
        try:
            result = run_experiment(
                index,
                experiment,
                args.classifier_type,
                args.metric,
                args.mean_metric,
                args.distance_metric,
                base_output_dir,
                dataset_cache_dir,
                data_cache,
                device,
                args.precision,
            )
            completed_count += 1
        except (IncompatibleExperiment, ValueError) as exc:
            if args.fail_fast:
                raise
            result = skipped_experiment_summary(index, experiment, exc)
            skipped_count += 1
            print(f"[MDM run {index}] skipped: {exc}")
        all_metrics.append(result)

    save_json(base_output_dir / "summary.json", all_metrics)
    print(
        f"All MDM runs complete: {base_output_dir} | "
        f"completed={completed_count}, skipped={skipped_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
