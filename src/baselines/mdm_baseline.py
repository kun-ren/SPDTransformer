from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
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

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    compute_metrics,
    config_hash,
    expand_grid,
    load_spd_like_train,
    load_yaml,
    matrix_exp,
    matrix_log,
    parse_bool,
    normalize_filter_bank,
    resolve_split_file,
    save_json,
)


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


def mdm_data_grid_values(key: str, value: Any) -> list[Any]:
    if key == "filter_bank":
        if is_filter_bank_scheme(value):
            return [normalize_filter_bank(value)]
        if isinstance(value, list) and value and all(
            is_filter_bank_scheme(item) for item in value
        ):
            return [normalize_filter_bank(item) for item in value]
    if isinstance(value, list):
        return value
    return [value]


def expand_mdm_data_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    if not section:
        return [{}]
    keys = list(section)
    value_lists = [mdm_data_grid_values(key, section[key]) for key in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


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
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def expand_mdm_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_mdm_data_grid(config.get("data", {}))
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


def obvious_original_mdm_token_count(data_cfg: dict[str, Any]) -> int | None:
    token_count = 1
    filter_bank = data_cfg.get("filter_bank")
    if is_filter_bank_scheme(filter_bank):
        token_count *= len(filter_bank)

    brain_region_mode = data_cfg.get("brain_region_mode")
    if not _is_none_like(brain_region_mode):
        # Exact region count is dataset-specific and resolved during preprocessing.
        # Any enabled region preset creates more than one SPD token per trial.
        token_count *= 2

    epoch_tmin = data_cfg.get("epoch_tmin")
    epoch_tmax = data_cfg.get("epoch_tmax")
    segment_duration = data_cfg.get("segment_duration")
    if (
        not _is_none_like(epoch_tmin)
        and not _is_none_like(epoch_tmax)
        and not _is_none_like(segment_duration)
    ):
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
            f"segment_duration={data_cfg.get('segment_duration')!r}, "
            f"epoch=({data_cfg.get('epoch_tmin')!r}, {data_cfg.get('epoch_tmax')!r}), "
            f"brain_region_mode={data_cfg.get('brain_region_mode')!r})."
        )


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
        metric_distance = metric_config_value(base_metric, "distance")
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
) -> tuple[np.ndarray, np.ndarray]:
    test_size = float(training_cfg.get("test_size", 0.2))
    seed = int(training_cfg.get("seed", 42))
    split_file = resolve_split_file(training_cfg.get("split_file"))
    allow_subject_overlap = parse_bool(
        training_cfg.get("allow_subject_overlap", True),
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
    cli_metric: str | None,
    cli_mean_metric: str | None,
    cli_distance_metric: str | None,
    base_output_dir: Path,
) -> dict:
    from pyriemann.classification import MDM

    data_cfg = experiment_cfg["data"]
    model_cfg = experiment_cfg.get("model", {})
    training_cfg = experiment_cfg["training"]

    validate_data_model_compatibility(data_cfg, model_cfg)

    x_spd, y, subject_labels, class_names = load_spd_like_train(data_cfg)
    x_trial_spd, pooling_summary = pool_spd_tokens(
        x_spd,
        model_cfg,
        eps=float(model_cfg.get("eps", data_cfg.get("eps", 1e-8))),
    )
    train_idx, test_idx = get_train_test_indices(
        y,
        training_cfg,
        subject_labels=subject_labels,
    )

    metric = resolve_mdm_metric(
        model_cfg,
        cli_metric=cli_metric,
        cli_mean_metric=cli_mean_metric,
        cli_distance_metric=cli_distance_metric,
    )
    classifier = MDM(metric=metric)
    classifier.fit(x_trial_spd[train_idx], y[train_idx])

    split_indices = {
        "train": train_idx,
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
        "token_pooling": pooling_summary,
        "splits": rows,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[MDM run {run_index}] saved {run_dir}")
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/mdm_baseline",
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = []
    completed_count = 0
    skipped_count = 0
    for index, experiment in enumerate(experiments, start=1):
        try:
            result = run_experiment(
                index,
                experiment,
                args.metric,
                args.mean_metric,
                args.distance_metric,
                base_output_dir,
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
