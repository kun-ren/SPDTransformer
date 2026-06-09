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

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.datasets.PhysioNetMI_preprocess import preprocess_spd
from src.models.SPDTransformer import SPDTransformerClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_grid.yaml"


class SPDDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


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


def make_run_dir(base_dir: Path, run_index: int, config: dict[str, Any]) -> Path:
    run_id = f"run_{run_index:03d}_{config_hash(config)}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def split_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=y[train_val_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        SPDDataset(x[train_idx], y[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        SPDDataset(x[val_idx], y[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        SPDDataset(x[test_idx], y[test_idx]),
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
) -> SPDTransformerClassifier:
    return SPDTransformerClassifier(
        spd_in_dim=spd_in_dim,
        spd_out_dim=int(model_cfg.get("spd_out_dim", spd_in_dim)),
        num_classes=num_classes,
        ffn_hidden_spd_dim=model_cfg.get("ffn_hidden_spd_dim"),
        metric=str(model_cfg.get("metric", "log-euclidean")),
        depth=int(model_cfg.get("depth", 1)),
        classifier_type=str(model_cfg.get("classifier_type", "pooling")),
        pooling=str(model_cfg.get("pooling", "attention")),
        dropout=float(model_cfg.get("dropout", 0.0)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
        debug_attention_dropout=bool(model_cfg.get("debug_attention_dropout", False)),
        debug_attention_shape=bool(model_cfg.get("debug_attention_shape", False)),
        learnable_metric_mode=str(model_cfg.get("learnable_metric_mode", "low-rank")),
        learnable_metric_rank=model_cfg.get("learnable_metric_rank"),
        metric_eps=float(model_cfg.get("metric_eps", 1e-6)),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def append_history(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_experiment(
    run_index: int,
    experiment_cfg: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    base_output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    training_cfg = experiment_cfg["training"]
    model_cfg = experiment_cfg["model"]
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    run_dir = make_run_dir(base_output_dir, run_index, experiment_cfg)
    save_yaml(run_dir / "config.yaml", experiment_cfg)

    train_idx, val_idx, test_idx = split_indices(
        y=y,
        test_size=float(training_cfg.get("test_size", 0.15)),
        val_size=float(training_cfg.get("val_size", 0.15)),
        seed=seed,
    )
    train_loader, val_loader, test_loader = make_loaders(
        x=x,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        batch_size=int(training_cfg.get("batch_size", 16)),
        num_workers=int(training_cfg.get("num_workers", 0)),
    )

    model = build_model(
        model_cfg=model_cfg,
        spd_in_dim=x.shape[-1],
        num_classes=len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    best_val_macro_f1 = -1.0
    best_epoch = 0
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_model.pt"
    epochs = int(training_cfg.get("epochs", 50))

    print(f"\n[Run {run_index}] {run_dir.name}")
    print(f"  model={model_cfg}")
    print(f"  training={training_cfg}")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        append_history(history_path, row)

        if val_metrics["macro_f1"] > best_val_macro_f1:
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    metrics = {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "class_names": class_names,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
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


def preprocess_dataset(data_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    filter_bank = normalize_filter_bank(data_cfg["filter_bank"])
    x, y, class_names = preprocess_spd(
        filter_bank=filter_bank,
        root_dir=str(data_cfg.get("root_dir", "data/MNE-eegbci-data/files/eegmmidb/1.0.0")),
        estimator=str(data_cfg.get("estimator", "lwf")),
        sfreq=float(data_cfg.get("sfreq", 160)),
        eps=float(data_cfg.get("eps", 1e-6)),
        segment_duration=float(data_cfg.get("segment_duration", 1.0)),
        stride_duration=data_cfg.get("stride_duration", 0.5),
    )
    return x.astype(np.float32), y.astype(np.int64), list(class_names)


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

    # Preprocess once from the first expanded data config. If you grid data
    # preprocessing parameters, each distinct data config will be handled below.
    data_cache: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    all_metrics = []

    print(f"Generated {len(experiments)} experiment(s)")
    print(f"Saving runs under: {base_output_dir}")
    print(f"Device: {device}")

    for run_index, experiment_cfg in enumerate(experiments, start=1):
        data_key = config_hash({"data": experiment_cfg["data"]})
        if data_key not in data_cache:
            print(f"\nPreprocessing data config {data_key}: {experiment_cfg['data']}")
            data_cache[data_key] = preprocess_dataset(experiment_cfg["data"])
            x_cached, y_cached, names_cached = data_cache[data_key]
            print(f"  X.shape={x_cached.shape}, y.shape={y_cached.shape}, classes={names_cached}")

        x, y, class_names = data_cache[data_key]
        metrics = train_experiment(
            run_index=run_index,
            experiment_cfg=experiment_cfg,
            x=x,
            y=y,
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
