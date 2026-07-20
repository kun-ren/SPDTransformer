from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.baselines.baseline_utils import (
    compute_metrics,
    config_hash,
    expand_grid,
    expand_data_grid,
    get_split_indices,
    load_spd_like_train,
    load_yaml,
    parse_bool,
    save_json,
)
from src.models.TangentTransformerMDMClassifier import (
    TangentTransformerMDMClassifier,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tangent_transformer_mdm.yaml"


class SPDSequenceDataset(Dataset):
    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            indices: np.ndarray,
            dtype: torch.dtype,
    ) -> None:
        self.x = torch.from_numpy(x[indices]).to(dtype=dtype)
        self.y = torch.from_numpy(y[indices]).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tangent-space native Transformer baseline with weighted "
            "Log-Euclidean MDM classification."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--precision",
        default=None,
        help="Override training precision: float32 or float64.",
    )
    return parser


def expand_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_grid = expand_data_grid(config.get("data", {}))
    model_grid = expand_grid(config.get("model", {}))
    training_grid = expand_grid(config.get("training", {}))
    return [
        {
            "data": deepcopy(data_cfg),
            "model": deepcopy(model_cfg),
            "training": deepcopy(training_cfg),
            "output": deepcopy(config.get("output", {})),
        }
        for data_cfg, model_cfg, training_cfg in itertools.product(
            data_grid,
            model_grid,
            training_grid,
        )
    ]


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


def make_loader(
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        dtype: torch.dtype,
        pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        SPDSequenceDataset(x, y, indices, dtype=dtype),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_model(
        model_cfg: dict[str, Any],
        x_shape: tuple[int, ...],
        num_classes: int,
) -> TangentTransformerMDMClassifier:
    spd_dim = int(x_shape[-1])
    token_shape = tuple(int(size) for size in x_shape[1:-2])
    tangent_dim = spd_dim * (spd_dim + 1) // 2
    d_model_value = model_cfg.get(
        "d_model",
        model_cfg.get("attention_dim", tangent_dim),
    )
    if isinstance(d_model_value, (list, tuple)):
        if len(d_model_value) != 1:
            raise ValueError(
                "This baseline expects one d_model value per expanded run, "
                f"got {d_model_value}."
            )
        d_model_value = d_model_value[0]
    d_model = int(d_model_value)
    dim_feedforward = model_cfg.get(
        "dim_feedforward",
        model_cfg.get("ffn_dim"),
    )
    if dim_feedforward is not None:
        dim_feedforward = int(dim_feedforward)

    return TangentTransformerMDMClassifier(
        spd_dim=spd_dim,
        num_classes=num_classes,
        token_shape=token_shape,
        d_model=d_model,
        nhead=int(model_cfg.get("nhead", model_cfg.get("head_nums", 1))),
        num_layers=int(model_cfg.get("num_layers", model_cfg.get("depth", 1))),
        dim_feedforward=dim_feedforward,
        dropout=float(model_cfg.get("dropout", 0.1)),
        activation=str(model_cfg.get("activation", "gelu")),
        norm_first=parse_bool(model_cfg.get("norm_first", False), default=False),
        pooling=str(model_cfg.get("pooling", "weighted")),
        use_position_embedding=parse_bool(
            model_cfg.get("use_position_embedding", True),
            default=True,
        ),
        eps=float(model_cfg.get("eps", 1e-6)),
    )


def train_one_epoch(
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
                "Non-finite tangent Transformer training loss detected. "
                "Check the input SPD eigenvalues and learning rate."
            )
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total_samples += y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = compute_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


def evaluate(
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
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
            y_true.extend(y_batch.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
    )
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_scheduler(
        optimizer: torch.optim.Optimizer,
        training_cfg: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_name = str(training_cfg.get("lr_scheduler", "none")).lower()
    if scheduler_name in {"", "none", "null", "false", "off"}:
        return None
    if scheduler_name not in {
        "plateau",
        "reduce_on_plateau",
        "reduce_lr_on_plateau",
    }:
        raise ValueError(
            "Only lr_scheduler='plateau' is supported by this baseline, "
            f"got {scheduler_name!r}."
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training_cfg.get("lr_scheduler_factor", 0.5)),
        patience=int(training_cfg.get("lr_scheduler_patience", 5)),
        threshold=float(training_cfg.get("lr_scheduler_threshold", 1e-4)),
        threshold_mode=str(training_cfg.get("lr_scheduler_threshold_mode", "abs")),
        cooldown=int(training_cfg.get("lr_scheduler_cooldown", 0)),
        min_lr=float(training_cfg.get("lr_scheduler_min_lr", 1e-6)),
        eps=float(training_cfg.get("lr_scheduler_eps", 1e-8)),
    )


def run_experiment(
        run_index: int,
        experiment_cfg: dict[str, Any],
        base_output_dir: Path,
        device: torch.device,
        precision_override: str | None,
) -> dict[str, Any]:
    data_cfg = experiment_cfg["data"]
    model_cfg = experiment_cfg["model"]
    training_cfg = experiment_cfg["training"]
    seed = int(training_cfg.get("seed", data_cfg.get("seed", 42)))
    set_seed(seed)

    x_spd, y, subject_labels, class_names = load_spd_like_train(data_cfg)
    split_cfg = dict(data_cfg)
    split_cfg.update(training_cfg)
    split_cfg["seed"] = seed
    train_idx, val_idx, test_idx = get_split_indices(
        y,
        split_cfg,
        subject_labels=subject_labels,
    )

    dtype = resolve_precision(
        precision_override or training_cfg.get("precision", "float32")
    )
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = parse_bool(
        training_cfg.get("pin_memory", device.type == "cuda"),
        default=device.type == "cuda",
    )
    loaders = {
        "train": make_loader(
            x_spd,
            y,
            train_idx,
            batch_size,
            num_workers,
            shuffle=True,
            dtype=dtype,
            pin_memory=pin_memory,
        ),
        "train_eval": make_loader(
            x_spd,
            y,
            train_idx,
            batch_size,
            num_workers,
            shuffle=False,
            dtype=dtype,
            pin_memory=pin_memory,
        ),
        "val": make_loader(
            x_spd,
            y,
            val_idx,
            batch_size,
            num_workers,
            shuffle=False,
            dtype=dtype,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            x_spd,
            y,
            test_idx,
            batch_size,
            num_workers,
            shuffle=False,
            dtype=dtype,
            pin_memory=pin_memory,
        ),
    }

    model = build_model(
        model_cfg,
        x_shape=tuple(x_spd.shape),
        num_classes=len(class_names),
    ).to(device=device, dtype=dtype)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = build_scheduler(optimizer, training_cfg)
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)

    epochs = int(training_cfg.get("epochs", 50))
    early_stopping_patience = training_cfg.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_delta = float(
        training_cfg.get("early_stopping_min_delta", 0.0)
    )

    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", experiment_cfg)

    print(f"\n[Tangent Transformer MDM run {run_index}] {run_dir.name}")
    print(
        f"  x_spd={x_spd.shape} token_shape={model.token_shape} "
        f"spd_dim={model.spd_dim} tangent_dim={model.tangent_dim} "
        f"d_model={model.d_model} classes={class_names}"
    )
    print(
        f"  dtype={dtype} device={device} epochs={epochs} "
        f"batch_size={batch_size} pooling={model.pooling}"
    )

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            gradient_clip_norm,
        )
        train_eval_metrics = evaluate(
            model,
            loaders["train_eval"],
            criterion,
            device,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_eval_metrics["accuracy"],
            "train_macro_f1": train_eval_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history_rows.append(row)

        improved = (
            val_metrics["macro_f1"]
            > best_val_macro_f1 + early_stopping_min_delta
        )
        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_eval_metrics['accuracy']:.4f} "
            f"mf1={train_eval_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"mf1={val_metrics['macro_f1']:.4f}"
        )
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(f"  early stopping at epoch {epoch}")
            break

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    model.load_state_dict(best_state_dict)
    write_csv(run_dir / "history.csv", history_rows)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "class_names": class_names,
            "config": experiment_cfg,
            "x_spd_shape": list(x_spd.shape),
            "token_shape": list(model.token_shape),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
        run_dir / "best_model.pt",
    )

    split_loaders = {
        "train": (train_idx, loaders["train_eval"]),
        "val": (val_idx, loaders["val"]),
        "test": (test_idx, loaders["test"]),
    }
    result_rows: list[dict[str, Any]] = []
    for split_name, (indices, loader) in split_loaders.items():
        metrics = evaluate(model, loader, criterion, device)
        result_rows.append(
            {
                "split": split_name,
                "n_samples": int(len(indices)),
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    write_csv(run_dir / "results.csv", result_rows)

    summary = {
        "baseline": "tangent_transformer_weighted_mdm",
        "architecture": (
            "SPD Log(I) map -> isometric vech -> torch TransformerEncoder -> "
            "symmetric tangent matrix -> weighted Log-Euclidean MDM"
        ),
        "x_spd_shape": list(x_spd.shape),
        "token_shape": list(model.token_shape),
        "spd_dim": model.spd_dim,
        "tangent_dim": model.tangent_dim,
        "d_model": model.d_model,
        "pooling": model.pooling,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "splits": result_rows,
    }
    save_json(run_dir / "summary.json", summary)
    test_row = result_rows[-1]
    print(
        f"[Tangent Transformer MDM run {run_index}] done | "
        f"best_epoch={best_epoch} best_val_mf1={best_val_macro_f1:.4f} "
        f"test_acc={test_row['accuracy']:.4f} "
        f"test_mf1={test_row['macro_f1']:.4f}"
    )
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test_accuracy": test_row["accuracy"],
        "test_macro_f1": test_row["macro_f1"],
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_experiments(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/tangent_transformer_mdm_baseline",
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = [
        run_experiment(
            index,
            experiment,
            base_output_dir,
            device,
            args.precision,
        )
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All tangent Transformer MDM runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
