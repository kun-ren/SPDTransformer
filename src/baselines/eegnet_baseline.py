from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    compute_metrics,
    config_hash,
    expand_data_training_experiments,
    get_split_indices,
    load_segmented_epochs_like_train,
    load_yaml,
    save_json,
)


class EEGWindowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        trial_indices: np.ndarray,
        dtype: torch.dtype,
        input_scale: float,
    ) -> None:
        windows = x[trial_indices]
        n_trials, n_segments, n_bands, n_channels, n_samples = windows.shape
        self.x = torch.from_numpy(
            windows.reshape(n_trials * n_segments * n_bands, n_channels, n_samples)
        ).to(dtype=dtype)
        self.x = self.x * input_scale
        self.y = torch.from_numpy(
            np.repeat(y[trial_indices], n_segments * n_bands)
        ).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        num_classes: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feature_dim = self.features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, channels, samples)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EEGNet baseline using the same data config as train.py."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--f2", type=int, default=16)
    parser.add_argument("--kernel-length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--output-dir", default="experiments/results/eegnet_baseline")
    return parser


def resolve_precision(precision: Any) -> torch.dtype:
    precision = str(precision or "float32").lower()
    if precision in {"float64", "double", "fp64"}:
        return torch.float64
    if precision in {"float32", "float", "single", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


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
    input_scale: float,
) -> DataLoader:
    return DataLoader(
        EEGWindowDataset(x, y, indices, dtype=dtype, input_scale=input_scale),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite EEGNet training loss detected.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total_samples += y_batch.size(0)
    return total_loss / total_samples


def predict_trials(
    model: nn.Module,
    x: np.ndarray,
    trial_indices: np.ndarray,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    input_scale: float,
) -> np.ndarray:
    model.eval()
    windows = x[trial_indices]
    n_trials, n_segments, n_bands, n_channels, n_samples = windows.shape
    flat_windows = windows.reshape(n_trials * n_segments * n_bands, n_channels, n_samples)
    logits = []
    with torch.no_grad():
        for start in range(0, len(flat_windows), batch_size):
            batch = torch.from_numpy(flat_windows[start:start + batch_size]).to(
                device=device,
                dtype=dtype,
            )
            batch = batch * input_scale
            logits.append(model(batch).cpu())
    flat_logits = torch.cat(logits, dim=0).numpy()
    trial_logits = flat_logits.reshape(n_trials, n_segments * n_bands, -1).mean(axis=1)
    return trial_logits.argmax(axis=1)


def evaluate_trials(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    input_scale: float,
) -> dict[str, float]:
    prediction = predict_trials(
        model=model,
        x=x,
        trial_indices=indices,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        input_scale=input_scale,
    )
    return compute_metrics(y[indices], prediction)


def run_experiment(
    run_index: int,
    experiment_cfg: dict,
    args: argparse.Namespace,
    base_output_dir: Path,
    device: torch.device,
) -> dict:
    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    x, y, class_names, filter_bank = load_segmented_epochs_like_train(data_cfg)
    train_idx, val_idx, test_idx = get_split_indices(y, training_cfg)

    dtype = resolve_precision(training_cfg.get("precision", "float32"))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)

    train_loader = make_loader(
        x=x,
        y=y,
        indices=train_idx,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        dtype=dtype,
        input_scale=args.input_scale,
    )

    n_channels = x.shape[-2]
    n_samples = x.shape[-1]
    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        num_classes=len(class_names),
        f1=args.f1,
        depth_multiplier=args.depth_multiplier,
        f2=args.f2,
        kernel_length=args.kernel_length,
        dropout=args.dropout,
    ).to(device=device, dtype=dtype)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )

    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "history.csv"
    checkpoint_path = run_dir / "best_model.pt"
    epochs = int(training_cfg.get("epochs", 50))

    best_val_macro_f1 = -1.0
    best_epoch = 0
    history_rows = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip_norm,
        )
        train_metrics = evaluate_trials(
            model, x, y, train_idx, batch_size, dtype, device, args.input_scale
        )
        val_metrics = evaluate_trials(
            model, x, y, val_idx, batch_size, dtype, device, args.input_scale
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history_rows.append(row)
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
            f"[EEGNet run {run_index}] epoch {epoch:03d}/{epochs} "
            f"loss={train_loss:.4f} "
            f"train_mf1={train_metrics['macro_f1']:.4f} "
            f"val_mf1={val_metrics['macro_f1']:.4f}"
        )

    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    rows = []
    for split_name, split_idx in split_indices.items():
        metrics = evaluate_trials(
            model, x, y, split_idx, batch_size, dtype, device, args.input_scale
        )
        row = {
            "split": split_name,
            "n_samples": int(len(split_idx)),
        }
        row.update(metrics)
        rows.append(row)

    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "baseline": "eegnet",
        "config": experiment_cfg,
        "class_names": class_names,
        "filter_bank": filter_bank,
        "x_shape": list(x.shape),
        "window_training": "trial windows are band x segment windows; trial prediction averages window logits",
        "input_scale": args.input_scale,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "splits": rows,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"[EEGNet run {run_index}] saved {run_dir}")
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "test_accuracy": rows[-1]["accuracy"],
        "test_macro_f1": rows[-1]["macro_f1"],
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = PROJECT_ROOT / args.output_dir / timestamp
    all_metrics = [
        run_experiment(index, experiment, args, base_output_dir, device)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All EEGNet runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
