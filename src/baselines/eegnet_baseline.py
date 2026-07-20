from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.baselines.baseline_utils import (
    compute_metrics,
    config_hash,
    expand_data_training_experiments,
    load_eegnet_author_data,
    load_yaml,
    parse_bool,
    save_json,
)


DEFAULT_EEGNET_CONFIG = PROJECT_ROOT / "configs" / "eegnet_author.yaml"


class EEGTrialDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        dtype: torch.dtype,
        input_scale: float,
    ) -> None:
        self.x = torch.from_numpy(x[indices]).to(dtype=dtype) * input_scale
        self.y = torch.from_numpy(y[indices]).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def _same_padding_1d(kernel_length: int) -> tuple[int, int]:
    total_padding = int(kernel_length) - 1
    left = total_padding // 2
    right = total_padding - left
    return left, right


class AuthorEEGNet(nn.Module):
    """
    PyTorch implementation of the EEGNet variant in MHersche's repository.

    The layer topology and default hyperparameters mirror models.EEGNet:
    temporal Conv2D -> depthwise spatial Conv2D -> separable Conv2D -> Dense.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        num_classes: int,
        reg_rate: float = 0.25,
        dropout_rate: float = 0.2,
        kern_length: int = 128,
        pool_length: int = 8,
        num_filters: int = 8,
        dropout_type: str = "Dropout",
        apply_max_norm: bool = True,
    ) -> None:
        super().__init__()
        f1 = int(num_filters)
        depth_multiplier = 2
        f2 = f1 * 2
        dropout_cls: type[nn.Module]
        if dropout_type == "SpatialDropout2D":
            dropout_cls = nn.Dropout2d
        elif dropout_type == "Dropout":
            dropout_cls = nn.Dropout
        else:
            raise ValueError("dropout_type must be 'Dropout' or 'SpatialDropout2D'.")

        temporal_left, temporal_right = _same_padding_1d(kern_length)
        separable_left, separable_right = _same_padding_1d(16)

        self.temporal_pad = nn.ZeroPad2d((temporal_left, temporal_right, 0, 0))
        self.temporal_conv = nn.Conv2d(
            1,
            f1,
            kernel_size=(1, kern_length),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(f1)

        self.depthwise_conv = nn.Conv2d(
            f1,
            f1 * depth_multiplier,
            kernel_size=(n_channels, 1),
            groups=f1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool_length))
        self.dropout1 = dropout_cls(dropout_rate)

        self.separable_pad = nn.ZeroPad2d((separable_left, separable_right, 0, 0))
        self.separable_depthwise = nn.Conv2d(
            f1 * depth_multiplier,
            f1 * depth_multiplier,
            kernel_size=(1, 16),
            groups=f1 * depth_multiplier,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(
            f1 * depth_multiplier,
            f2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(f2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = dropout_cls(dropout_rate)

        self.reg_rate = float(reg_rate)
        self.apply_max_norm = bool(apply_max_norm)

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_samples)
            feature_dim = self._forward_features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(self.temporal_pad(x))
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable_depthwise(self.separable_pad(x))
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.classifier(x)

    @staticmethod
    def _max_norm_(weight: torch.Tensor, max_norm: float) -> None:
        flat = weight.view(weight.shape[0], -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        desired = norm.clamp(max=float(max_norm))
        flat.mul_(desired / norm)

    def apply_author_constraints_(self) -> None:
        if not self.apply_max_norm:
            return
        with torch.no_grad():
            self._max_norm_(self.depthwise_conv.weight, 1.0)
            self._max_norm_(self.classifier.weight, self.reg_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Author-style EEGNet baseline for motor imagery EEG datasets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_EEGNET_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-scale", type=float, default=None)
    parser.add_argument("--disable-max-norm", action="store_true")
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


def parse_learning_rate_schedule(value: Any, default_lr: float) -> list[tuple[int, float]]:
    if value in {None, ""}:
        return [(0, float(default_lr))]
    if isinstance(value, (int, float)):
        return [(0, float(value))]
    if isinstance(value, str):
        steps = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            epoch_text, lr_text = part.split(":", 1)
            steps.append((int(epoch_text), float(lr_text)))
        if not steps:
            return [(0, float(default_lr))]
        return sorted(steps, key=lambda item: item[0])
    raise ValueError(f"Unsupported learning_rate_schedule: {value!r}")


def scheduled_lr(schedule: list[tuple[int, float]], epoch_index: int) -> float:
    lr = schedule[0][1]
    for start_epoch, candidate_lr in schedule:
        if epoch_index >= start_epoch:
            lr = candidate_lr
        else:
            break
    return float(lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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
        EEGTrialDataset(x, y, indices, dtype=dtype, input_scale=input_scale),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def train_one_epoch(
    model: AuthorEEGNet,
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
        model.apply_author_constraints_()

        total_loss += loss.item() * y_batch.size(0)
        total_samples += y_batch.size(0)
    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

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


def build_model_from_config(
    x: np.ndarray,
    num_classes: int,
    data_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    apply_max_norm: bool,
) -> AuthorEEGNet:
    n_ds = int(data_cfg.get("eegnet_downsample", 1))
    return AuthorEEGNet(
        n_channels=x.shape[1],
        n_samples=x.shape[2],
        num_classes=num_classes,
        reg_rate=float(training_cfg.get("eegnet_reg_rate", 0.25)),
        dropout_rate=float(training_cfg.get("dropout", 0.2)),
        kern_length=int(np.ceil(128 / n_ds)),
        pool_length=int(np.ceil(8 / n_ds)),
        num_filters=int(training_cfg.get("eegnet_num_filters", 8)),
        dropout_type=str(training_cfg.get("dropout_type", "Dropout")),
        apply_max_norm=apply_max_norm,
    )


def make_train_test_split(
    y: np.ndarray,
    subject_labels: np.ndarray,
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split_cfg = dict(training_cfg)
    if data_cfg is not None:
        for key in ("seed", "test_size", "allow_subject_overlap"):
            if key in data_cfg:
                split_cfg[key] = data_cfg[key]
    seed = int(split_cfg.get("seed", 42))
    test_size = float(split_cfg.get("test_size", 0.2))
    allow_subject_overlap = parse_bool(
        split_cfg.get("allow_subject_overlap", False),
        default=False,
    )
    indices = np.arange(len(y), dtype=np.int64)

    if allow_subject_overlap:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )
        split_strategy = "trial"
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(
            splitter.split(indices, y, groups=subject_labels)
        )
        train_idx = indices[train_idx]
        test_idx = indices[test_idx]
        split_strategy = "subject"

    train_subjects = set(subject_labels[train_idx].tolist())
    test_subjects = set(subject_labels[test_idx].tolist())
    subject_overlap = sorted(train_subjects & test_subjects)
    if not allow_subject_overlap and subject_overlap:
        raise RuntimeError(
            "Subject-level EEGNet split failed: train/test subject overlap "
            f"detected: {subject_overlap}."
        )

    metadata = {
        "split_strategy": split_strategy,
        "allow_subject_overlap": allow_subject_overlap,
        "test_size": test_size,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_subjects": int(len(train_subjects)),
        "n_test_subjects": int(len(test_subjects)),
        "subject_overlap": subject_overlap,
        "train_idx": train_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
    }
    return train_idx.astype(np.int64), test_idx.astype(np.int64), metadata


def run_experiment(
    run_index: int,
    experiment_cfg: dict[str, Any],
    args: argparse.Namespace,
    base_output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]
    seed = int(training_cfg.get("seed", data_cfg.get("seed", 42)))
    set_seed(seed)

    x, y, subject_labels, class_names = load_eegnet_author_data(data_cfg)
    dtype = resolve_precision(training_cfg.get("precision", "float32"))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    epochs = int(training_cfg.get("epochs", 100))
    input_scale = (
        float(args.input_scale)
        if args.input_scale is not None
        else float(training_cfg.get("input_scale", 1.0))
    )
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 0.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)

    schedule = parse_learning_rate_schedule(
        training_cfg.get("learning_rate_schedule"),
        default_lr=float(training_cfg.get("learning_rate", 1e-2)),
    )
    train_idx, test_idx, split_metadata = make_train_test_split(
        y,
        subject_labels,
        training_cfg,
        data_cfg=data_cfg,
    )
    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", experiment_cfg)
    save_json(run_dir / "split.json", split_metadata)

    print(f"\n[EEGNet run {run_index}] {run_dir.name}")
    print(f"  data shape={x.shape} class_names={class_names}")
    print(
        "  split="
        f"{split_metadata['split_strategy']} "
        f"train/test={split_metadata['n_train']}/{split_metadata['n_test']} "
        f"subjects={split_metadata['n_train_subjects']}/"
        f"{split_metadata['n_test_subjects']} "
        f"allow_subject_overlap={split_metadata['allow_subject_overlap']}"
    )
    print(f"  lr_schedule={schedule}")

    train_loader = make_loader(
        x,
        y,
        train_idx,
        batch_size,
        num_workers,
        shuffle=True,
        dtype=dtype,
        input_scale=input_scale,
    )
    train_eval_loader = make_loader(
        x,
        y,
        train_idx,
        batch_size,
        num_workers,
        shuffle=False,
        dtype=dtype,
        input_scale=input_scale,
    )
    test_loader = make_loader(
        x,
        y,
        test_idx,
        batch_size,
        num_workers,
        shuffle=False,
        dtype=dtype,
        input_scale=input_scale,
    )
    model = build_model_from_config(
        x,
        num_classes=len(class_names),
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        apply_max_norm=not args.disable_max_norm,
    ).to(device=device, dtype=dtype)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-2)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    history_rows = []
    for epoch in range(1, epochs + 1):
        lr = scheduled_lr(schedule, epoch - 1)
        set_optimizer_lr(optimizer, lr)
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip_norm,
        )
        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
        }
        history_rows.append(row)
        print(
            f"[EEGNet run {run_index}] epoch {epoch:03d}/{epochs} "
            f"lr={lr:.1e} loss={train_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f}"
        )

    write_csv(run_dir / "history.csv", history_rows)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "config": experiment_cfg,
            "split": split_metadata,
            "epochs": epochs,
        },
        run_dir / "final_model.pt",
    )

    final_train_metrics = evaluate(model, train_eval_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    result_rows = [
        {
            "split": "train",
            "n_samples": int(len(train_idx)),
            "loss": final_train_metrics["loss"],
            "accuracy": final_train_metrics["accuracy"],
            "macro_f1": final_train_metrics["macro_f1"],
        },
        {
            "split": "test",
            "n_samples": int(len(test_idx)),
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
        },
    ]
    write_csv(run_dir / "results.csv", result_rows)
    summary = {
        "baseline": "eegnet_author",
        "author_repository": "https://github.com/MHersche/eegnet-based-embedded-bci",
        "x_shape": list(x.shape),
        "class_names": class_names,
        "subjects": int(len(np.unique(subject_labels))),
        "split": split_metadata,
        "train": result_rows[0],
        "test": result_rows[1],
    }
    save_json(run_dir / "summary.json", summary)
    print(
        f"[EEGNet run {run_index}] test acc="
        f"{test_metrics['accuracy']:.4f} test mf1={test_metrics['macro_f1']:.4f} "
        f"| saved {run_dir}"
    )
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "allow_subject_overlap": split_metadata["allow_subject_overlap"],
        "split_strategy": split_metadata["split_strategy"],
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/eegnet_baseline",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = [
        run_experiment(index, experiment, args, base_output_dir, device)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All EEGNet runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
