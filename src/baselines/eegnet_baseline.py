"""EEGNet-8,2 baseline with subject-wise and transfer-learning protocols.

The network follows the official EEGNet 2018 implementation.  Evaluation is
kept separate from optimization: every model is trained for a fixed number of
epochs and a test partition is evaluated exactly once after training.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.baseline_utils import (
    DEFAULT_CONFIG,
    compute_metrics,
    config_hash,
    expand_data_training_experiments,
    load_eegnet_author_data,
    load_segmented_epochs_like_train,
    load_yaml,
    make_subject_specific_trial_splits,
    parse_bool,
    parse_subjects,
    save_json,
    summarize_subject_fold_metrics,
)


REPORT_METRICS = ("accuracy", "macro_f1", "cohen_kappa")
OFFICIAL_REPOSITORY = "https://github.com/vlawhern/arl-eegmodels"
OFFICIAL_PAPER = "https://doi.org/10.1088/1741-2552/aace8c"
PHYSIONET_REPOSITORY = "https://github.com/MHersche/eegnet-based-embedded-bci"
PHYSIONET_PAPER = "https://arxiv.org/abs/2004.00077"


class EEGTrialDataset(Dataset):
    """Lazy trial dataset with train-only channel standardization."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        *,
        input_scale: float,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.x = x
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)
        self.input_scale = float(input_scale)
        self.mean = torch.from_numpy(mean.astype(np.float32))[:, None]
        self.std = torch.from_numpy(std.astype(np.float32))[:, None]

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        trial = torch.from_numpy(self.x[index]).float()
        trial = (trial * self.input_scale - self.mean) / self.std
        return trial.unsqueeze(0), torch.tensor(int(self.y[index]), dtype=torch.long)


class EEGNet(nn.Module):
    """PyTorch port of the official EEGNet-8,2 architecture."""

    def __init__(
        self,
        *,
        num_classes: int,
        channels: int,
        samples: int,
        kernel_length: int = 64,
        separable_kernel_length: int = 16,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        dropout_rate: float = 0.5,
        pool1_length: int = 4,
        pool2_length: int = 8,
        depthwise_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
        batch_norm_momentum: float = 0.01,
        batch_norm_epsilon: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if min(
            num_classes,
            channels,
            samples,
            kernel_length,
            f1,
            depth_multiplier,
            f2,
            pool1_length,
            pool2_length,
        ) < 1:
            raise ValueError("EEGNet dimensions must be positive.")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        if samples < pool1_length * pool2_length:
            raise ValueError(
                "EEGNet input is shorter than its temporal pooling reduction."
            )

        f1d = int(f1 * depth_multiplier)
        self.temporal_conv = nn.Conv2d(
            1,
            f1,
            kernel_size=(1, kernel_length),
            padding="same",
            bias=False,
        )
        self.temporal_bn = nn.BatchNorm2d(
            f1,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon,
        )
        self.depthwise_conv = nn.Conv2d(
            f1,
            f1d,
            kernel_size=(channels, 1),
            groups=f1,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(
            f1d,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon,
        )
        self.activation = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1_length))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.separable_depthwise = nn.Conv2d(
            f1d,
            f1d,
            kernel_size=(1, separable_kernel_length),
            padding="same",
            groups=f1d,
            bias=False,
        )
        self.separable_pointwise = nn.Conv2d(f1d, f2, kernel_size=1, bias=False)
        self.separable_bn = nn.BatchNorm2d(
            f2,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon,
        )
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2_length))
        self.dropout2 = nn.Dropout(dropout_rate)

        pooled_samples = (samples // pool1_length) // pool2_length
        self.classifier = nn.Linear(f2 * pooled_samples, num_classes)
        self.depthwise_max_norm = float(depthwise_max_norm)
        self.classifier_max_norm = float(classifier_max_norm)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras defaults used by the author code are Glorot-uniform kernels.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _constrain_output_filters_(weight: torch.Tensor, max_norm: float) -> None:
        if max_norm <= 0.0:
            return
        with torch.no_grad():
            flattened = weight.reshape(weight.shape[0], -1)
            norms = flattened.norm(p=2, dim=1, keepdim=True).clamp_min(1.0e-12)
            scales = torch.clamp(max_norm / norms, max=1.0)
            weight.mul_(scales.reshape((-1,) + (1,) * (weight.ndim - 1)))

    def apply_max_norm_(self) -> None:
        self._constrain_output_filters_(
            self.depthwise_conv.weight,
            self.depthwise_max_norm,
        )
        self._constrain_output_filters_(
            self.classifier.weight,
            self.classifier_max_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_bn(self.temporal_conv(x))
        x = self.depthwise_conv(x)
        x = self.dropout1(self.pool1(self.activation(self.depthwise_bn(x))))
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.dropout2(self.pool2(self.activation(self.separable_bn(x))))
        return self.classifier(x.flatten(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_protocol(value: Any) -> str:
    protocol = str(value or "subject_wise").strip().lower().replace("-", "_")
    aliases = {
        "subject_specific": "subject_wise",
        "subject_dependent": "subject_wise",
        "subject_wise": "subject_wise",
        "subject_wise_cv": "subject_wise",
        "transfer": "transfer",
        "pretrain_finetune": "transfer",
        "pretrain_fine_tune": "transfer",
        "paper_global_ss_tl": "paper_global_ss_tl",
        "global_ss_tl": "paper_global_ss_tl",
        "physionet_paper": "paper_global_ss_tl",
    }
    if protocol not in aliases:
        raise ValueError(
            f"Unknown EEGNet protocol {value!r}; use subject_wise, transfer, "
            "or paper_global_ss_tl."
        )
    return aliases[protocol]


def extract_single_trial_eeg(x: np.ndarray) -> np.ndarray:
    if x.ndim != 5:
        raise ValueError(
            "EEGNet expects (trial, segment, band, channel, sample), "
            f"got {x.shape}."
        )
    if x.shape[1] != 1 or x.shape[2] != 1:
        raise ValueError(
            "EEGNet consumes one full trial and one frequency band. Set one "
            f"segment and one band; got segments={x.shape[1]}, bands={x.shape[2]}."
        )
    return np.asarray(x[:, 0, 0], dtype=np.float32)


def fit_channel_standardizer(
    x: np.ndarray,
    indices: np.ndarray,
    *,
    input_scale: float,
    enabled: bool,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    channels = int(x.shape[1])
    if not enabled:
        return np.zeros(channels, dtype=np.float32), np.ones(channels, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        raise ValueError("Cannot fit standardization on an empty training set.")
    total = np.zeros(channels, dtype=np.float64)
    total_square = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        values = x[indices[start : start + chunk_size]].astype(np.float64)
        values *= float(input_scale)
        total += values.sum(axis=(0, 2))
        total_square += np.square(values).sum(axis=(0, 2))
        count += int(values.shape[0] * values.shape[2])
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1.0e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    input_scale: float,
    mean: np.ndarray,
    std: np.ndarray,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        EEGTrialDataset(
            x,
            y,
            indices,
            input_scale=input_scale,
            mean=mean,
            std=std,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def train_one_epoch(
    model: EEGNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_batch), y_batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite EEGNet training loss detected.")
        loss.backward()
        optimizer.step()
        model.apply_max_norm_()
        total_loss += float(loss.item()) * y_batch.size(0)
        total_samples += y_batch.size(0)
    return total_loss / max(total_samples, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * y_batch.size(0)
            total_samples += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    true_array = np.asarray(y_true, dtype=np.int64)
    pred_array = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_metrics(true_array, pred_array)
    metrics.update(
        {
            "loss": total_loss / max(total_samples, 1),
            "cohen_kappa": float(cohen_kappa_score(true_array, pred_array)),
            "y_true": true_array,
            "y_pred": pred_array,
        }
    )
    return metrics


def train_fixed_epochs(
    model: EEGNet,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    learning_rate_schedule: Any = None,
    device: torch.device,
    log_every: int,
    label: str,
) -> list[dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    # Match Keras Adam defaults used by the official implementation.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-7,
        weight_decay=0.0,
    )
    history: list[dict[str, Any]] = []
    schedule = parse_learning_rate_schedule(
        learning_rate_schedule,
        default_learning_rate=learning_rate,
    )
    for epoch in range(1, epochs + 1):
        epoch_index = epoch - 1
        current_learning_rate = next(
            rate
            for start, rate in reversed(schedule)
            if epoch_index >= start
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        should_log = epoch == 1 or epoch % log_every == 0 or epoch == epochs
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_loss,
            "train_accuracy": None,
            "train_macro_f1": None,
        }
        if should_log:
            metrics = evaluate(model, train_eval_loader, criterion, device)
            row["train_accuracy"] = metrics["accuracy"]
            row["train_macro_f1"] = metrics["macro_f1"]
            print(
                f"[{label}] epoch {epoch:03d}/{epochs} | "
                f"loss={train_loss:.4f} accuracy={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )
        history.append(row)
    return history


def parse_learning_rate_schedule(
    value: Any,
    *,
    default_learning_rate: float,
) -> list[tuple[int, float]]:
    """Parse an epoch:rate schedule without making it a YAML grid dimension."""

    if value is None or str(value).strip() == "":
        return [(0, float(default_learning_rate))]
    schedule: list[tuple[int, float]] = []
    for item in str(value).split(","):
        epoch_text, separator, rate_text = item.strip().partition(":")
        if not separator:
            raise ValueError(
                "learning_rate_schedule must use epoch:rate pairs, for example "
                "0:0.01,20:0.001,50:0.0001."
            )
        epoch = int(epoch_text)
        rate = float(rate_text)
        if epoch < 0 or rate <= 0.0:
            raise ValueError("Learning-rate schedule epochs and rates must be positive.")
        schedule.append((epoch, rate))
    schedule.sort(key=lambda item: item[0])
    if not schedule or schedule[0][0] != 0:
        schedule.insert(0, (0, float(default_learning_rate)))
    if len({epoch for epoch, _rate in schedule}) != len(schedule):
        raise ValueError("Learning-rate schedule contains duplicate epochs.")
    return schedule


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subject_wise_splits(
    y: np.ndarray,
    subject_labels: np.ndarray,
    *,
    n_splits: int,
    train_size: float,
    test_size: float,
    seed: int,
) -> list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]]:
    """Return SPDTransformer-aligned 70/30 or optional subject-wise K-fold splits."""

    if n_splits <= 1:
        return make_subject_specific_trial_splits(
            y,
            subject_labels,
            train_size=train_size,
            test_size=test_size,
            seed=seed,
        )

    from sklearn.model_selection import StratifiedKFold

    splits: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for position, subject in enumerate(sorted(np.unique(subject_labels).tolist())):
        subject_idx = np.flatnonzero(subject_labels == subject)
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + position,
        )
        for fold, (local_train, local_test) in enumerate(
            splitter.split(subject_idx, y[subject_idx]),
            start=1,
        ):
            splits.append(
                (
                    str(subject),
                    fold,
                    subject_idx[local_train].astype(np.int64),
                    np.empty(0, dtype=np.int64),
                    subject_idx[local_test].astype(np.int64),
                )
            )
    return splits


def stratified_train_test_split(
    y: np.ndarray,
    indices: np.ndarray,
    *,
    train_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    if not np.isclose(train_size + test_size, 1.0):
        raise ValueError("train_size and test_size must sum to 1.0.")
    train_idx, test_idx = train_test_split(
        np.asarray(indices, dtype=np.int64),
        train_size=train_size,
        test_size=test_size,
        shuffle=True,
        stratify=y[indices],
        random_state=seed,
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def stratified_train_validation_test_split(
    y: np.ndarray,
    indices: np.ndarray,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split pooled trials into stratified train/validation/test partitions."""

    from sklearn.model_selection import train_test_split

    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("Pretrain train/validation/test sizes must sum to 1.0.")
    indices = np.asarray(indices, dtype=np.int64)
    train_validation_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        shuffle=True,
        stratify=y[indices],
        random_state=seed,
    )
    validation_fraction = validation_size / (1.0 - test_size)
    train_idx, validation_idx = train_test_split(
        train_validation_idx,
        test_size=validation_fraction,
        shuffle=True,
        stratify=y[train_validation_idx],
        random_state=seed + 1,
    )
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(validation_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def global_subject_folds(
    subject_labels: np.ndarray,
    *,
    n_splits: int,
) -> list[tuple[int, np.ndarray, np.ndarray, list[str], list[str]]]:
    """Match the paper's non-shuffled KFold split over ordered subjects."""

    from sklearn.model_selection import KFold

    subjects = np.asarray(sorted(np.unique(subject_labels).astype(str).tolist()))
    splitter = KFold(n_splits=n_splits, shuffle=False)
    folds = []
    for fold, (train_subject_idx, test_subject_idx) in enumerate(
        splitter.split(subjects),
        start=1,
    ):
        train_subjects = subjects[train_subject_idx].tolist()
        test_subjects = subjects[test_subject_idx].tolist()
        train_idx = np.flatnonzero(np.isin(subject_labels, train_subjects)).astype(
            np.int64
        )
        test_idx = np.flatnonzero(np.isin(subject_labels, test_subjects)).astype(
            np.int64
        )
        folds.append((fold, train_idx, test_idx, train_subjects, test_subjects))
    return folds


def resolve_target_labels(
    requested_subjects: list[int] | None,
    available_labels: np.ndarray,
) -> list[str]:
    available = sorted(np.unique(available_labels).astype(str).tolist())
    if requested_subjects is None:
        return available
    by_number = {int(label.upper().lstrip("S")): label for label in available}
    missing = [subject for subject in requested_subjects if subject not in by_number]
    if missing:
        raise ValueError(f"Requested target subjects are absent after loading: {missing}.")
    return [by_number[subject] for subject in requested_subjects]


def make_model(
    model_cfg: dict[str, Any],
    *,
    num_classes: int,
    channels: int,
    samples: int,
    dropout_rate: float,
    device: torch.device,
) -> EEGNet:
    return EEGNet(
        num_classes=num_classes,
        channels=channels,
        samples=samples,
        kernel_length=int(model_cfg.get("kernel_length", 40)),
        separable_kernel_length=int(model_cfg.get("separable_kernel_length", 16)),
        f1=int(model_cfg.get("F1", 8)),
        depth_multiplier=int(model_cfg.get("D", 2)),
        f2=int(model_cfg.get("F2", 16)),
        dropout_rate=dropout_rate,
        pool1_length=int(model_cfg.get("pool1_length", 4)),
        pool2_length=int(model_cfg.get("pool2_length", 8)),
        depthwise_max_norm=float(model_cfg.get("depthwise_max_norm", 1.0)),
        classifier_max_norm=float(model_cfg.get("classifier_max_norm", 0.25)),
    ).to(device)


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for metric in REPORT_METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
        }
    return aggregate


def split_data_config_for_protocol(
    data_cfg: dict[str, Any],
    protocol: str,
) -> tuple[dict[str, Any], list[int] | None]:
    target_subjects = parse_subjects(data_cfg.get("subjects"))
    load_cfg = deepcopy(data_cfg)
    if protocol == "paper_global_ss_tl":
        # The paper's subject range is a loading cohort. Its author-style
        # loader removes the four excluded subjects before defining CV folds.
        return load_cfg, None
    if protocol == "transfer":
        pretrain_subjects = parse_subjects(data_cfg.get("pretrain_subjects"))
        if pretrain_subjects is None:
            raise ValueError("EEGNet transfer requires data.pretrain_subjects.")
        load_cfg["subjects"] = sorted(set(pretrain_subjects) | set(target_subjects or []))
    return load_cfg, target_subjects


def preprocessing_parameters(
    data_cfg: dict[str, Any],
) -> tuple[float, bool]:
    input_scale = float(data_cfg.get("eegnet_input_scale", 1.0e6))
    standardize = parse_bool(data_cfg.get("eegnet_standardize", True), default=True)
    return input_scale, standardize


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    input_scale: float,
    standardize: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    mean, std = fit_channel_standardizer(
        x,
        train_idx,
        input_scale=input_scale,
        enabled=standardize,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "input_scale": input_scale,
        "mean": mean,
        "std": std,
        "pin_memory": pin_memory,
    }
    train_loader = make_loader(x, y, train_idx, shuffle=True, **common)
    train_eval_loader = make_loader(x, y, train_idx, shuffle=False, **common)
    test_loader = make_loader(x, y, test_idx, shuffle=False, **common)
    return train_loader, train_eval_loader, test_loader, mean, std


def make_loaders_with_standardizer(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    input_scale: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "input_scale": input_scale,
        "mean": mean,
        "std": std,
        "pin_memory": pin_memory,
    }
    return (
        make_loader(x, y, train_idx, shuffle=True, **common),
        make_loader(x, y, train_idx, shuffle=False, **common),
        make_loader(x, y, test_idx, shuffle=False, **common),
    )


def run_subject_wise(
    *,
    x: np.ndarray,
    y: np.ndarray,
    subject_labels: np.ndarray,
    class_names: list[str],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed = int(training_cfg.get("seed", 42))
    n_splits = int(training_cfg.get("subject_wise_n_splits", 1))
    train_size = float(training_cfg.get("train_size", 0.7))
    test_size = float(training_cfg.get("test_size", 0.3))
    epochs = int(training_cfg.get("epochs", 500))
    learning_rate = float(training_cfg.get("learning_rate", 1.0e-3))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = parse_bool(training_cfg.get("pin_memory", True), default=True)
    log_every = max(1, int(training_cfg.get("log_every", 10)))
    dropout_rate = float(model_cfg.get("within_subject_dropout", 0.5))
    input_scale, standardize = preprocessing_parameters(data_cfg)
    splits = subject_wise_splits(
        y,
        subject_labels,
        n_splits=n_splits,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    save_json(
        run_dir / "splits.json",
        [
            {
                "subject": subject,
                "fold": fold if n_splits > 1 else None,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
            }
            for subject, fold, train_idx, _validation_idx, test_idx in splits
        ],
    )

    rows: list[dict[str, Any]] = []
    criterion = nn.CrossEntropyLoss()
    for position, (subject, fold, train_idx, _validation_idx, test_idx) in enumerate(
        splits,
        start=1,
    ):
        set_seed(seed + position - 1)
        fold_name = f"fold_{fold:02d}" if n_splits > 1 else "trial_random_split"
        fold_dir = run_dir / subject / fold_name
        fold_dir.mkdir(parents=True, exist_ok=False)
        model = make_model(
            model_cfg,
            num_classes=len(class_names),
            channels=x.shape[1],
            samples=x.shape[2],
            dropout_rate=dropout_rate,
            device=device,
        )
        train_loader, train_eval_loader, test_loader, mean, std = make_loaders(
            x,
            y,
            train_idx,
            test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            input_scale=input_scale,
            standardize=standardize,
        )
        label = f"EEGNet {subject} {fold_name}"
        history = train_fixed_epochs(
            model,
            train_loader,
            train_eval_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            log_every=log_every,
            label=label,
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        row = {
            "subject": subject,
            "fold": fold if n_splits > 1 else None,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "epochs_completed": epochs,
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "cohen_kappa": test_metrics["cohen_kappa"],
            "_y_true": test_metrics["y_true"].tolist(),
            "_y_pred": test_metrics["y_pred"].tolist(),
            "_subject_labels": subject_labels[test_idx].astype(str).tolist(),
        }
        rows.append(row)
        write_csv(fold_dir / "history.csv", history)
        save_json(
            fold_dir / "test_metrics.json",
            {key: value for key, value in row.items() if not key.startswith("_")},
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "channel_mean": mean,
                "channel_std": std,
                "input_scale": input_scale,
                "train_indices": train_idx,
                "test_indices": test_idx,
            },
            fold_dir / "final_model.pt",
        )
        print(
            f"[{label}] test_accuracy={row['accuracy']:.4f} "
            f"balanced_accuracy={row['balanced_accuracy']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} kappa={row['cohen_kappa']:.4f}",
            flush=True,
        )

    subject_rows = summarize_subject_fold_metrics(rows, REPORT_METRICS)
    evaluation = {
        "strategy": (
            "subject_wise_stratified_kfold"
            if n_splits > 1
            else "subject_wise_stratified_trial_train_test"
        ),
        "n_splits": n_splits if n_splits > 1 else None,
        "train_size": None if n_splits > 1 else train_size,
        "test_size": 1.0 / n_splits if n_splits > 1 else test_size,
        "uses_validation": False,
        "seed": seed,
    }
    return rows, subject_rows, evaluation


def run_transfer(
    *,
    x: np.ndarray,
    y: np.ndarray,
    subject_labels: np.ndarray,
    target_labels: list[str],
    class_names: list[str],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed = int(training_cfg.get("seed", 42))
    target_train_size = float(training_cfg.get("train_size", 0.7))
    target_test_size = float(training_cfg.get("test_size", 0.3))
    pretrain_train_size = float(training_cfg.get("pretrain_train_size", 0.7))
    pretrain_validation_size = float(
        training_cfg.get("pretrain_validation_size", 0.0)
    )
    pretrain_test_size = float(
        training_cfg.get("pretrain_test_size", 0.3 - pretrain_validation_size)
    )
    pretrain_epochs = int(training_cfg.get("pretrain_epochs", 500))
    fine_tune_epochs = int(training_cfg.get("fine_tune_epochs", 100))
    pretrain_lr = float(training_cfg.get("pretrain_learning_rate", 1.0e-3))
    fine_tune_lr = float(training_cfg.get("fine_tune_learning_rate", 1.0e-4))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = parse_bool(training_cfg.get("pin_memory", True), default=True)
    log_every = max(1, int(training_cfg.get("log_every", 10)))
    freeze_features = parse_bool(
        training_cfg.get("freeze_feature_extractor", False), default=False
    )
    dropout_rate = float(model_cfg.get("cross_subject_dropout", 0.25))
    input_scale, standardize = preprocessing_parameters(data_cfg)
    criterion = nn.CrossEntropyLoss()
    rows: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []

    for target in target_labels:
        target_number = int(target.upper().lstrip("S"))
        pretrain_seed = seed + target_number
        fine_tune_seed = seed + target_number * 100
        set_seed(pretrain_seed)
        target_idx = np.flatnonzero(subject_labels == target).astype(np.int64)
        other_idx = np.flatnonzero(subject_labels != target).astype(np.int64)
        if np.any(subject_labels[other_idx] == target):
            raise RuntimeError(f"Target subject {target} leaked into pretraining.")
        if pretrain_validation_size > 0.0:
            pretrain_idx, pretrain_validation_idx, pretrain_test_idx = (
                stratified_train_validation_test_split(
                    y,
                    other_idx,
                    train_size=pretrain_train_size,
                    validation_size=pretrain_validation_size,
                    test_size=pretrain_test_size,
                    seed=pretrain_seed,
                )
            )
        else:
            pretrain_idx, pretrain_test_idx = stratified_train_test_split(
                y,
                other_idx,
                train_size=pretrain_train_size,
                test_size=pretrain_test_size,
                seed=pretrain_seed,
            )
            pretrain_validation_idx = np.empty(0, dtype=np.int64)
        fine_tune_idx, target_test_idx = stratified_train_test_split(
            y,
            target_idx,
            train_size=target_train_size,
            test_size=target_test_size,
            seed=fine_tune_seed,
        )
        if np.intersect1d(other_idx, target_idx).size:
            raise RuntimeError(f"Target subject {target} overlaps pretraining subjects.")

        target_dir = run_dir / target
        pretrain_dir = target_dir / "pretrain"
        fine_tune_dir = target_dir / "fine_tune"
        pretrain_dir.mkdir(parents=True, exist_ok=False)
        fine_tune_dir.mkdir(parents=True, exist_ok=False)
        model = make_model(
            model_cfg,
            num_classes=len(class_names),
            channels=x.shape[1],
            samples=x.shape[2],
            dropout_rate=dropout_rate,
            device=device,
        )

        pretrain_loaders = make_loaders(
            x,
            y,
            pretrain_idx,
            pretrain_test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            input_scale=input_scale,
            standardize=standardize,
        )
        pretrain_loader, pretrain_eval_loader, pretrain_test_loader, pre_mean, pre_std = (
            pretrain_loaders
        )
        pretrain_validation_loader = (
            make_loader(
                x,
                y,
                pretrain_validation_idx,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                input_scale=input_scale,
                mean=pre_mean,
                std=pre_std,
                pin_memory=pin_memory,
            )
            if len(pretrain_validation_idx)
            else None
        )
        pretrain_history = train_fixed_epochs(
            model,
            pretrain_loader,
            pretrain_eval_loader,
            epochs=pretrain_epochs,
            learning_rate=pretrain_lr,
            device=device,
            log_every=log_every,
            label=f"EEGNet pretrain excluding {target}",
        )
        pretrain_metrics = evaluate(model, pretrain_test_loader, criterion, device)
        pretrain_validation_metrics = (
            evaluate(model, pretrain_validation_loader, criterion, device)
            if pretrain_validation_loader is not None
            else None
        )
        write_csv(pretrain_dir / "history.csv", pretrain_history)
        save_json(
            pretrain_dir / "test_metrics.json",
            {
                "n_samples": int(len(pretrain_test_idx)),
                "n_validation_samples": int(len(pretrain_validation_idx)),
                "validation_accuracy": (
                    pretrain_validation_metrics["accuracy"]
                    if pretrain_validation_metrics is not None
                    else None
                ),
                "loss": pretrain_metrics["loss"],
                "accuracy": pretrain_metrics["accuracy"],
                "balanced_accuracy": pretrain_metrics["balanced_accuracy"],
                "macro_f1": pretrain_metrics["macro_f1"],
                "cohen_kappa": pretrain_metrics["cohen_kappa"],
            },
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "excluded_target_subject": target,
                "pretrain_subjects": sorted(np.unique(subject_labels[other_idx]).tolist()),
                "channel_mean": pre_mean,
                "channel_std": pre_std,
                "input_scale": input_scale,
            },
            pretrain_dir / "pretrained_model.pt",
        )

        if freeze_features:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        set_seed(fine_tune_seed)
        fine_tune_loaders = make_loaders(
            x,
            y,
            fine_tune_idx,
            target_test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            input_scale=input_scale,
            standardize=standardize,
        )
        fine_tune_loader, fine_tune_eval_loader, target_test_loader, ft_mean, ft_std = (
            fine_tune_loaders
        )
        fine_tune_history = train_fixed_epochs(
            model,
            fine_tune_loader,
            fine_tune_eval_loader,
            epochs=fine_tune_epochs,
            learning_rate=fine_tune_lr,
            device=device,
            log_every=log_every,
            label=f"EEGNet fine-tune {target}",
        )
        target_metrics = evaluate(model, target_test_loader, criterion, device)
        write_csv(fine_tune_dir / "history.csv", fine_tune_history)
        row = {
            "subject": target,
            "n_pretrain": int(len(pretrain_idx)),
            "n_pretrain_validation": int(len(pretrain_validation_idx)),
            "n_pretrain_test": int(len(pretrain_test_idx)),
            "n_train": int(len(fine_tune_idx)),
            "n_test": int(len(target_test_idx)),
            "pretrain_accuracy": pretrain_metrics["accuracy"],
            "epochs_completed": fine_tune_epochs,
            "loss": target_metrics["loss"],
            "accuracy": target_metrics["accuracy"],
            "balanced_accuracy": target_metrics["balanced_accuracy"],
            "macro_f1": target_metrics["macro_f1"],
            "cohen_kappa": target_metrics["cohen_kappa"],
            "_y_true": target_metrics["y_true"].tolist(),
            "_y_pred": target_metrics["y_pred"].tolist(),
            "_subject_labels": subject_labels[target_test_idx].astype(str).tolist(),
        }
        rows.append(row)
        save_json(
            fine_tune_dir / "test_metrics.json",
            {key: value for key, value in row.items() if not key.startswith("_")},
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "target_subject": target,
                "freeze_feature_extractor": freeze_features,
                "channel_mean": ft_mean,
                "channel_std": ft_std,
                "input_scale": input_scale,
                "fine_tune_indices": fine_tune_idx,
                "test_indices": target_test_idx,
            },
            fine_tune_dir / "fine_tuned_model.pt",
        )
        split_records.append(
            {
                "target_subject": target,
                "pretrain_subjects": sorted(np.unique(subject_labels[other_idx]).tolist()),
                "pretrain_indices": pretrain_idx.tolist(),
                "pretrain_validation_indices": pretrain_validation_idx.tolist(),
                "pretrain_test_indices": pretrain_test_idx.tolist(),
                "fine_tune_indices": fine_tune_idx.tolist(),
                "test_indices": target_test_idx.tolist(),
            }
        )
        print(
            f"[EEGNet transfer {target}] pretrain_test_accuracy="
            f"{pretrain_metrics['accuracy']:.4f} target_test_accuracy="
            f"{row['accuracy']:.4f} target_macro_f1={row['macro_f1']:.4f} "
            f"target_kappa={row['cohen_kappa']:.4f}",
            flush=True,
        )

    save_json(run_dir / "splits.json", split_records)
    subject_rows = summarize_subject_fold_metrics(rows, REPORT_METRICS)
    evaluation = {
        "strategy": "other_subject_pretraining_then_target_subject_fine_tuning",
        "pretrain_train_size": pretrain_train_size,
        "pretrain_validation_size": pretrain_validation_size,
        "pretrain_test_size": pretrain_test_size,
        "target_train_size": target_train_size,
        "target_test_size": target_test_size,
        "uses_validation": pretrain_validation_size > 0.0,
        "target_excluded_from_pretraining": True,
        "freeze_feature_extractor": freeze_features,
        "seed": seed,
    }
    return rows, subject_rows, evaluation


def run_paper_global_ss_tl(
    *,
    x: np.ndarray,
    y: np.ndarray,
    subject_labels: np.ndarray,
    class_names: list[str],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Reproduce the PhysioNet paper's 5-fold global CV and 4-fold SS-TL."""

    from sklearn.model_selection import StratifiedKFold

    seed = int(training_cfg.get("seed", 42))
    global_n_splits = int(training_cfg.get("global_n_splits", 5))
    subject_n_splits = int(training_cfg.get("ss_tl_n_splits", 4))
    global_epochs = int(training_cfg.get("global_epochs", 100))
    global_lr = float(training_cfg.get("global_learning_rate", 1.0e-2))
    global_lr_schedule = training_cfg.get(
        "global_learning_rate_schedule",
        "0:0.01,20:0.001,50:0.0001",
    )
    fine_tune_epochs = int(training_cfg.get("ss_tl_epochs", 5))
    fine_tune_lr = float(training_cfg.get("ss_tl_learning_rate", 1.0e-3))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = parse_bool(training_cfg.get("pin_memory", True), default=True)
    log_every = max(1, int(training_cfg.get("log_every", 10)))
    save_ss_models = parse_bool(
        training_cfg.get("save_ss_tl_checkpoints", False),
        default=False,
    )
    dropout_rate = float(model_cfg.get("cross_subject_dropout", 0.2))
    input_scale, standardize = preprocessing_parameters(data_cfg)
    criterion = nn.CrossEntropyLoss()

    global_rows: list[dict[str, Any]] = []
    ss_rows: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    folds = global_subject_folds(subject_labels, n_splits=global_n_splits)

    for fold, global_train_idx, global_test_idx, train_subjects, test_subjects in folds:
        set_seed(seed * fold)
        fold_dir = run_dir / f"global_fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        global_model = make_model(
            model_cfg,
            num_classes=len(class_names),
            channels=x.shape[1],
            samples=x.shape[2],
            dropout_rate=dropout_rate,
            device=device,
        )
        global_loaders = make_loaders(
            x,
            y,
            global_train_idx,
            global_test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            input_scale=input_scale,
            standardize=standardize,
        )
        (
            global_train_loader,
            global_train_eval_loader,
            global_test_loader,
            global_mean,
            global_std,
        ) = global_loaders
        global_history = train_fixed_epochs(
            global_model,
            global_train_loader,
            global_train_eval_loader,
            epochs=global_epochs,
            learning_rate=global_lr,
            learning_rate_schedule=global_lr_schedule,
            device=device,
            log_every=log_every,
            label=f"EEGNet paper global fold {fold}",
        )
        global_metrics = evaluate(global_model, global_test_loader, criterion, device)
        global_row = {
            "subject": f"global_fold_{fold:02d}",
            "fold": fold,
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subjects),
            "n_train": int(len(global_train_idx)),
            "n_test": int(len(global_test_idx)),
            "epochs_completed": global_epochs,
            "loss": global_metrics["loss"],
            "accuracy": global_metrics["accuracy"],
            "balanced_accuracy": global_metrics["balanced_accuracy"],
            "macro_f1": global_metrics["macro_f1"],
            "cohen_kappa": global_metrics["cohen_kappa"],
            "_y_true": global_metrics["y_true"].tolist(),
            "_y_pred": global_metrics["y_pred"].tolist(),
            "_subject_labels": subject_labels[global_test_idx].astype(str).tolist(),
        }
        global_rows.append(global_row)
        write_csv(fold_dir / "history.csv", global_history)
        save_json(
            fold_dir / "test_metrics.json",
            {key: value for key, value in global_row.items() if not key.startswith("_")},
        )
        global_state = deepcopy(global_model.state_dict())
        torch.save(
            {
                "model_state_dict": global_state,
                "class_names": class_names,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "channel_mean": global_mean,
                "channel_std": global_std,
                "input_scale": input_scale,
            },
            fold_dir / "global_model.pt",
        )

        fold_record = {
            "global_fold": fold,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "global_train_indices": global_train_idx.tolist(),
            "global_test_indices": global_test_idx.tolist(),
            "subjects": [],
        }
        for subject_position, subject in enumerate(test_subjects):
            subject_idx = np.flatnonzero(subject_labels == subject).astype(np.int64)
            subject_splitter = StratifiedKFold(
                n_splits=subject_n_splits,
                shuffle=True,
                random_state=42,
            )
            subject_record = {"subject": subject, "folds": []}
            for subject_fold, (local_train, local_test) in enumerate(
                subject_splitter.split(subject_idx, y[subject_idx]),
                start=1,
            ):
                fine_tune_idx = subject_idx[local_train]
                target_test_idx = subject_idx[local_test]
                set_seed(seed + fold * 10_000 + subject_position * 100 + subject_fold)
                model = make_model(
                    model_cfg,
                    num_classes=len(class_names),
                    channels=x.shape[1],
                    samples=x.shape[2],
                    dropout_rate=dropout_rate,
                    device=device,
                )
                model.load_state_dict(global_state)
                train_loader, train_eval_loader, test_loader = (
                    make_loaders_with_standardizer(
                        x,
                        y,
                        fine_tune_idx,
                        target_test_idx,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        input_scale=input_scale,
                        mean=global_mean,
                        std=global_std,
                    )
                )
                before_metrics = evaluate(model, test_loader, criterion, device)
                label = f"EEGNet paper SS-TL {subject} fold {subject_fold}"
                history = train_fixed_epochs(
                    model,
                    train_loader,
                    train_eval_loader,
                    epochs=fine_tune_epochs,
                    learning_rate=fine_tune_lr,
                    device=device,
                    log_every=log_every,
                    label=label,
                )
                metrics = evaluate(model, test_loader, criterion, device)
                subject_fold_dir = (
                    fold_dir / "ss_tl" / subject / f"fold_{subject_fold:02d}"
                )
                subject_fold_dir.mkdir(parents=True, exist_ok=False)
                row = {
                    "subject": subject,
                    "global_fold": fold,
                    "subject_fold": subject_fold,
                    "n_global_train": int(len(global_train_idx)),
                    "n_train": int(len(fine_tune_idx)),
                    "n_test": int(len(target_test_idx)),
                    "global_accuracy": before_metrics["accuracy"],
                    "epochs_completed": fine_tune_epochs,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "cohen_kappa": metrics["cohen_kappa"],
                    "_y_true": metrics["y_true"].tolist(),
                    "_y_pred": metrics["y_pred"].tolist(),
                    "_subject_labels": subject_labels[target_test_idx]
                    .astype(str)
                    .tolist(),
                }
                ss_rows.append(row)
                write_csv(subject_fold_dir / "history.csv", history)
                save_json(
                    subject_fold_dir / "test_metrics.json",
                    {key: value for key, value in row.items() if not key.startswith("_")},
                )
                if save_ss_models:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "class_names": class_names,
                            "global_fold": fold,
                            "target_subject": subject,
                            "subject_fold": subject_fold,
                            "fine_tune_indices": fine_tune_idx,
                            "test_indices": target_test_idx,
                        },
                        subject_fold_dir / "fine_tuned_model.pt",
                    )
                subject_record["folds"].append(
                    {
                        "subject_fold": subject_fold,
                        "fine_tune_indices": fine_tune_idx.tolist(),
                        "test_indices": target_test_idx.tolist(),
                    }
                )
            fold_record["subjects"].append(subject_record)

        split_records.append(fold_record)
        print(
            f"[EEGNet paper global fold {fold}] test_accuracy="
            f"{global_row['accuracy']:.4f}",
            flush=True,
        )

    save_json(run_dir / "splits.json", split_records)
    global_subject_rows = summarize_subject_fold_metrics(global_rows, REPORT_METRICS)
    ss_subject_rows = summarize_subject_fold_metrics(ss_rows, REPORT_METRICS)
    evaluation = {
        "strategy": "paper_5_fold_global_subject_cv_then_4_fold_subject_ss_tl",
        "global_n_splits": global_n_splits,
        "ss_tl_n_splits": subject_n_splits,
        "global_subject_order_shuffled": False,
        "ss_tl_trials_stratified_and_shuffled": True,
        "global_aggregates": aggregate_rows(global_rows),
        "ss_tl_aggregates": aggregate_rows(ss_rows),
        "uses_validation": True,
        "test_used_during_training": False,
        "seed": seed,
    }
    return ss_rows, ss_subject_rows, evaluation, global_rows, global_subject_rows


def run_experiment(
    run_index: int,
    experiment_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    *,
    base_output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]
    protocol = normalize_protocol(training_cfg.get("protocol", "subject_wise"))
    load_cfg, requested_targets = split_data_config_for_protocol(data_cfg, protocol)
    author_preprocessing = parse_bool(
        data_cfg.get("eegnet_author_preprocessing", False),
        default=False,
    )
    if author_preprocessing:
        x, y, subject_labels, class_names = load_eegnet_author_data(load_cfg)
        filter_bank: list[list[float]] = []
    else:
        loaded_x, y, subject_labels, class_names, filter_bank = (
            load_segmented_epochs_like_train(load_cfg)
        )
        x = extract_single_trial_eeg(loaded_x)
    target_labels = resolve_target_labels(requested_targets, subject_labels)
    effective_cfg = deepcopy(experiment_cfg)
    effective_cfg["model"] = deepcopy(model_cfg)
    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(effective_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", effective_cfg)
    print(
        f"\n[EEGNet run {run_index}] protocol={protocol} x={x.shape} "
        f"classes={class_names} targets={target_labels} device={device}",
        flush=True,
    )

    global_rows: list[dict[str, Any]] = []
    global_subject_rows: list[dict[str, Any]] = []
    if protocol == "subject_wise":
        rows, subject_rows, evaluation = run_subject_wise(
            x=x,
            y=y,
            subject_labels=subject_labels,
            class_names=class_names,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            run_dir=run_dir,
            device=device,
        )
    elif protocol == "transfer":
        rows, subject_rows, evaluation = run_transfer(
            x=x,
            y=y,
            subject_labels=subject_labels,
            target_labels=target_labels,
            class_names=class_names,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            run_dir=run_dir,
            device=device,
        )
    else:
        (
            rows,
            subject_rows,
            evaluation,
            global_rows,
            global_subject_rows,
        ) = run_paper_global_ss_tl(
            x=x,
            y=y,
            subject_labels=subject_labels,
            class_names=class_names,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            run_dir=run_dir,
            device=device,
        )

    visible_rows = public_rows(rows)
    write_csv(run_dir / "results.csv", visible_rows)
    write_csv(run_dir / "per_subject_results.csv", visible_rows)
    write_csv(run_dir / "per_subject_summary.csv", subject_rows)
    if global_rows:
        write_csv(run_dir / "global_results.csv", public_rows(global_rows))
        write_csv(run_dir / "global_per_subject_summary.csv", global_subject_rows)
    aggregates = aggregate_rows(rows)
    summary = {
        "baseline": "eegnet",
        "protocol": protocol,
        "source_repository": (
            PHYSIONET_REPOSITORY
            if protocol == "paper_global_ss_tl"
            else OFFICIAL_REPOSITORY
        ),
        "paper": PHYSIONET_PAPER if protocol == "paper_global_ss_tl" else OFFICIAL_PAPER,
        "base_eegnet_repository": OFFICIAL_REPOSITORY,
        "architecture": "EEGNet-8,2",
        "official_architecture_parameters": {
            "F1": int(model_cfg.get("F1", 8)),
            "D": int(model_cfg.get("D", 2)),
            "F2": int(model_cfg.get("F2", 16)),
            "kernel_length": int(model_cfg.get("kernel_length", 40)),
            "kernel_duration_seconds": int(model_cfg.get("kernel_length", 40))
            / float(data_cfg.get("sfreq", 160)),
            "separable_kernel_length": int(model_cfg.get("separable_kernel_length", 16)),
            "pool1_length": int(model_cfg.get("pool1_length", 4)),
            "pool2_length": int(model_cfg.get("pool2_length", 8)),
            "within_subject_dropout": float(model_cfg.get("within_subject_dropout", 0.5)),
            "cross_subject_dropout": float(model_cfg.get("cross_subject_dropout", 0.25)),
            "depthwise_max_norm": float(model_cfg.get("depthwise_max_norm", 1.0)),
            "classifier_max_norm": float(model_cfg.get("classifier_max_norm", 0.25)),
        },
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-7,
            "weight_decay": 0.0,
        },
        "input_shape": list(x.shape),
        "filter_bank": filter_bank,
        "class_names": class_names,
        "config": effective_cfg,
        "evaluation": evaluation,
        "folds": visible_rows,
        "subjects": subject_rows,
        "global_folds": public_rows(global_rows),
        "global_subjects": global_subject_rows,
        "aggregates": aggregates,
        "test_used_during_training": False,
    }
    save_json(run_dir / "summary.json", summary)
    print(
        f"[EEGNet run {run_index}] saved {run_dir} | "
        f"mean_acc={aggregates['accuracy']['mean']:.4f} "
        f"mean_mf1={aggregates['macro_f1']['mean']:.4f} "
        f"mean_kappa={aggregates['cohen_kappa']['mean']:.4f}",
        flush=True,
    )
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "protocol": protocol,
        "test_accuracy_mean": aggregates["accuracy"]["mean"],
        "test_macro_f1_mean": aggregates["macro_f1"]["mean"],
        "test_cohen_kappa_mean": aggregates["cohen_kappa"]["mean"],
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    model_cfg = dict(config.get("model", {}))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or config.get("output", {}).get(
            "dir",
            "experiments/results/eegnet_baseline",
        )
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    base_output_dir = output_dir / timestamp
    all_metrics = [
        run_experiment(
            index,
            experiment,
            model_cfg,
            base_output_dir=base_output_dir,
            device=device,
        )
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All EEGNet runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
