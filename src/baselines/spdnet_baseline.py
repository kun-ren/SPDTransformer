from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.BiMap import BiMap
from src.models.SPDAttention import spd_log


DEFAULT_FILTER_BANK = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
)
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "experiments" / "splits" / "physionet_mi_seed42.json"

from src.training.shared_split import load_or_create_split_indices


class SPDDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class ReEig(nn.Module):
    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 0.5 * (x + x.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(x)
        eigvals = eigvals.clamp_min(self.eps)
        y = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        return 0.5 * (y + y.transpose(-1, -2))


class SPDNetClassifier(nn.Module):
    """
    Classic SPDNet-style baseline:
        BiMap -> ReEig -> BiMap -> ReEig -> LogEig -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_spd_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.bimap1 = BiMap(input_dim, hidden_dim, eps=eps)
        self.reeig1 = ReEig(eps=eps)
        self.bimap2 = BiMap(hidden_dim, output_spd_dim, eps=eps)
        self.reeig2 = ReEig(eps=eps)
        self.feature_dim = output_spd_dim * (output_spd_dim + 1) // 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bimap1(x)
        x = self.reeig1(x)
        x = self.bimap2(x)
        x = self.reeig2(x)
        log_x = spd_log(x)
        features = self.upper_triangular_vectorize(log_x)
        return self.classifier(features)

    @staticmethod
    def upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        rows, cols = torch.triu_indices(dim, dim, device=x.device)
        return x[..., rows, cols]


def parse_filter_bank(raw: str) -> list[tuple[float, float]]:
    bands = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        low_raw, high_raw = part.split("-", 1)
        bands.append((float(low_raw), float(high_raw)))
    if not bands:
        raise argparse.ArgumentTypeError("filter bank cannot be empty")
    return bands


def matrix_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = 0.5 * (x + np.swapaxes(x, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(x)
    eigvals = np.clip(eigvals, eps, None)
    log_eigvals = np.log(eigvals)
    return (eigvecs * log_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)


def matrix_exp(x: np.ndarray) -> np.ndarray:
    x = 0.5 * (x + np.swapaxes(x, -1, -2))
    eigvals, eigvecs = np.linalg.eigh(x)
    exp_eigvals = np.exp(eigvals)
    y = (eigvecs * exp_eigvals[..., None, :]) @ np.swapaxes(eigvecs, -1, -2)
    return 0.5 * (y + np.swapaxes(y, -1, -2))


def log_euclidean_mean(covs: np.ndarray) -> np.ndarray:
    log_covs = matrix_log(covs)
    pooled_log = log_covs.mean(axis=(1, 2))
    return matrix_exp(pooled_log)


def build_spd_features(
    X: np.ndarray,
    filter_bank: list[tuple[float, float]],
    sfreq: float,
    segment_duration: float,
    stride_duration: float | None,
    estimator: str,
    eps: float,
) -> np.ndarray:
    from pyriemann.estimation import Covariances
    from src.datasets.PhysioNetMI_preprocess import (
        bandpass_filter,
        regularize_spd,
        segment_epochs,
        trace_normalize,
    )

    band_covariances = []
    for low_freq, high_freq in filter_bank:
        filtered = bandpass_filter(
            X,
            sfreq=sfreq,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        segments = segment_epochs(
            filtered,
            sfreq=sfreq,
            segment_duration=segment_duration,
            stride_duration=stride_duration,
        )
        n_epochs, n_segments, n_channels, n_samples = segments.shape
        covs = Covariances(estimator=estimator).fit_transform(
            segments.reshape(n_epochs * n_segments, n_channels, n_samples)
        )
        covs = covs.reshape(n_epochs, n_segments, n_channels, n_channels)
        covs = trace_normalize(covs, eps=eps)
        covs = regularize_spd(covs, eps=eps)
        band_covariances.append(covs.astype(np.float64))

    covs = np.stack(band_covariances, axis=1)
    return log_euclidean_mean(covs).astype(np.float32)


def resolve_split_file(split_file: str | None) -> Path | None:
    if split_file in {None, ""}:
        return None
    path = Path(split_file)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SPDDataset(X[indices], y[indices]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(y_true)
    return metrics


def save_results(output_dir: Path, history: list[dict], summary: dict, model_state: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    torch.save(model_state, output_dir / "best_model.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPDNet baseline on raw PhysioNetMI epochs.")
    parser.add_argument("--root-dir", default="data/MNE-eegbci-data/files/eegmmidb/1.0.0")
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--sfreq", type=float, default=160.0)
    parser.add_argument(
        "--filter-bank",
        type=parse_filter_bank,
        default=list(DEFAULT_FILTER_BANK),
        help="Comma-separated bands, for example: 8-12,12-16,16-20,20-24,24-28,28-32",
    )
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--stride-duration", type=float, default=0.5)
    parser.add_argument("--estimator", default="lwf")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--output-spd-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="experiments/results/spdnet_baseline")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from src.datasets.PhysioNetMI_preprocess import build_dataset, encode_labels

    dataset = build_dataset(args.root_dir, tmin=args.tmin, tmax=args.tmax)
    X = dataset["X"]
    y, class_names = encode_labels(dataset["y"])
    print(f"Raw X shape: {X.shape}")
    print(f"Classes: {class_names.tolist()}")

    X_spd = build_spd_features(
        X=X,
        filter_bank=args.filter_bank,
        sfreq=args.sfreq,
        segment_duration=args.segment_duration,
        stride_duration=args.stride_duration,
        estimator=args.estimator,
        eps=args.eps,
    )
    print(f"Pooled SPD shape: {X_spd.shape}")

    train_idx, val_idx, test_idx = load_or_create_split_indices(
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        split_file=resolve_split_file(args.split_file),
    )
    train_loader = make_loader(X_spd, y, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(X_spd, y, val_idx, args.batch_size, shuffle=False)
    test_loader = make_loader(X_spd, y, test_idx, args.batch_size, shuffle=False)

    model = SPDNetClassifier(
        input_dim=X_spd.shape[-1],
        hidden_dim=args.hidden_dim,
        output_spd_dim=args.output_spd_dim,
        num_classes=len(class_names),
        dropout=args.dropout,
        eps=args.eps,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict] = []
    best_state = None
    best_val_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names.tolist(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val_macro_f1,
            }

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"mf1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"mf1={val_metrics['macro_f1']:.4f}"
        )

    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device)

    summary = {
        "baseline": "spdnet",
        "class_names": class_names.tolist(),
        "raw_shape": list(X.shape),
        "spd_shape": list(X_spd.shape),
        "filter_bank": [list(band) for band in args.filter_bank],
        "segment_duration": args.segment_duration,
        "stride_duration": args.stride_duration,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "test": test_metrics,
    }
    best_state["test_metrics"] = test_metrics

    output_dir = PROJECT_ROOT / args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(output_dir, history, summary, best_state)

    print("\nTest")
    print(test_metrics)
    print(f"\nSaved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
