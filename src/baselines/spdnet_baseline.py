from __future__ import annotations

import argparse
import copy
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
    load_spd_like_train,
    load_yaml,
    log_euclidean_token_mean,
    save_json,
)


class SPDTrialDataset(Dataset):
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


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def reconstruct_symmetric(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
) -> torch.Tensor:
    y = (
        eigenvectors
        * eigenvalues.unsqueeze(-2)
    ) @ eigenvectors.transpose(-1, -2)
    return symmetrize(y)


class SPDNetBiMap(nn.Module):
    """BiMap layer Y = W^T X W with column-orthonormal W."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if out_dim > in_dim:
            raise ValueError(
                f"SPDNet BiMap requires out_dim <= in_dim, got {out_dim}>{in_dim}."
            )
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.weight = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            random_matrix = torch.randn(
                self.in_dim,
                self.out_dim,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            q, _ = torch.linalg.qr(random_matrix, mode="reduced")
            self.weight.copy_(q[:, :self.out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("ia,...ij,jb->...ab", self.weight, x, self.weight)
        return symmetrize(y)

    def project_stiefel_(self) -> None:
        with torch.no_grad():
            q, r = torch.linalg.qr(self.weight, mode="reduced")
            signs = torch.sign(torch.diagonal(r))
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            self.weight.copy_(q * signs.unsqueeze(0))


class SPDNetReEig(nn.Module):
    def __init__(self, epsilon: float = 1e-4) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetrize(x))
        eigenvalues = eigenvalues.clamp_min(self.epsilon)
        return reconstruct_symmetric(eigenvalues, eigenvectors)


class SPDNetLogEig(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetrize(x))
        log_eigenvalues = eigenvalues.clamp_min(self.epsilon).log()
        return reconstruct_symmetric(log_eigenvalues, eigenvectors)


class SPDNetClassifier(nn.Module):
    """
    PyTorch SPDNet baseline:
        BiMap -> ReEig -> ... -> BiMap -> LogEig -> full-matrix FC.
    """

    def __init__(
        self,
        dims: list[int],
        num_classes: int,
        reig_epsilon: float = 1e-4,
        log_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("SPDNet dims must include input and at least one output dim.")
        if any(dim < 1 for dim in dims):
            raise ValueError(f"SPDNet dims must be positive, got {dims}.")
        if any(left < right for left, right in zip(dims, dims[1:])):
            raise ValueError(
                "SPDNet dims must be non-increasing for BiMap projection, "
                f"got {dims}."
            )

        layers: list[nn.Module] = []
        for layer_index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(SPDNetBiMap(in_dim, out_dim))
            is_last_bimap = layer_index == len(dims) - 2
            if is_last_bimap:
                layers.append(SPDNetLogEig(epsilon=log_epsilon))
            else:
                layers.append(SPDNetReEig(epsilon=reig_epsilon))

        self.dims = list(dims)
        self.features = nn.Sequential(*layers)
        final_dim = dims[-1]
        self.classifier = nn.Linear(final_dim * final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def project_stiefel_(self) -> None:
        for module in self.modules():
            if isinstance(module, SPDNetBiMap):
                module.project_stiefel_()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPDNet baseline using the same SPD preprocessing/split config."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dims",
        default=None,
        help=(
            "Comma-separated SPDNet dimensions. If the input dimension is omitted, "
            "it is prepended automatically. Example: 16,8 for 21->16->8."
        ),
    )
    parser.add_argument("--reig-epsilon", type=float, default=1e-4)
    parser.add_argument("--log-epsilon", type=float, default=1e-6)
    parser.add_argument("--disable-stiefel-projection", action="store_true")
    return parser


def resolve_precision(precision: Any) -> torch.dtype:
    precision = str(precision or "float64").lower()
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


def parse_dims(raw_dims: str | None, input_dim: int) -> list[int]:
    if raw_dims is None or not str(raw_dims).strip():
        if input_dim >= 16:
            return [input_dim, 16, 8]
        if input_dim >= 8:
            return [input_dim, 8, 4]
        return [input_dim, max(1, input_dim // 2)]

    dims = [int(part.strip()) for part in str(raw_dims).split(",") if part.strip()]
    if not dims:
        raise ValueError("--dims did not contain any integer dimensions.")
    if dims[0] != input_dim:
        dims = [input_dim, *dims]
    return dims


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    dtype: torch.dtype,
) -> DataLoader:
    return DataLoader(
        SPDTrialDataset(x, y, indices, dtype=dtype),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def build_optimizer(
    model: SPDNetClassifier,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    bimap_parameter_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, SPDNetBiMap)
    }
    bimap_parameters = []
    euclidean_parameters = []
    for parameter in model.parameters():
        if id(parameter) in bimap_parameter_ids:
            bimap_parameters.append(parameter)
        else:
            euclidean_parameters.append(parameter)

    parameter_groups = []
    if bimap_parameters:
        parameter_groups.append({"params": bimap_parameters, "weight_decay": 0.0})
    if euclidean_parameters:
        parameter_groups.append(
            {"params": euclidean_parameters, "weight_decay": weight_decay}
        )
    return torch.optim.Adam(parameter_groups, lr=learning_rate)


def train_one_epoch(
    model: SPDNetClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None,
    project_stiefel: bool,
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
            raise RuntimeError("Non-finite SPDNet training loss detected.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if project_stiefel:
            model.project_stiefel_()

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


def run_experiment(
    run_index: int,
    experiment_cfg: dict[str, Any],
    args: argparse.Namespace,
    base_output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]
    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    x_spd, y, subject_labels, class_names = load_spd_like_train(data_cfg)
    x_trial_spd = log_euclidean_token_mean(x_spd).astype(np.float32)
    train_idx, val_idx, test_idx = get_split_indices(
        y,
        training_cfg,
        subject_labels=subject_labels,
        data_cfg=data_cfg,
    )

    dtype = resolve_precision(training_cfg.get("precision", "float64"))
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 0))
    epochs = int(training_cfg.get("epochs", 50))
    learning_rate = float(
        training_cfg.get("spdnet_learning_rate", training_cfg.get("learning_rate", 1e-3))
    )
    weight_decay = float(
        training_cfg.get("spdnet_weight_decay", training_cfg.get("weight_decay", 0.0))
    )
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 0.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)

    dims = parse_dims(args.dims, input_dim=x_trial_spd.shape[-1])
    model = SPDNetClassifier(
        dims=dims,
        num_classes=len(class_names),
        reig_epsilon=args.reig_epsilon,
        log_epsilon=args.log_epsilon,
    ).to(device=device, dtype=dtype)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    project_stiefel = not args.disable_stiefel_projection

    train_loader = make_loader(
        x_trial_spd,
        y,
        train_idx,
        batch_size,
        num_workers,
        shuffle=True,
        dtype=dtype,
    )
    train_eval_loader = make_loader(
        x_trial_spd,
        y,
        train_idx,
        batch_size,
        num_workers,
        shuffle=False,
        dtype=dtype,
    )
    val_loader = make_loader(
        x_trial_spd,
        y,
        val_idx,
        batch_size,
        num_workers,
        shuffle=False,
        dtype=dtype,
    )
    test_loader = make_loader(
        x_trial_spd,
        y,
        test_idx,
        batch_size,
        num_workers,
        shuffle=False,
        dtype=dtype,
    )

    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(experiment_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", experiment_cfg)

    print(f"\n[SPDNet run {run_index}] {run_dir.name}")
    print(
        f"  x_spd={x_spd.shape} x_trial_spd={x_trial_spd.shape} "
        f"class_names={class_names}"
    )
    print(
        f"  dims={dims} epochs={epochs} batch_size={batch_size} "
        f"lr={learning_rate:g} weight_decay={weight_decay:g} "
        f"project_stiefel={project_stiefel}"
    )

    best_val_macro_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    history_rows = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_clip_norm,
            project_stiefel,
        )
        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history_rows.append(row)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"  epoch {epoch:03d}/{epochs} | "
            f"train loss={train_loss:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"mf1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"mf1={val_metrics['macro_f1']:.4f}"
        )

    if best_state_dict is None:
        raise RuntimeError("SPDNet did not produce a best checkpoint.")
    model.load_state_dict(best_state_dict)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "class_names": class_names,
            "config": experiment_cfg,
            "dims": dims,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
        run_dir / "best_model.pt",
    )
    write_csv(run_dir / "history.csv", history_rows)

    split_loaders = {
        "train": (train_idx, train_eval_loader),
        "val": (val_idx, val_loader),
        "test": (test_idx, test_loader),
    }
    result_rows = []
    for split_name, (split_idx, loader) in split_loaders.items():
        metrics = evaluate(model, loader, criterion, device)
        result_rows.append(
            {
                "split": split_name,
                "n_samples": int(len(split_idx)),
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    write_csv(run_dir / "results.csv", result_rows)

    summary = {
        "baseline": "spdnet",
        "source_repository": "https://github.com/zhiwu-huang/SPDNet",
        "paper": "A Riemannian Network for SPD Matrix Learning, AAAI 2017",
        "architecture": "BiMap-ReEig blocks followed by LogEig and linear FC",
        "token_pooling": "log_euclidean_mean_over_segment_and_frequency",
        "dims": dims,
        "reig_epsilon": args.reig_epsilon,
        "log_epsilon": args.log_epsilon,
        "project_stiefel": project_stiefel,
        "x_spd_shape": list(x_spd.shape),
        "x_trial_spd_shape": list(x_trial_spd.shape),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "splits": result_rows,
    }
    save_json(run_dir / "summary.json", summary)
    test_row = result_rows[-1]
    print(
        f"[SPDNet run {run_index}] done | best_epoch={best_epoch} "
        f"best_val_mf1={best_val_macro_f1:.4f} "
        f"test_acc={test_row['accuracy']:.4f} "
        f"test_mf1={test_row['macro_f1']:.4f}"
    )
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "test_accuracy": test_row["accuracy"],
        "test_macro_f1": test_row["macro_f1"],
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "dims": dims,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/spdnet_baseline",
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = [
        run_experiment(index, experiment, args, base_output_dir, device)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All SPDNet runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
