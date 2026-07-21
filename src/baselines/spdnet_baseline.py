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
    load_spd_like_train,
    load_yaml,
    parse_bool,
    save_json,
)


SPDNET_REPORT_METRICS = ("accuracy", "macro_f1", "cohen_kappa")


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

    def project_stiefel_gradients_(self) -> None:
        """Project Euclidean BiMap gradients onto the Stiefel tangent space."""
        with torch.no_grad():
            for module in self.modules():
                if not isinstance(module, SPDNetBiMap):
                    continue
                gradient = module.weight.grad
                if gradient is None:
                    continue
                wt_gradient = module.weight.transpose(0, 1) @ gradient
                tangent_gradient = gradient - module.weight @ symmetrize(wt_gradient)
                gradient.copy_(tangent_gradient)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SPDNet baseline using one covariance matrix per trial and "
            "stratified K-fold evaluation without a validation split."
        )
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
    parser.add_argument("--reig-epsilon", type=float, default=None)
    parser.add_argument("--log-epsilon", type=float, default=None)
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


def parse_dims(raw_dims: Any, input_dim: int) -> list[int]:
    if raw_dims is None or not str(raw_dims).strip():
        if input_dim >= 32:
            return [input_dim, input_dim // 2, input_dim // 4, input_dim // 8]
        if input_dim >= 16:
            return [input_dim, 16, 8, 4]
        if input_dim >= 8:
            return [input_dim, 8, 4]
        return [input_dim, max(1, input_dim // 2)]

    if isinstance(raw_dims, (list, tuple)):
        dims = [int(part) for part in raw_dims]
    else:
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
    optimizer_name: str = "sgd",
    momentum: float = 0.0,
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
    normalized_name = str(optimizer_name).strip().lower()
    if normalized_name == "sgd":
        return torch.optim.SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=momentum,
        )
    if normalized_name == "adam":
        return torch.optim.Adam(parameter_groups, lr=learning_rate)
    raise ValueError(
        f"Unsupported SPDNet optimizer {optimizer_name!r}; use 'sgd' or 'adam'."
    )


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
        if project_stiefel:
            model.project_stiefel_gradients_()
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
    from sklearn.metrics import cohen_kappa_score

    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["loss"] = float(total_loss / max(total_samples, 1))
    return metrics


def single_trial_covariances(x_spd: np.ndarray) -> np.ndarray:
    """Return one covariance matrix per trial without temporal/frequency pooling."""
    if x_spd.ndim != 5:
        raise ValueError(
            "SPDNet expects preprocessed SPD data shaped "
            "(trial, segment, frequency, channel, channel); "
            f"got {x_spd.shape}."
        )
    n_segments, n_bands = int(x_spd.shape[1]), int(x_spd.shape[2])
    if n_segments != 1 or n_bands != 1:
        raise ValueError(
            "Classic SPDNet uses exactly one covariance matrix per trial. "
            "Configure one full-trial segment and one frequency band; got "
            f"n_segments={n_segments}, n_bands={n_bands}."
        )
    return np.asarray(x_spd[:, 0, 0], dtype=np.float32)


def validate_spdnet_cv_config(
    training_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
) -> tuple[int, int, bool]:
    n_splits = int(training_cfg.get("n_splits", 5))
    seed = int(training_cfg.get("seed", 42))
    test_size = float(data_cfg.get("test_size", training_cfg.get("test_size", 0.2)))
    val_size = float(data_cfg.get("val_size", training_cfg.get("val_size", 0.0)))
    allow_subject_overlap = parse_bool(
        data_cfg.get(
            "allow_subject_overlap",
            training_cfg.get("allow_subject_overlap", True),
        ),
        default=True,
    )
    if n_splits < 2:
        raise ValueError(f"training.n_splits must be at least 2, got {n_splits}.")
    if not np.isclose(val_size, 0.0):
        raise ValueError(
            "SPDNet K-fold evaluation does not use a validation dataset; "
            f"set val_size to 0.0, got {val_size}."
        )
    expected_test_size = 1.0 / n_splits
    if not np.isclose(test_size, expected_test_size):
        raise ValueError(
            "For K-fold evaluation, test_size must equal 1 / n_splits; "
            f"got test_size={test_size} and n_splits={n_splits} "
            f"(expected {expected_test_size:.6f})."
        )
    return n_splits, seed, allow_subject_overlap


def make_spdnet_cv_splits(
    y: np.ndarray,
    subject_labels: np.ndarray,
    n_splits: int,
    seed: int,
    allow_subject_overlap: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    indices = np.arange(len(y))
    if allow_subject_overlap:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(indices, y)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        iterator = splitter.split(indices, y, groups=subject_labels)
    return [
        (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
        )
        for train_idx, test_idx in iterator
    ]


def aggregate_spdnet_fold_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = {}
    for metric_name in SPDNET_REPORT_METRICS:
        values = np.asarray([row[metric_name] for row in rows], dtype=float)
        aggregates[metric_name] = {
            "mean": float(values.mean()),
            "max": float(values.max()),
            "min": float(values.min()),
        }
    return aggregates


def print_spdnet_fold_summary(
    rows: list[dict[str, Any]],
    aggregates: dict[str, dict[str, float]],
) -> None:
    print("\nSPDNet five-fold test results (no validation dataset)")
    print("fold | train | test | accuracy | macro_f1 | Cohen's kappa")
    for row in rows:
        print(
            f"{row['fold']:>4} | {row['n_train']:>5} | {row['n_test']:>4} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['cohen_kappa']:.4f}"
        )
    print("\nFive-fold aggregate (test folds)")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    run_index: int,
    experiment_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    args: argparse.Namespace,
    base_output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    data_cfg = experiment_cfg["data"]
    training_cfg = experiment_cfg["training"]
    x_spd, y, subject_labels, class_names = load_spd_like_train(data_cfg)
    x_trial_spd = single_trial_covariances(x_spd)
    n_splits, seed, allow_subject_overlap = validate_spdnet_cv_config(
        training_cfg,
        data_cfg,
    )
    folds = make_spdnet_cv_splits(
        y,
        subject_labels,
        n_splits=n_splits,
        seed=seed,
        allow_subject_overlap=allow_subject_overlap,
    )

    dtype = resolve_precision(training_cfg.get("precision", "float64"))
    batch_size = int(training_cfg.get("batch_size", 30))
    num_workers = int(training_cfg.get("num_workers", 0))
    epochs = int(training_cfg.get("epochs", 200))
    learning_rate = float(
        training_cfg.get("spdnet_learning_rate", training_cfg.get("learning_rate", 1e-2))
    )
    weight_decay = float(
        training_cfg.get("spdnet_weight_decay", training_cfg.get("weight_decay", 0.0))
    )
    optimizer_name = str(training_cfg.get("spdnet_optimizer", "sgd"))
    momentum = float(training_cfg.get("spdnet_momentum", 0.0))
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", 0.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    log_every = max(1, int(training_cfg.get("log_every", 10)))

    raw_dims = getattr(args, "dims", None)
    if raw_dims is None:
        raw_dims = model_cfg.get("dims")
    dims = parse_dims(raw_dims, input_dim=x_trial_spd.shape[-1])
    reig_epsilon_override = getattr(args, "reig_epsilon", None)
    log_epsilon_override = getattr(args, "log_epsilon", None)
    reig_epsilon = float(
        reig_epsilon_override
        if reig_epsilon_override is not None
        else model_cfg.get("reig_epsilon", 1e-4)
    )
    log_epsilon = float(
        log_epsilon_override
        if log_epsilon_override is not None
        else model_cfg.get("log_epsilon", 1e-6)
    )
    project_stiefel = parse_bool(
        model_cfg.get("project_stiefel", True),
        default=True,
    ) and not bool(getattr(args, "disable_stiefel_projection", False))

    effective_cfg = dict(experiment_cfg)
    effective_cfg["model"] = dict(model_cfg)
    effective_cfg["model"].update(
        {
            "dims": dims,
            "reig_epsilon": reig_epsilon,
            "log_epsilon": log_epsilon,
            "project_stiefel": project_stiefel,
        }
    )
    run_dir = base_output_dir / f"run_{run_index:03d}_{config_hash(effective_cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", effective_cfg)

    print(f"\n[SPDNet run {run_index}] {run_dir.name}")
    print(
        f"  x_spd={x_spd.shape} x_trial_spd={x_trial_spd.shape} "
        f"class_names={class_names}"
    )
    print(
        f"  dims={dims} epochs={epochs} batch_size={batch_size} "
        f"lr={learning_rate:g} weight_decay={weight_decay:g} "
        f"optimizer={optimizer_name} momentum={momentum:g} "
        f"project_stiefel={project_stiefel} folds={n_splits}"
    )

    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(folds, start=1):
        set_seed(seed + fold_index - 1)
        fold_dir = run_dir / f"fold_{fold_index:02d}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        model = SPDNetClassifier(
            dims=dims,
            num_classes=len(class_names),
            reig_epsilon=reig_epsilon,
            log_epsilon=log_epsilon,
        ).to(device=device, dtype=dtype)
        criterion = nn.CrossEntropyLoss()
        optimizer = build_optimizer(
            model,
            learning_rate,
            weight_decay,
            optimizer_name=optimizer_name,
            momentum=momentum,
        )
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
        test_loader = make_loader(
            x_trial_spd,
            y,
            test_idx,
            batch_size,
            num_workers,
            shuffle=False,
            dtype=dtype,
        )

        history_rows = []
        print(
            f"[SPDNet fold {fold_index}/{n_splits}] "
            f"train={len(train_idx)} test={len(test_idx)}"
        )
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
            history_rows.append({"epoch": epoch, "train_loss": train_loss})
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(
                    f"  fold {fold_index}/{n_splits} epoch {epoch:03d}/{epochs} | "
                    f"train_loss={train_loss:.4f}"
                )

        train_metrics = evaluate(model, train_eval_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        result_rows = []
        for split_name, split_idx, metrics in (
            ("train", train_idx, train_metrics),
            ("test", test_idx, test_metrics),
        ):
            result_rows.append(
                {
                    "split": split_name,
                    "n_samples": int(len(split_idx)),
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "cohen_kappa": metrics["cohen_kappa"],
                }
            )
        write_csv(fold_dir / "history.csv", history_rows)
        write_csv(fold_dir / "results.csv", result_rows)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "config": effective_cfg,
                "dims": dims,
                "fold": fold_index,
                "epochs": epochs,
                "train_indices": train_idx,
                "test_indices": test_idx,
            },
            fold_dir / "final_model.pt",
        )

        fold_row = {
            "fold": fold_index,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "cohen_kappa": test_metrics["cohen_kappa"],
        }
        fold_rows.append(fold_row)
        print(
            f"[SPDNet fold {fold_index}/{n_splits}] "
            f"accuracy={fold_row['accuracy']:.4f} "
            f"mf1={fold_row['macro_f1']:.4f} "
            f"kappa={fold_row['cohen_kappa']:.4f}",
            flush=True,
        )

    aggregates = aggregate_spdnet_fold_metrics(fold_rows)
    print_spdnet_fold_summary(fold_rows, aggregates)
    write_csv(run_dir / "fold_results.csv", fold_rows)
    write_csv(run_dir / "results.csv", fold_rows)

    summary = {
        "baseline": "spdnet",
        "source_repository": "https://github.com/zhiwu-huang/SPDNet",
        "paper": "A Riemannian Network for SPD Matrix Learning, AAAI 2017",
        "architecture": "BiMap-ReEig blocks followed by LogEig and linear FC",
        "input_representation": "one_full_trial_covariance_matrix",
        "temporal_segmentation": False,
        "token_pooling": "none",
        "dims": dims,
        "reig_epsilon": reig_epsilon,
        "log_epsilon": log_epsilon,
        "project_stiefel": project_stiefel,
        "x_spd_shape": list(x_spd.shape),
        "x_trial_spd_shape": list(x_trial_spd.shape),
        "class_names": class_names,
        "evaluation": {
            "strategy": (
                "stratified_kfold"
                if allow_subject_overlap
                else "stratified_group_kfold"
            ),
            "n_splits": n_splits,
            "test_size_per_fold": 1.0 / n_splits,
            "validation_size": 0.0,
            "seed": seed,
        },
        "folds": fold_rows,
        "aggregates": aggregates,
    }
    save_json(run_dir / "summary.json", summary)
    print(
        f"[SPDNet run {run_index}] saved {run_dir} | "
        f"mean_acc={aggregates['accuracy']['mean']:.4f} "
        f"mean_mf1={aggregates['macro_f1']['mean']:.4f} "
        f"mean_kappa={aggregates['cohen_kappa']['mean']:.4f}"
    )
    return {
        "run_index": run_index,
        "run_dir": str(run_dir),
        "test_accuracy_mean": aggregates["accuracy"]["mean"],
        "test_macro_f1_mean": aggregates["macro_f1"]["mean"],
        "test_cohen_kappa_mean": aggregates["cohen_kappa"]["mean"],
        "dims": dims,
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    experiments = expand_data_training_experiments(config)
    model_cfg = dict(config.get("model", {}))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or config.get("output", {}).get(
        "dir",
        "experiments/results/spdnet_baseline",
    )
    base_output_dir = PROJECT_ROOT / output_dir / timestamp
    all_metrics = [
        run_experiment(index, experiment, model_cfg, args, base_output_dir, device)
        for index, experiment in enumerate(experiments, start=1)
    ]
    save_json(base_output_dir / "summary.json", all_metrics)
    print(f"All SPDNet runs complete: {base_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
