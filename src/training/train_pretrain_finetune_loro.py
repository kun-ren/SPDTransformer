"""Cross-subject pretraining followed by target-subject LORO fine-tuning.

For each requested target subject:
1. train a fresh model on every trial from all *other* subjects;
2. restore that identical pretrained state for every target EDF run;
3. fine-tune on the target subject's remaining runs;
4. evaluate the held-out run exactly once after the final fine-tuning epoch.

There is deliberately no validation set, test-driven scheduler, early stopping,
or checkpoint selection in this protocol.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import geoopt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.PhysioNetMI_pretrain_finetune_preprocess import (  # noqa: E402
    load_or_preprocess_spd_with_runs,
    normalize_data_config,
    parse_subjects,
    selected_run_ids,
)
from src.models.MotorImageryDataset import MotorImageryDataset  # noqa: E402
from src.training.train import (  # noqa: E402
    build_model,
    normalize_precision_name,
    parse_bool,
    predict_loader,
    resolve_precision,
    save_confusion_matrices,
    save_per_class_metrics,
    set_seed,
    split_params,
    train_one_epoch,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_physionet_pretrain_finetune_loro.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    for section in ("data", "model", "pretrain", "fine_tune", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section {section!r} in {path}.")
    return config


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def make_loader(
    full_dataset: MotorImageryDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if len(indices) == 0:
        raise ValueError("Cannot create a loader with zero trials.")
    return DataLoader(
        Subset(full_dataset, indices.astype(int).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def make_model(
    model_cfg: dict[str, Any],
    x: np.ndarray,
    num_classes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model = build_model(
        model_cfg=copy.deepcopy(model_cfg),
        spd_in_dim=int(x.shape[-1]),
        num_classes=num_classes,
        time_sequence_length=int(x.shape[1]),
        frequency_sequence_length=int(x.shape[2]) if x.ndim >= 5 else 1,
        brain_region_sequence_length=int(x.shape[3]) if x.ndim >= 6 else 1,
    )
    return model.to(device=device, dtype=dtype)


def make_optimizers(
    model: nn.Module,
    cfg: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    stiefel_params, decay_params, no_decay_params = split_params(model)
    euclidean_groups = []
    if decay_params:
        euclidean_groups.append(
            {
                "params": decay_params,
                "weight_decay": float(cfg.get("weight_decay", 0.0)),
            }
        )
    if no_decay_params:
        euclidean_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not euclidean_groups:
        raise RuntimeError("Model has no Euclidean trainable parameters.")
    optimizer = torch.optim.AdamW(
        euclidean_groups,
        lr=float(cfg.get("learning_rate", 1e-3)),
    )
    optimizer_stiefel = None
    if stiefel_params:
        optimizer_stiefel = geoopt.optim.RiemannianAdam(
            stiefel_params,
            lr=float(cfg.get("stiefel_learning_rate", 3e-4)),
            weight_decay=0.0,
            stabilize=10,
        )
    return optimizer, optimizer_stiefel


def train_fixed_epochs(
    model: nn.Module,
    loader: DataLoader,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    history_path: Path,
    stage_name: str,
) -> list[dict[str, Any]]:
    """Train for a fixed epoch count using training data only."""

    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError(f"{stage_name}.epochs must be at least 1.")
    optimizer, optimizer_stiefel = make_optimizers(model, cfg)
    criterion = nn.CrossEntropyLoss()
    gradient_clip_norm = cfg.get("gradient_clip_norm", 1.0)
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    condition_weight = float(cfg.get("condition_regularization_weight", 0.0))
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            optimizer_stiefel,
            device,
            gradient_clip_norm=gradient_clip_norm,
            condition_regularization_weight=condition_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(metrics["loss"]),
            "train_accuracy": float(metrics["accuracy"]),
            "train_macro_f1": float(metrics["macro_f1"]),
        }
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"    {stage_name} epoch {epoch:03d}/{epochs}: "
                f"loss={row['train_loss']:.4f}, "
                f"accuracy={row['train_accuracy']:.4f}"
            )
    _write_csv(history_path, history)
    return history


def validate_protocol(
    y: np.ndarray,
    subject_labels: np.ndarray,
    run_labels: np.ndarray,
    target_subjects: list[int],
    *,
    num_classes: int,
) -> dict[str, list[int]]:
    """Audit target availability and class coverage without changing splits."""

    available_subjects = set(subject_labels.tolist())
    run_map: dict[str, list[int]] = {}
    for target_number in target_subjects:
        target = f"S{target_number:03d}"
        if target not in available_subjects:
            raise ValueError(f"Target subject {target} is absent from the dataset.")
        pretrain_mask = subject_labels != target
        target_mask = subject_labels == target
        if len(set(subject_labels[pretrain_mask].tolist())) == 0:
            raise ValueError(f"{target} has no other subjects available for pretraining.")
        if set(np.unique(y[pretrain_mask]).tolist()) != set(range(num_classes)):
            raise ValueError(f"Other-subject pretraining data for {target} lacks classes.")
        target_runs = sorted(int(value) for value in np.unique(run_labels[target_mask]))
        if len(target_runs) < 2:
            raise ValueError(f"{target} needs at least two retained runs for LORO.")
        for test_run in target_runs:
            fine_tune_mask = target_mask & (run_labels != test_run)
            test_mask = target_mask & (run_labels == test_run)
            if not fine_tune_mask.any() or not test_mask.any():
                raise RuntimeError(f"Empty LORO split for {target} R{test_run:02d}.")
            if set(np.unique(y[fine_tune_mask]).tolist()) != set(range(num_classes)):
                raise ValueError(
                    f"Fine-tuning data for {target} excluding R{test_run:02d} "
                    "does not contain every class."
                )
        run_map[target] = target_runs
    return run_map


def _run_metrics(
    predictions: dict[str, Any],
    *,
    num_classes: int,
) -> dict[str, Any]:
    y_true = np.asarray(predictions["y_true"], dtype=np.int64)
    y_pred = np.asarray(predictions["y_pred"], dtype=np.int64)
    present = sorted(np.unique(y_true).astype(int).tolist())
    return {
        "test_accuracy": float(predictions["accuracy"]),
        "test_macro_f1_all_classes": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "test_macro_f1_present_classes": float(
            f1_score(
                y_true,
                y_pred,
                labels=present,
                average="macro",
                zero_division=0,
            )
        ),
        "test_cohen_kappa": float(predictions["cohen_kappa"]),
        "test_present_class_indices": ",".join(str(value) for value in present),
    }


def summarize_subjects(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject_id in sorted({str(row["target_subject"]) for row in run_rows}):
        subject_runs = [row for row in run_rows if row["target_subject"] == subject_id]
        accuracies = [float(row["test_accuracy"]) for row in subject_runs]
        pooled_true = np.concatenate([row["_y_true"] for row in subject_runs])
        pooled_pred = np.concatenate([row["_y_pred"] for row in subject_runs])
        rows.append(
            {
                "target_subject": subject_id,
                "n_test_runs": len(subject_runs),
                "mean_run_accuracy": statistics.fmean(accuracies),
                "std_run_accuracy": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "min_run_accuracy": min(accuracies),
                "max_run_accuracy": max(accuracies),
                "pooled_trial_accuracy": float((pooled_true == pooled_pred).mean()),
                "pooled_four_class_macro_f1": float(
                    f1_score(
                        pooled_true,
                        pooled_pred,
                        labels=np.arange(4),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--target-subjects",
        help="Override config targets, for example 1, 1-10, or 1-3,8.",
    )
    parser.add_argument(
        "--data-subjects",
        help="Optional dataset-cohort override, mainly for a small smoke test.",
    )
    parser.add_argument("--device", help="Device override, for example cuda:0 or cpu.")
    parser.add_argument("--pretrain-epochs", type=int, help="Smoke-test override.")
    parser.add_argument("--fine-tune-epochs", type=int, help="Smoke-test override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the protocol without loading EEG data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    data_cfg = normalize_data_config(config["data"])
    if args.data_subjects is not None:
        data_cfg["subjects"] = parse_subjects(args.data_subjects)
    target_subjects = parse_subjects(
        args.target_subjects
        if args.target_subjects is not None
        else config.get("target_subjects")
    )
    if not target_subjects:
        raise ValueError("Specify target_subjects in the config or CLI.")
    dataset_subjects = data_cfg.get("subjects")
    if dataset_subjects is not None:
        missing_targets = sorted(set(target_subjects) - set(dataset_subjects))
        if missing_targets:
            raise ValueError(
                f"Targets are outside data.subjects: {missing_targets}."
            )

    pretrain_cfg = copy.deepcopy(config["pretrain"])
    fine_tune_cfg = copy.deepcopy(config["fine_tune"])
    if args.pretrain_epochs is not None:
        pretrain_cfg["epochs"] = args.pretrain_epochs
    if args.fine_tune_epochs is not None:
        fine_tune_cfg["epochs"] = args.fine_tune_epochs
    for name, cfg in (("pretrain", pretrain_cfg), ("fine_tune", fine_tune_cfg)):
        if int(cfg.get("epochs", 0)) < 1:
            raise ValueError(f"{name}.epochs must be at least 1.")

    expected_runs = selected_run_ids(
        imaged=bool(data_cfg["imaged"]),
        executed=bool(data_cfg["executed"]),
        task_types=tuple(data_cfg["task_types"]),
    )
    print("Protocol: other-subject pretraining -> target-subject leave-one-run-out.")
    print("Validation set: none. Test runs are evaluated only after fixed-epoch training.")
    print(
        f"Targets={','.join(f'S{value:03d}' for value in target_subjects)}, "
        f"expected runs={expected_runs}, pretrain epochs={pretrain_cfg['epochs']}, "
        f"fine-tune epochs={fine_tune_cfg['epochs']}."
    )
    if args.dry_run:
        return 0

    output_cfg = config["output"]
    output_root = Path(
        str(output_cfg.get("dir", "experiments/results/pretrain_finetune_loro"))
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = Path(
        str(output_cfg.get("dataset_cache_dir", "experiments/cache/preprocessed_datasets"))
    )
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                **config,
                "data": data_cfg,
                "target_subjects": target_subjects,
                "pretrain": pretrain_cfg,
                "fine_tune": fine_tune_cfg,
            },
            handle,
            sort_keys=False,
        )

    x, y, subject_labels, run_labels, class_names = (
        load_or_preprocess_spd_with_runs(data_cfg, cache_dir)
    )
    num_classes = len(class_names)
    if num_classes != 4:
        raise ValueError(
            f"This experiment expects four classes, got {num_classes}: {class_names}."
        )
    run_map = validate_protocol(
        y,
        subject_labels,
        run_labels,
        target_subjects,
        num_classes=num_classes,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = normalize_precision_name(pretrain_cfg.get("precision", "float32"))
    fine_tune_precision = normalize_precision_name(
        fine_tune_cfg.get("precision", precision)
    )
    if fine_tune_precision != precision:
        raise ValueError("pretrain and fine_tune precision must match.")
    dtype = resolve_precision(precision)
    # Construct one tensor view for the whole cohort. Subset keeps only integer
    # indices, avoiding a multi-gigabyte NumPy copy for every target subject.
    full_dataset = MotorImageryDataset(x, y, dtype=dtype)
    num_workers = int(pretrain_cfg.get("num_workers", 0))
    pin_memory = parse_bool(
        pretrain_cfg.get("pin_memory", device.type == "cuda"),
        default=device.type == "cuda",
    )
    seed = int(pretrain_cfg.get("seed", 42))
    run_rows: list[dict[str, Any]] = []

    for target_position, target_number in enumerate(target_subjects, start=1):
        target = f"S{target_number:03d}"
        target_dir = run_dir / target
        target_dir.mkdir()
        target_mask = subject_labels == target
        pretrain_indices = np.flatnonzero(~target_mask)
        pretrain_subject_count = len(set(subject_labels[pretrain_indices].tolist()))
        print(
            f"\n[{target_position}/{len(target_subjects)}] {target}: pretraining on "
            f"{len(pretrain_indices)} shuffled trials from {pretrain_subject_count} other subjects."
        )
        set_seed(seed + target_number)
        base_model = make_model(
            config["model"],
            x,
            num_classes,
            device=device,
            dtype=dtype,
        )
        pretrain_loader = make_loader(
            full_dataset,
            pretrain_indices,
            batch_size=int(pretrain_cfg.get("batch_size", 128)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        train_fixed_epochs(
            base_model,
            pretrain_loader,
            pretrain_cfg,
            device=device,
            history_path=target_dir / "pretrain_history.csv",
            stage_name="pretrain",
        )
        pretrained_state = _cpu_state_dict(base_model)
        if parse_bool(
            output_cfg.get("save_pretrained_checkpoints", True), default=True
        ):
            torch.save(
                {
                    "model_state_dict": pretrained_state,
                    "excluded_target_subject": target,
                    "class_names": class_names,
                    "pretrain_subjects": sorted(
                        set(subject_labels[pretrain_indices].tolist())
                    ),
                },
                target_dir / "pretrained_other_subjects.pt",
            )
        del base_model, pretrain_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for fold_position, test_run in enumerate(run_map[target], start=1):
            fold_dir = target_dir / f"test_R{test_run:02d}"
            fold_dir.mkdir()
            fine_tune_indices = np.flatnonzero(target_mask & (run_labels != test_run))
            test_indices = np.flatnonzero(target_mask & (run_labels == test_run))
            if set(fine_tune_indices.tolist()) & set(test_indices.tolist()):
                raise RuntimeError(f"Train/test overlap for {target} R{test_run:02d}.")
            print(
                f"  [{fold_position}/{len(run_map[target])}] test R{test_run:02d}: "
                f"fine-tune={len(fine_tune_indices)} trials, test={len(test_indices)} trials."
            )
            set_seed(seed + target_number * 100 + test_run)
            model = make_model(
                config["model"],
                x,
                num_classes,
                device=device,
                dtype=dtype,
            )
            model.load_state_dict(pretrained_state)
            fine_tune_loader = make_loader(
                full_dataset,
                fine_tune_indices,
                batch_size=int(fine_tune_cfg.get("batch_size", 32)),
                shuffle=True,
                num_workers=int(fine_tune_cfg.get("num_workers", num_workers)),
                pin_memory=parse_bool(
                    fine_tune_cfg.get("pin_memory", pin_memory),
                    default=pin_memory,
                ),
            )
            train_fixed_epochs(
                model,
                fine_tune_loader,
                fine_tune_cfg,
                device=device,
                history_path=fold_dir / "fine_tune_history.csv",
                stage_name="fine-tune",
            )
            test_loader = make_loader(
                full_dataset,
                test_indices,
                batch_size=int(fine_tune_cfg.get("batch_size", 32)),
                shuffle=False,
                num_workers=int(fine_tune_cfg.get("num_workers", num_workers)),
                pin_memory=parse_bool(
                    fine_tune_cfg.get("pin_memory", pin_memory),
                    default=pin_memory,
                ),
            )
            predictions = predict_loader(
                model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
                condition_regularization_weight=float(
                    fine_tune_cfg.get("condition_regularization_weight", 0.0)
                ),
            )
            metrics = _run_metrics(predictions, num_classes=num_classes)
            present_names = [
                class_names[int(value)]
                for value in metrics["test_present_class_indices"].split(",")
            ]
            row: dict[str, Any] = {
                "target_subject": target,
                "test_run": int(test_run),
                "fine_tune_runs": ",".join(
                    str(value)
                    for value in sorted(np.unique(run_labels[fine_tune_indices]))
                ),
                "n_pretrain_trials": int(len(pretrain_indices)),
                "n_fine_tune_trials": int(len(fine_tune_indices)),
                "n_test_trials": int(len(test_indices)),
                "test_present_classes": ",".join(present_names),
                **metrics,
                "_y_true": predictions["y_true"],
                "_y_pred": predictions["y_pred"],
            }
            run_rows.append(row)
            save_per_class_metrics(
                fold_dir / "per_class_metrics.csv",
                {"test": predictions},
                class_names,
            )
            save_confusion_matrices(
                fold_dir / "confusion_matrix.csv",
                {"test": predictions},
                class_names,
            )
            with (fold_dir / "split.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "target_subject": target,
                        "test_run": int(test_run),
                        "fine_tune_runs": sorted(
                            int(value)
                            for value in np.unique(run_labels[fine_tune_indices])
                        ),
                        "pretrain_subjects": sorted(
                            set(subject_labels[pretrain_indices].tolist())
                        ),
                        "fine_tune_indices": fine_tune_indices.astype(int).tolist(),
                        "test_indices": test_indices.astype(int).tolist(),
                    },
                    handle,
                    indent=2,
                )
            if parse_bool(
                output_cfg.get("save_fine_tuned_checkpoints", False), default=False
            ):
                torch.save(
                    {
                        "model_state_dict": _cpu_state_dict(model),
                        "target_subject": target,
                        "test_run": int(test_run),
                        "class_names": class_names,
                    },
                    fold_dir / "fine_tuned_model.pt",
                )
            print(
                f"    test accuracy={row['test_accuracy']:.4f}, "
                f"present-class macro-F1={row['test_macro_f1_present_classes']:.4f}"
            )
            del model, fine_tune_loader, test_loader
            if device.type == "cuda":
                torch.cuda.empty_cache()

    public_run_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in run_rows
    ]
    _write_csv(run_dir / "per_run_results.csv", public_run_rows)
    subject_rows = summarize_subjects(run_rows)
    _write_csv(run_dir / "per_subject_summary.csv", subject_rows)
    pooled_true = np.concatenate([row["_y_true"] for row in run_rows])
    pooled_pred = np.concatenate([row["_y_pred"] for row in run_rows])
    subject_mean_accuracies = [float(row["mean_run_accuracy"]) for row in subject_rows]
    overall = {
        "protocol": "other-subject pretraining plus target-subject leave-one-run-out fine-tuning",
        "validation_set": None,
        "n_target_subjects": len(subject_rows),
        "class_names": class_names,
        "chance_accuracy": 1.0 / num_classes,
        "mean_of_subject_mean_run_accuracies": statistics.fmean(
            subject_mean_accuracies
        ),
        "between_subject_accuracy_sd": (
            statistics.stdev(subject_mean_accuracies)
            if len(subject_mean_accuracies) > 1
            else 0.0
        ),
        "pooled_trial_accuracy": float((pooled_true == pooled_pred).mean()),
        "pooled_four_class_macro_f1": float(
            f1_score(
                pooled_true,
                pooled_pred,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
    }
    with (run_dir / "overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    matrix = confusion_matrix(
        pooled_true,
        pooled_pred,
        labels=np.arange(num_classes),
    )
    confusion_rows = []
    for true_index, class_name in enumerate(class_names):
        confusion_rows.append(
            {
                "true_class": class_name,
                **{
                    f"pred_{predicted_name}": int(matrix[true_index, pred_index])
                    for pred_index, predicted_name in enumerate(class_names)
                },
            }
        )
    _write_csv(run_dir / "pooled_confusion_matrix.csv", confusion_rows)
    print(
        "\nPer-subject LORO accuracy: "
        f"{overall['mean_of_subject_mean_run_accuracies'] * 100:.2f} +/- "
        f"{overall['between_subject_accuracy_sd'] * 100:.2f}% "
        "(mean +/- between-subject SD)"
    )
    print(f"Saved results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
