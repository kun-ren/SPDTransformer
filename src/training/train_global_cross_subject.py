"""Global subject-wise CV with train/test only and last-epoch checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.PhysioNetMI_pretrain_finetune_preprocess import (
    load_or_preprocess_spd_with_runs, normalize_data_config,
)
from src.models.MotorImageryDataset import MotorImageryDataset
from src.training.config_grid import expand_data_grid, expand_grid
from src.training.domain_adversarial import SubjectDomainDataset, encode_subject_domains
from src.training.losses import prototype_loss_options
from src.training.train import (
    parse_bool, predict_loader, resolve_precision, save_confusion_matrices,
    save_per_class_metrics, set_seed,
)
from src.training.train_pretrain_finetune_loro import (
    _write_csv, format_subject_id, make_loader, make_model, summarize_subjects,
    train_final_epoch, validate_final_epoch_config,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/train_physionet_global_motor_11x9.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    resolved = {}
    for name in ("data", "model", "training", "output"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing mapping section {name!r}.")
        if name == "output":
            resolved[name] = copy.deepcopy(config[name])
        else:
            candidates = (expand_data_grid if name == "data" else expand_grid)(config[name])
            if len(candidates) != 1:
                raise ValueError(f"{name}: use singleton values, not a hyperparameter grid.")
            resolved[name] = candidates[0]
    return resolved


def make_subject_folds(subject_labels, n_splits=5, *, shuffle=True, seed=42):
    subjects = np.unique(subject_labels)
    if not 2 <= n_splits <= len(subjects):
        raise ValueError(f"cv_n_splits must be between 2 and {len(subjects)}, got {n_splits}.")
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    folds = []
    for train_subject_idx, test_subject_idx in splitter.split(subjects):
        train_subjects, test_subjects = subjects[train_subject_idx], subjects[test_subject_idx]
        train_idx = np.flatnonzero(np.isin(subject_labels, train_subjects))
        test_idx = np.flatnonzero(np.isin(subject_labels, test_subjects))
        folds.append((train_subjects, test_subjects, train_idx, test_idx))
    return folds


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="For example cuda:0 or cpu.")
    parser.add_argument("--subjects", help="Override the entire CV cohort, e.g. 1-10.")
    parser.add_argument("--n-splits", type=int, help="Override subject fold count (default 5).")
    parser.add_argument("--epochs", type=int, help="Override fixed epochs per fold.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned subject folds without loading EEG.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    if args.subjects is not None:
        config["data"]["subjects"] = args.subjects
    training = config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.n_splits is not None:
        training["cv_n_splits"] = args.n_splits
    validate_final_epoch_config(training, "training")
    data = normalize_data_config(config["data"])
    config["data"] = data
    n_splits = int(training.get("cv_n_splits", 5))
    shuffle = parse_bool(training.get("cv_shuffle", True))
    cv_seed = int(training.get("cv_seed", 42))
    cohort = data.get("subjects")
    if not cohort:
        raise ValueError("Specify an explicit data.subjects cohort or --subjects.")
    expected_subjects = [format_subject_id(value, data["dataset"]) for value in cohort]
    planned_folds = make_subject_folds(np.array(expected_subjects), n_splits, shuffle=shuffle, seed=cv_seed)
    print(f"Global subject CV: {n_splits} folds, {len(expected_subjects)} subjects; "
          f"no validation, no fine-tuning, epochs={training['epochs']}, checkpoint=last.")
    for fold, (train_subjects, test_subjects, _, _) in enumerate(planned_folds, 1):
        print(f"  Fold {fold}: {len(train_subjects)} train / {len(test_subjects)} test subjects; "
              f"test={','.join(test_subjects)}")
    if args.dry_run:
        return 0

    output = config["output"]
    def resolve_path(value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    cache_dir = resolve_path(output.get("dataset_cache_dir", "experiments/cache/preprocessed_datasets"))
    x, y, subject_labels, run_labels, class_names = load_or_preprocess_spd_with_runs(data, cache_dir)
    loaded = set(subject_labels.tolist())
    if loaded != set(expected_subjects):
        raise ValueError(f"Loaded cohort differs from configuration: "
                         f"missing={sorted(set(expected_subjects) - loaded)}, "
                         f"extra={sorted(loaded - set(expected_subjects))}.")
    folds = make_subject_folds(subject_labels, n_splits, shuffle=shuffle, seed=cv_seed)
    num_classes = len(class_names)
    if num_classes < 2:
        raise ValueError("At least two classes are required.")
    for fold, (_, _, train_idx, _) in enumerate(folds, 1):
        if set(np.unique(y[train_idx])) != set(range(num_classes)):
            raise ValueError(f"Fold {fold} training subjects do not contain all classes.")

    run_dir = resolve_path(output["dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = resolve_precision(training.get("precision", "float32"))
    if device.type == "cuda":
        tf32 = parse_bool(training.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
    dataset = MotorImageryDataset(x, y, dtype=dtype)
    loader_options = {
        "batch_size": int(training.get("batch_size", 128)),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": parse_bool(training.get("pin_memory", device.type == "cuda")),
    }
    domain_enabled = parse_bool(config["model"].get("domain_adversarial", False))
    fold_rows, subject_predictions, prediction_rows = [], [], []
    for fold, (train_subjects, test_subjects, train_idx, test_idx) in enumerate(folds, 1):
        fold_dir = run_dir / f"fold_{fold:02d}"
        fold_dir.mkdir()
        fold_seed = int(training.get("seed", 42)) + fold - 1
        mapping, source_dataset = {}, dataset
        if domain_enabled:
            labels, mapping = encode_subject_domains(subject_labels, train_subjects.tolist())
            source_dataset = SubjectDomainDataset(dataset, labels)
        print(f"Fold {fold}/{n_splits}: train subjects={','.join(train_subjects)}, "
              f"test subjects={','.join(test_subjects)}; trials={len(train_idx)}/{len(test_idx)}; "
              f"train class counts={np.bincount(y[train_idx], minlength=num_classes).tolist()}.")
        split = {
            "fold": fold, "seed": fold_seed, "cv_seed": cv_seed, "cv_shuffle": shuffle,
            "train_subjects": train_subjects.tolist(), "test_subjects": test_subjects.tolist(),
            "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(),
            "validation_indices": [], "checkpoint_selection": "last", "domain_subject_mapping": mapping,
        }
        write_json(fold_dir / "split.json", split)
        set_seed(fold_seed)
        model = make_model(config["model"], x, num_classes, device=device, dtype=dtype,
                           num_domains=len(mapping) if domain_enabled else None)
        train_loader = make_loader(source_dataset, train_idx, shuffle=True, **loader_options)
        state, epoch = train_final_epoch(
            model, train_loader, training, device=device,
            history_path=fold_dir / "train_history.csv", stage_name=f"global fold {fold}",
        )
        if parse_bool(output.get("save_fold_checkpoints", True)):
            torch.save({
                **split, "epoch": epoch, "model_state_dict": state, "class_names": class_names,
                "model_config": config["model"], "training_config": training,
                "input_token_shape": list(x.shape[1:]),
            }, fold_dir / "global_last.pt")
        # Held-out subjects are evaluated only after the final training update.
        test_loader = make_loader(dataset, test_idx, shuffle=False, **loader_options)
        predictions = predict_loader(
            model, test_loader, torch.nn.CrossEntropyLoss(), device,
            condition_regularization_weight=float(training.get("condition_regularization_weight", 0)),
            **prototype_loss_options(training),
        )
        row = {
            "fold": fold, "epoch": epoch, "checkpoint_selection": "last",
            "n_train_subjects": len(train_subjects), "n_test_subjects": len(test_subjects),
            "n_train_trials": len(train_idx), "n_test_trials": len(test_idx),
            "test_accuracy": float(predictions["accuracy"]),
            "test_macro_f1": float(predictions["macro_f1"]),
            "test_cohen_kappa": float(predictions["cohen_kappa"]),
        }
        fold_rows.append(row)
        write_json(fold_dir / "test_metrics.json", row)
        save_per_class_metrics(fold_dir / "per_class_metrics.csv", {"test": predictions}, class_names)
        save_confusion_matrices(fold_dir / "confusion_matrix.csv", {"test": predictions}, class_names)
        for subject in test_subjects:
            mask = subject_labels[test_idx] == subject
            subject_predictions.append({
                "target_subject": str(subject), "_y_true": predictions["y_true"][mask],
                "_y_pred": predictions["y_pred"][mask],
            })
        for index, truth, predicted in zip(test_idx, predictions["y_true"], predictions["y_pred"]):
            prediction_rows.append({
                "fold": fold, "trial_index": int(index), "subject": str(subject_labels[index]),
                "run": int(run_labels[index]), "y_true": int(truth), "y_pred": int(predicted),
            })
        _write_csv(run_dir / "per_fold_results.csv", fold_rows)
        _write_csv(run_dir / "test_predictions.csv", prediction_rows)
        print(f"  Fold {fold} final epoch {epoch}: test accuracy={row['test_accuracy']:.4f}")
        del model, state, train_loader, test_loader, source_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    subject_rows = summarize_subjects(subject_predictions, num_classes=num_classes)
    _write_csv(run_dir / "per_subject_summary.csv", subject_rows)
    scores = [row["test_accuracy"] for row in fold_rows]
    summary = {
        "protocol": "subject-wise global cross-validation, no validation or fine-tuning",
        "n_splits": n_splits, "n_subjects": len(loaded), "epochs": int(training["epochs"]),
        "checkpoint_selection": "last", "mean_fold_test_accuracy": statistics.fmean(scores),
        "fold_test_accuracy_sd": statistics.stdev(scores),
        "mean_subject_accuracy": statistics.fmean(row["Accuracy (%)"] / 100 for row in subject_rows),
        "pooled_trial_accuracy": statistics.fmean(row["y_true"] == row["y_pred"] for row in prediction_rows),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"Mean {n_splits}-fold test accuracy={summary['mean_fold_test_accuracy']:.4f}; "
          f"mean subject accuracy={summary['mean_subject_accuracy']:.4f}; results={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
