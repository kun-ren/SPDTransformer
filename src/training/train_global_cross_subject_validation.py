"""Global SPDTransformer tuning with subject-disjoint train/validation/test.

Validation selects epochs and hyperparameters. Optional outer-test evaluation
uses only the candidate selected WITHIN that fold, never the global ranking.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import train_pretrain_finetune_loro as core
from src.training.train_global_cross_subject import make_subject_folds

DEFAULT_CONFIG = ROOT / "configs/train_physionet_global_validation_motor_11x9.yaml"
PROTOCOL_KEYS = ("cv_n_splits", "cv_shuffle", "cv_seed", "validation_subject_fraction",
                 "selection_metric", "evaluate_test")


def load_candidates(path):
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    data = core.expand_data_grid(raw["data"])
    if len(data) != 1:
        raise ValueError("Freeze preprocessing: this tuner supports only one data configuration.")
    candidates = [{"data": copy.deepcopy(data[0]), "model": model, "training": training,
                   "output": copy.deepcopy(raw["output"])}
                  for model, training in itertools.product(core.expand_grid(raw["model"]), core.expand_grid(raw["training"]))]
    for cfg in candidates:
        tr = cfg["training"]
        if not core.parse_bool(tr.get("use_validation", True)) or tr.get("checkpoint_selection", "best_validation") != "best_validation":
            raise ValueError("Validation tuner requires use_validation=true and checkpoint_selection=best_validation.")
        if int(tr.get("epochs", 0)) < 1 or tr.get("selection_metric", "accuracy") not in {"accuracy", "macro_f1"}:
            raise ValueError("Need positive epochs and selection_metric=accuracy or macro_f1.")
        if core.parse_bool(tr.get("early_stopping", True)) and int(tr.get("early_stopping_patience", 20)) < 1:
            raise ValueError("Set early_stopping=false to disable stopping; otherwise patience must be positive.")
        if float(tr.get("early_stopping_min_delta", 0)) < 0:
            raise ValueError("early_stopping_min_delta must be nonnegative.")
        if "lr_scheduler_metric" in tr and tr["lr_scheduler_metric"] not in {"validation_loss", "validation_accuracy", "validation_macro_f1"}:
            raise ValueError("The scheduler may only monitor validation metrics in this tuner.")
        if any(tr.get(k) != candidates[0]["training"].get(k) for k in PROTOCOL_KEYS):
            raise ValueError("All candidates must use identical splits, selection metric and test policy.")
    return candidates


def make_validation_folds(subjects, training):
    fraction = float(training.get("validation_subject_fraction", .2))
    if not 0 < fraction < 1:
        raise ValueError("validation_subject_fraction must be in (0,1).")
    seed = int(training.get("cv_seed", 42))
    folds = []
    for fold, (development, test_subjects, _, test) in enumerate(make_subject_folds(
        subjects, int(training.get("cv_n_splits", 5)), shuffle=core.parse_bool(training.get("cv_shuffle", True)), seed=seed,
    ), 1):
        count = math.ceil(len(development) * fraction)
        if count >= len(development):
            raise ValueError("Too few training subjects after reserving validation subjects.")
        validation_subjects = np.sort(np.random.default_rng(seed + 10000 + fold).permutation(development)[:count])
        train_subjects = np.setdiff1d(development, validation_subjects)
        folds.append({"fold": fold, "train_subjects": train_subjects.tolist(),
                      "validation_subjects": validation_subjects.tolist(), "test_subjects": test_subjects.tolist(),
                      "train_indices": np.flatnonzero(np.isin(subjects, train_subjects)).tolist(),
                      "validation_indices": np.flatnonzero(np.isin(subjects, validation_subjects)).tolist(),
                      "test_indices": test.tolist()})
    return folds


def fit_validation(model, train_loader, validation_loader, cfg, *, device, path):
    optimizer, stiefel = core.make_optimizers(model, cfg)
    scheduler_cfg = dict(cfg)
    scheduler_cfg.setdefault("lr_scheduler_metric", "validation_" + cfg.get("selection_metric", "accuracy"))
    scheduler_cfg.setdefault("lr_scheduler_mode", "min" if scheduler_cfg["lr_scheduler_metric"] == "validation_loss" else "max")
    name, scheduler_metric, scheduler, stiefel_scheduler = core.build_lr_schedulers(scheduler_cfg, optimizer, stiefel)
    criterion = torch.nn.CrossEntropyLoss()
    loss_options = {"condition_regularization_weight": float(cfg.get("condition_regularization_weight", 0)),
                    **core.prototype_loss_options(cfg)}
    metric = cfg.get("selection_metric", "accuracy")
    best, stop_reference, stale, state = -np.inf, -np.inf, 0, None
    history, best_row = [], None
    for epoch in range(1, int(cfg["epochs"]) + 1):
        lr = core.optimizer_lr_values(optimizer)[0]
        stiefel_lr = core.optimizer_lr_values(stiefel)[0] if stiefel else None
        train = core.train_one_epoch(
            model, train_loader, criterion, optimizer, stiefel, device,
            gradient_clip_norm=cfg.get("gradient_clip_norm", 1.),
            debug_anomaly=core.parse_bool(cfg.get("debug_anomaly", False)),
            **loss_options, **core.domain_epoch_options(model, cfg, epoch, int(cfg["epochs"])),
        )
        validation = core.evaluate(model, validation_loader, criterion, device, **loss_options)
        if not all(np.isfinite(validation[key]) for key in ("loss", "accuracy", "macro_f1")):
            raise RuntimeError("Non-finite validation metrics; refusing to select a checkpoint.")
        score = float(validation[metric])
        row = {"epoch": epoch, "train_loss": float(train["loss"]), "train_accuracy": float(train["accuracy"]),
               "train_macro_f1": float(train["macro_f1"]), "validation_loss": float(validation["loss"]),
               "validation_accuracy": float(validation["accuracy"]), "validation_macro_f1": float(validation["macro_f1"]),
               "accuracy_gap": float(train["accuracy"] - validation["accuracy"]),
               "euclid_lr": lr, "stiefel_lr": stiefel_lr,
               **core.auxiliary_loss_history(train), **core.auxiliary_loss_history(validation, "validation")}
        # min_delta controls patience, not whether the true best checkpoint is saved.
        if score > best:
            best, best_row, state = score, dict(row), core._cpu_state_dict(model)
        if score > stop_reference + float(cfg.get("early_stopping_min_delta", 0)):
            stop_reference, stale = score, 0
        else:
            stale += 1
        if scheduler is not None:
            if name == "multistep":
                scheduler.step()
                if stiefel_scheduler is not None:
                    stiefel_scheduler.step()
            else:
                value = {"validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                         "validation_macro_f1": validation["macro_f1"]}[scheduler_metric]
                scheduler.step(float(value))
                if stiefel_scheduler is not None:
                    stiefel_scheduler.step(float(value))
        history.append(row)
        core._write_csv(path / "history.csv", history)
        print(f"  epoch {epoch}/{cfg['epochs']}: train acc={row['train_accuracy']:.4f}, "
              f"val acc={row['validation_accuracy']:.4f}, val MF1={row['validation_macro_f1']:.4f}, "
              f"val loss={row['validation_loss']:.4f}, gap={row['accuracy_gap']:.4f}, "
              f"lr={lr:.3e}/{stiefel_lr}; best {metric}={best:.4f} at epoch {best_row['epoch']}")
        if core.parse_bool(cfg.get("early_stopping", True)) and stale >= int(cfg.get("early_stopping_patience", 20)):
            print(f"  Early stopping: {stale} epochs without min_delta improvement.")
            break
    model.load_state_dict(state, strict=True)
    return state, {**best_row, "epochs_run": len(history), "selection_score": best}


def rank_candidates(rows, candidates):
    ranking = []
    for index, cfg in enumerate(candidates, 1):
        folds = [r for r in rows if r["candidate"] == index]
        record = {"candidate": index, "n_folds": len(folds), "selection_metric": cfg["training"].get("selection_metric", "accuracy"),
                  "model_json": json.dumps(cfg["model"], sort_keys=True), "training_json": json.dumps(cfg["training"], sort_keys=True)}
        for key in ("selection_score", "validation_accuracy", "validation_macro_f1", "validation_loss", "train_accuracy", "accuracy_gap", "epoch"):
            values = [r[key] for r in folds]
            record[key + "_mean"] = float(np.mean(values))
            record[key + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        ranking.append(record)
    return sorted(ranking, key=lambda r: (-r["selection_score_mean"], r["candidate"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    candidates = load_candidates(args.config)
    for cfg in candidates:
        if args.subjects:
            cfg["data"]["subjects"] = args.subjects
        if args.epochs is not None:
            if args.epochs < 1:
                raise ValueError("--epochs must be positive.")
            cfg["training"]["epochs"] = args.epochs
    data = core.normalize_data_config(candidates[0]["data"])
    if not data.get("subjects"):
        raise ValueError("An explicit subject cohort is required.")
    expected = np.array([core.format_subject_id(s, data["dataset"]) for s in data["subjects"]])
    training = candidates[0]["training"]
    planned = make_validation_folds(expected, training)
    print(f"Global validation tuning: {len(candidates)} candidates x {len(planned)} outer folds; "
          f"checkpoint=best validation {training.get('selection_metric', 'accuracy')}; "
          f"evaluate_test={core.parse_bool(training.get('evaluate_test', False))}.")
    for split in planned:
        print(f"  Fold {split['fold']}: {len(split['train_subjects'])} train / "
              f"{len(split['validation_subjects'])} validation / {len(split['test_subjects'])} test subjects")
    if args.dry_run:
        return 0
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    output = candidates[0]["output"]
    x, y, subjects, runs, names = core.load_or_preprocess_spd_with_runs(
        data, resolve(output.get("dataset_cache_dir", "experiments/cache/preprocessed_datasets")),
    )
    if set(subjects) != set(expected):
        raise ValueError("Loaded cohort differs from the requested subjects.")
    splits = make_validation_folds(subjects, training)
    for split in splits:
        for part in ("train", "validation"):
            if set(np.unique(y[split[part + "_indices"]])) != set(range(len(names))):
                raise ValueError(f"Fold {split['fold']} {part} subjects do not contain every class.")
    directory = resolve(output["dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True)
    (directory / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, selections, tests, predictions = [], [], [], []
    for split in splits:
        winner_state, winner_cfg, winner_row, winner_domains = None, None, None, None
        for candidate, cfg in enumerate(candidates, 1):
            tr = cfg["training"]
            core.set_seed(int(tr.get("seed", 42)) + split["fold"] - 1)
            dtype = core.resolve_precision(tr.get("precision", "float32"))
            if device.type == "cuda":
                tf32 = core.parse_bool(tr.get("allow_tf32", False))
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.backends.cudnn.allow_tf32 = tf32
                torch.set_float32_matmul_precision("high" if tf32 else "highest")
            dataset = core.MotorImageryDataset(x, y, dtype=dtype)
            mapping, source = {}, dataset
            if core.parse_bool(cfg["model"].get("domain_adversarial", False)):
                domains, mapping = core.encode_subject_domains(subjects, split["train_subjects"])
                source = core.SubjectDomainDataset(dataset, domains)
            model = core.make_model(cfg["model"], x, len(names), device=device, dtype=dtype,
                                    num_domains=len(mapping) if mapping else None)
            loader_args = {"batch_size": int(tr.get("batch_size", 128)), "num_workers": int(tr.get("num_workers", 0)),
                           "pin_memory": core.parse_bool(tr.get("pin_memory", device.type == "cuda"))}
            train_loader = core.make_loader(source, np.asarray(split["train_indices"]), shuffle=True, **loader_args)
            validation_loader = core.make_loader(dataset, np.asarray(split["validation_indices"]), shuffle=False, **loader_args)
            path = directory / f"candidate_{candidate:03d}" / f"fold_{split['fold']:02d}"
            path.mkdir(parents=True)
            (path.parent / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Fold {split['fold']}, candidate {candidate}/{len(candidates)}, trainable parameters={parameter_count}")
            state, result = fit_validation(model, train_loader, validation_loader, tr, device=device, path=path)
            row = {"candidate": candidate, "fold": split["fold"], "trainable_parameters": parameter_count, **result}
            rows.append(row)
            torch.save({"model_state_dict": state, "epoch": result["epoch"], "config": cfg,
                        "split": split, "class_names": names, "domain_subject_mapping": mapping,
                        "checkpoint_selection": "best_validation", "selection_metric": tr.get("selection_metric", "accuracy")},
                       path / "best_validation.pt")
            if winner_row is None or row["selection_score"] > winner_row["selection_score"]:
                winner_state, winner_cfg, winner_row, winner_domains = state, cfg, row, mapping
            core._write_csv(directory / "validation_fold_results.csv", rows)
            del model, train_loader, validation_loader, source, dataset, state
            if device.type == "cuda":
                torch.cuda.empty_cache()
        selections.append(winner_row)
        core._write_csv(directory / "fold_selections.csv", selections)
        if core.parse_bool(training.get("evaluate_test", False)):
            tr = winner_cfg["training"]
            dtype = core.resolve_precision(tr.get("precision", "float32"))
            if device.type == "cuda":
                tf32 = core.parse_bool(tr.get("allow_tf32", False))
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.backends.cudnn.allow_tf32 = tf32
                torch.set_float32_matmul_precision("high" if tf32 else "highest")
            model = core.make_model(winner_cfg["model"], x, len(names), device=device, dtype=dtype,
                                    num_domains=len(winner_domains) if winner_domains else None)
            model.load_state_dict(winner_state, strict=True)
            dataset = core.MotorImageryDataset(x, y, dtype=dtype)
            loader = core.make_loader(dataset, np.asarray(split["test_indices"]), shuffle=False,
                                      batch_size=int(tr.get("batch_size", 128)), num_workers=int(tr.get("num_workers", 0)), pin_memory=False)
            result = core.predict_loader(model, loader, torch.nn.CrossEntropyLoss(), device)
            tests.append({"fold": split["fold"], "candidate": winner_row["candidate"],
                          **{k: float(result[k]) for k in ("accuracy", "macro_f1", "cohen_kappa")}})
            predictions.extend({"fold": split["fold"], "subject": str(subjects[i]), "trial_index": int(i),
                                "run": int(runs[i]), "y_true": int(a), "y_pred": int(b)}
                               for i, a, b in zip(split["test_indices"], result["y_true"], result["y_pred"], strict=True))
            core._write_csv(directory / "outer_test_results.csv", tests)
            core._write_csv(directory / "outer_test_predictions.csv", predictions)
            del model, dataset, loader
        del winner_state
    ranking = rank_candidates(rows, candidates)
    core._write_csv(directory / "candidate_ranking.csv", ranking)
    best = ranking[0]
    recommendation = {"candidate": best["candidate"], "mean_validation_score": best["selection_score_mean"],
                      "test_evaluated": core.parse_bool(training.get("evaluate_test", False)),
                      "note": "Validation tuning recommendation, not an unbiased final test score.",
                      "config": candidates[best["candidate"] - 1]}
    (directory / "selected_hyperparameters.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    print(f"Best mean validation candidate={best['candidate']}, score={best['selection_score_mean']:.4f}; output={directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
