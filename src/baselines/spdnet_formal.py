"""No-validation SPDNet global CV and source-pretrain / target-run fine-tuning."""

from copy import deepcopy
from pathlib import Path
import statistics

import numpy as np
import torch

from src.baselines import spdnet_baseline as spd
from src.baselines.baseline_utils import (
    config_hash, load_spd_like_train, make_subject_specific_loro_splits,
    parse_bool, parse_subjects, save_json, summarize_subject_fold_metrics,
)
from src.training.train_global_cross_subject import make_subject_folds
from src.training.train_pretrain_finetune_loro import format_subject_id, validate_final_epoch_config


def fit_last(model, x, y, indices, cfg, *, device, dtype, project_stiefel, path):
    """Fresh optimizer/scheduler; neither early stopping nor test-based selection."""
    validate_final_epoch_config(cfg, "SPDNet")
    path.mkdir(parents=True, exist_ok=True)
    optimizer = spd.build_optimizer(
        model, float(cfg.get("spdnet_learning_rate", 0.01)),
        float(cfg.get("spdnet_weight_decay", 0.0005)),
        optimizer_name=cfg.get("spdnet_optimizer", "sgd"),
        momentum=float(cfg.get("spdnet_momentum", 0)),
    )
    scheduler = None
    if str(cfg.get("lr_scheduler", "none")).lower() in {"multistep", "multi_step", "multisteplr"}:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.get("lr_scheduler_milestones", []),
            gamma=float(cfg.get("lr_scheduler_gamma", 0.5)),
        )
    loader = spd.make_loader(x, y, indices, int(cfg.get("batch_size", 30)),
                             int(cfg.get("num_workers", 0)), True, dtype)
    criterion, history = torch.nn.CrossEntropyLoss(), []
    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        lr = float(optimizer.param_groups[0]["lr"])
        loss = spd.train_one_epoch(model, loader, criterion, optimizer, device,
                                   cfg.get("gradient_clip_norm", 5.0), project_stiefel)
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": loss, "learning_rate": lr}
        if epoch == 1 or epoch == epochs or epoch % max(1, int(cfg.get("log_every", 10))) == 0:
            metrics = spd.evaluate(model, loader, criterion, device)
            row["train_accuracy"] = metrics["accuracy"]
            print(f"  {path.name} {epoch}/{epochs}: loss={loss:.4f}, "
                  f"train accuracy={metrics['accuracy']:.4f}, lr={lr:.3g}; checkpoint=last")
        else:
            row["train_accuracy"] = None
        history.append(row)
        spd.write_csv(path / "train_history.csv", history)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def run_formal_experiment(run_index, experiment_cfg, model_cfg, args, base_output_dir, device):
    cfg = deepcopy(experiment_cfg)
    cfg["model"] = deepcopy(model_cfg)
    training = cfg["training"]
    protocol = training["protocol"]
    if protocol not in {"global", "pretrain_finetune"}:
        raise ValueError(f"Unknown SPDNet formal protocol: {protocol}")
    validate_final_epoch_config(training, "training")
    fine = cfg.get("fine_tune", {})
    if protocol == "pretrain_finetune":
        validate_final_epoch_config(fine, "fine_tune")
        if fine.get("split_strategy") != "leave_one_run_out":
            raise ValueError("SPDNet formal fine-tuning requires leave_one_run_out.")
        if spd.resolve_precision(fine.get("precision", training.get("precision"))) != spd.resolve_precision(training.get("precision")):
            raise ValueError("Pretrain/fine-tune precision must match.")
    x_spd, y, subjects, runs, names = load_spd_like_train(cfg["data"], return_runs=True)
    requested = parse_subjects(cfg["data"].get("subjects"))
    if requested:
        expected = {format_subject_id(s, cfg["data"].get("dataset", "physionet_mi")) for s in requested}
        if expected != set(subjects):
            raise ValueError(f"Loaded subject cohort differs from requested cohort: {expected ^ set(subjects)}")
    x = spd.single_trial_covariances(x_spd, model_cfg.get("token_pooling", "original"),
                                      float(cfg["data"].get("eps", 1e-6)))
    del x_spd
    dtype = spd.resolve_precision(training.get("precision", "float64"))
    seed = int(training.get("seed", 42))
    dims = spd.parse_dims(getattr(args, "dims", None) or model_cfg.get("dims"), x.shape[-1])
    model_options = {
        "dims": dims, "num_classes": len(names),
        "reig_epsilon": getattr(args, "reig_epsilon", None) or model_cfg.get("reig_epsilon", 1e-4),
        "log_epsilon": getattr(args, "log_epsilon", None) or model_cfg.get("log_epsilon", 1e-6),
    }
    project = parse_bool(model_cfg.get("project_stiefel", True)) and not getattr(args, "disable_stiefel_projection", False)
    cfg["model"].update(model_options, project_stiefel=project)
    run_dir = Path(base_output_dir) / f"run_{run_index:03d}_{config_hash(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", cfg)
    rows, splits, predictions, global_rows = [], [], [], []

    def new_model(fold_seed, state=None):
        spd.set_seed(fold_seed)
        model = spd.SPDNetClassifier(**model_options).to(device=device, dtype=dtype)
        if state is not None:
            model.load_state_dict(state, strict=True)
        return model

    def fit(model, indices, stage, path, filename):
        if set(np.unique(y[indices])) != set(range(len(names))):
            raise ValueError(f"Training split at {path} lacks classes.")
        state = fit_last(model, x, y, indices, stage, device=device, dtype=dtype,
                         project_stiefel=project, path=path)
        torch.save({"model_state_dict": state, "epoch": int(stage["epochs"]),
                    "checkpoint_selection": "last", "model_config": cfg["model"],
                    "class_names": names, "train_indices": indices.tolist()}, path / filename)
        return state

    def test(model, indices, stage):
        loader = spd.make_loader(x, y, indices, int(stage.get("batch_size", 30)),
                                 int(stage.get("num_workers", 0)), False, dtype)
        return spd.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)

    def record(metrics, indices, identity, train_indices):
        row = {**identity, "n_train": len(train_indices), "n_validation": 0, "n_test": len(indices),
               **{key: float(metrics[key]) for key in ("accuracy", "macro_f1", "cohen_kappa", "loss")}}
        rows.append(row)
        for i, truth, prediction in zip(indices, metrics["y_true"], metrics["y_pred"]):
            predictions.append({**identity, "subject": str(subjects[i]), "trial_index": int(i),
                                "run": int(runs[i]), "y_true": int(truth), "y_pred": int(prediction)})
        splits.append({**identity, "train_indices": train_indices.tolist(),
                       "validation_indices": [], "test_indices": indices.tolist()})
        spd.write_csv(run_dir / "fold_results.csv", rows)
        spd.write_csv(run_dir / "test_predictions.csv", predictions)
        save_json(run_dir / "splits.json", splits)
        print(f"  {identity}: test accuracy={row['accuracy']:.4f}, n_test={len(indices)}")

    if protocol == "global":
        folds = make_subject_folds(subjects, int(training.get("n_splits", 5)),
                                   shuffle=parse_bool(training.get("cv_shuffle", True)), seed=seed)
        for fold, (source, target, train, held) in enumerate(folds, 1):
            path = run_dir / f"fold_{fold:02d}"
            print(f"SPDNet global fold {fold}: {len(source)} train / {len(target)} test subjects; validation=0")
            model = new_model(seed + fold)
            fit(model, train, training, path, "global_last.pt")
            record(test(model, held, training), held, {"fold": fold}, train)
            del model
    else:
        target_value = getattr(args, "target_subjects", None) or fine.get("target_subjects", "all")
        targets = (sorted(set(subjects)) if str(target_value).lower() == "all" else
                   [format_subject_id(s, cfg["data"].get("dataset", "physionet_mi")) for s in parse_subjects(target_value)])
        if not targets or set(targets) - set(subjects):
            raise ValueError(f"Missing or empty target subjects: {targets}")
        # Validate every target/run before starting expensive pretraining.
        mask = np.isin(subjects, targets)
        original_indices = np.flatnonzero(mask)
        local = make_subject_specific_loro_splits(y[mask], subjects[mask], runs[mask],
                                                  seed=seed, held_out_run_validation_size=0)
        folds = [(s, r, original_indices[tr], original_indices[te]) for s, r, tr, _, te in local]
        for target in targets:
            source = np.flatnonzero(subjects != target)
            held = np.flatnonzero(subjects == target)
            target_seed = seed + int(target[1:])
            path = run_dir / target
            print(f"SPDNet {target}: pretrain={len(source)} other-subject trials; target excluded; validation=0")
            model = new_model(target_seed)
            state = fit(model, source, training, path, "pretrained_last.pt")
            save_json(path / "pretrain_split.json", {"excluded_target_subject": target,
                "train_indices": source.tolist(), "validation_indices": [], "test_indices": []})
            global_metrics = test(model, held, training)
            global_rows.append({"subject": target, "accuracy": float(global_metrics["accuracy"]), "n_test": len(held)})
            spd.write_csv(run_dir / "global_before_finetune.csv", global_rows)
            del model
            for fold, (_, run, train, held_run) in enumerate([f for f in folds if f[0] == target], 1):
                model = new_model(target_seed * 10000 + fold, state)
                fold_path = path / f"test_run_{run:02d}"
                fit(model, train, fine, fold_path, "fine_tuned_last.pt")
                record(test(model, held_run, fine), held_run, {"subject": target, "test_run": run}, train)
                del model
            del state
        spd.write_csv(run_dir / "per_run_results.csv", rows)

    subject_rows = summarize_subject_fold_metrics([
        {"subject": row["subject"], "_y_true": [row["y_true"]], "_y_pred": [row["y_pred"]]}
        for row in predictions
    ], ("accuracy", "macro_f1", "cohen_kappa"))
    spd.write_csv(run_dir / "per_subject_summary.csv", subject_rows)
    summary = {"baseline": "spdnet", "protocol": protocol, "validation_used": False,
               "checkpoint_selection": "last", "class_names": names, "folds": rows,
               "mean_subject_accuracy": statistics.fmean(r["Accuracy (%)"] / 100 for r in subject_rows),
               "mean_fold_accuracy": statistics.fmean(r["accuracy"] for r in rows),
               "pooled_trial_accuracy": statistics.fmean(r["y_true"] == r["y_pred"] for r in predictions)}
    save_json(run_dir / "summary.json", summary)
    print(f"SPDNet mean subject accuracy={summary['mean_subject_accuracy']:.4f}; results={run_dir}")
    return {"run_index": run_index, "run_dir": str(run_dir), **summary}
