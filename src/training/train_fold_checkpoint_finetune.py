"""Fine-tune each held-out subject from its immutable global-fold checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.training import train_pretrain_finetune_loro as core
from src.training import train_global_cross_subject as global_train
from src.training import paper_suite_results as report
from src.training.train_global_cross_subject_cv import read_yaml, write_json, write_csv
from src.training.config_grid import expand_grid
from src.baselines import baseline_utils as utils, spdnet_baseline as spd
from src.baselines.spdnet_formal import fit_last
from src.models.MotorImageryDataset import MotorImageryDataset


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_artifacts(source, n_splits):
    """Require complete out-of-fold predictions and every final checkpoint."""
    rows = report.load_predictions(source, n_splits)
    root = report.unique_file(source["output"], "test_predictions.csv").parent
    names = json.loads((root / "summary.json").read_text(encoding="utf-8"))["class_names"]
    checkpoints = {}
    for fold in range(1, n_splits + 1):
        path = root / f"fold_{fold:02d}" / "global_last.pt"
        if not path.is_file():
            raise ValueError(f"Missing global checkpoint: {path}. Cannot reuse this fold.")
        checkpoints[fold] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    return rows, names, checkpoints


def validate_checkpoint(checkpoint, train_indices, names, epoch):
    if checkpoint.get("checkpoint_selection") != "last" or checkpoint.get("epoch") != epoch:
        raise ValueError("Expected the configured final-epoch global checkpoint.")
    if checkpoint.get("class_names") != names:
        raise ValueError("Checkpoint class order differs from global predictions.")
    if set(checkpoint.get("train_indices", [])) != set(train_indices):
        raise ValueError("Checkpoint training indices differ from the global subject fold.")


def resolve_targets(config, subjects, data):
    value = config.get("reuse_target_subjects", "all")
    targets = (sorted(set(subjects)) if str(value).lower() == "all" else
               [core.format_subject_id(s, data["dataset"]) for s in (core.parse_subjects(value) or [])])
    if not targets or set(targets) - set(subjects):
        raise ValueError("Requested adaptation subjects are missing from the global cohort.")
    return targets


def run(config, device):
    source, n_splits = config["source_global"], int(config["source_n_splits"])
    kind = source["model"]
    rows, names, artifacts = source_artifacts(source, n_splits)
    raw_source = read_yaml(source["config"])
    if kind == "transformer":
        source_cfg = global_train.load_config(Path(source["config"]))
        candidates = expand_grid(config["fine_tune"])
        if len(candidates) != 1:
            raise ValueError("Freeze fine_tune hyperparameters before checkpoint reuse.")
        fine = candidates[0]
        data = core.normalize_data_config(source_cfg["data"])
        cache = Path(source_cfg["output"].get("dataset_cache_dir", "experiments/cache/preprocessed_datasets"))
        x, y, subjects, runs, loaded_names = core.load_or_preprocess_spd_with_runs(
            data, cache if cache.is_absolute() else ROOT / cache)
    elif kind == "spdnet":
        candidates = utils.expand_data_training_experiments(raw_source)
        if len(candidates) != 1:
            raise ValueError("Expected one frozen global SPDNet experiment.")
        source_cfg = candidates[0]
        source_cfg["model"] = raw_source["model"]
        fine, data = config["fine_tune"], source_cfg["data"]
        x, y, subjects, runs, loaded_names = utils.load_spd_like_train(data, return_runs=True)
        x = spd.single_trial_covariances(x, source_cfg["model"].get("token_pooling", "original"),
                                         float(data.get("eps", 1e-6)))
    else:
        raise ValueError(f"Unsupported neural model: {kind}")
    core.validate_final_epoch_config(fine, "fine_tune")
    if fine.get("split_strategy") != "leave_one_run_out":
        raise ValueError("Checkpoint adaptation requires complete run holdout.")
    dtype = core.resolve_precision(source_cfg["training"].get("precision", "float32"))
    if core.resolve_precision(fine.get("precision", source_cfg["training"].get("precision"))) != dtype:
        raise ValueError("Fine-tune precision must match the source checkpoint.")
    if names != loaded_names or len(rows) != len(y) or {r["trial_index"] for r in rows} != set(range(len(y))):
        raise ValueError("Loaded trial count/class order differs from the global experiment.")
    for row in rows:
        i = row["trial_index"]
        if (str(subjects[i]), int(y[i]), str(int(runs[i]))) != (row["subject"], row["y_true"], str(row["run"])):
            raise ValueError(f"Global trial identity changed at index {i}.")
    targets = resolve_targets(config, subjects, data)
    mask = np.isin(subjects, targets)
    indices = np.flatnonzero(mask)
    local = utils.make_subject_specific_loro_splits(y[mask], subjects[mask], runs[mask],
                                                    seed=42, held_out_run_validation_size=0)
    folds = [(s, r, indices[tr], indices[te]) for s, r, tr, _, te in local]
    output = Path(config["output"]["dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing fine-tune results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "source_checkpoints.json", artifacts)
    write_json(output / "config.json", config)
    if device.type == "cuda":
        tf32 = core.parse_bool(fine.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    dataset = MotorImageryDataset(x, y, dtype=dtype) if kind == "transformer" else None
    predictions, splits, metrics_rows, global_rows = [], [], [], []
    source_rows = {r["trial_index"]: r for r in rows}
    save_models = core.parse_bool(config["output"].get("save_fine_tuned_checkpoints", True))
    for fold, artifact in artifacts.items():
        held = [r["trial_index"] for r in rows if r["fold"] == fold]
        source_indices = sorted(set(range(len(y))) - set(held))
        fold_targets = sorted(set(subjects[held]) & set(targets))
        if not fold_targets:
            continue
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Source checkpoint changed while preparing adaptation.")
        checkpoint = torch.load(artifact["path"], map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, source_indices, names, int(source_cfg["training"]["epochs"]))
        if kind == "transformer" and checkpoint.get("input_token_shape") != list(x.shape[1:]):
            raise ValueError("Global checkpoint token shape differs from reloaded input.")
        state = checkpoint["model_state_dict"]

        def new_model(seed):
            core.set_seed(seed)
            if kind == "transformer":
                mapping = checkpoint.get("domain_subject_mapping", {})
                model = core.make_model(checkpoint["model_config"], x, len(names), device=device,
                                        dtype=dtype, num_domains=len(mapping) or None)
                model.load_state_dict(state, strict=True)
                model.set_domain_head_trainable(False)
            else:
                options = checkpoint["model_config"]
                model = spd.SPDNetClassifier(options["dims"], len(names), options["reig_epsilon"],
                                             options["log_epsilon"]).to(device=device, dtype=dtype)
                model.load_state_dict(state, strict=True)
            return model

        print(f"Global fold {fold}: reuse {artifact['path']}; targets={','.join(fold_targets)}; pretrain calls=0")
        for target in fold_targets:
            target_dir = output / target
            target_dir.mkdir()
            provenance = {"source_global_job": source["id"], "source_cv_fold": fold,
                          "source_checkpoint": artifact["path"], "source_checkpoint_sha256": artifact["sha256"]}
            write_json(target_dir / "pretrain_split.json", {
                **provenance, "train_indices": source_indices, "validation_indices": [], "test_indices": [],
                "excluded_subjects": sorted(set(subjects[held])), "pretraining_reused": True,
            })
            # These are already measured global predictions, not another global test pass.
            target_global = [r for r in rows if r["subject"] == target]
            global_rows.append({"subject": target, **provenance, **report.scores(target_global, list(range(len(names))))})
            write_csv(output / "global_before_finetune.csv", global_rows)
            for run_id, train, test in [(r, tr, te) for s, r, tr, te in folds if s == target]:
                path = target_dir / f"test_run_{run_id:02d}"
                path.mkdir()
                model = new_model(int(source_cfg["training"].get("seed", 42)) + int(target[1:]) * 1000 + run_id)
                if kind == "transformer":
                    options = dict(batch_size=int(fine.get("batch_size", 32)),
                                   num_workers=int(fine.get("num_workers", 0)), pin_memory=False)
                    final_state, epoch = core.train_final_epoch(
                        model, core.make_loader(dataset, train, shuffle=True, **options), fine, device=device,
                        history_path=path / "fine_tune_history.csv", stage_name="fine-tune")
                    measured = core.predict_loader(model, core.make_loader(dataset, test, shuffle=False, **options),
                                                    torch.nn.CrossEntropyLoss(), device)
                else:
                    final_state = fit_last(model, x, y, train, fine, device=device, dtype=dtype,
                                           project_stiefel=bool(checkpoint["model_config"].get("project_stiefel", True)), path=path)
                    epoch = int(fine["epochs"])
                    loader = spd.make_loader(x, y, test, int(fine.get("batch_size", 30)), 0, False, dtype)
                    measured = spd.evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)
                split = {"subject": target, "target_subject": target, "test_run": int(run_id),
                         **provenance, "train_indices": train.tolist(), "validation_indices": [],
                         "test_indices": test.tolist(), "epoch": epoch, "checkpoint_selection": "last"}
                write_json(path / "split.json", split)
                splits.append(split)
                if save_models:
                    torch.save({**split, "model_state_dict": final_state, "class_names": names,
                                "model_config": checkpoint["model_config"]}, path / "fine_tuned_last.pt")
                fold_rows = [{"subject": target, "trial_index": int(i), "run": int(run_id),
                              "source_cv_fold": source_rows[int(i)]["fold"], "y_true": int(truth), "y_pred": int(pred)}
                             for i, truth, pred in zip(test, measured["y_true"], measured["y_pred"], strict=True)]
                predictions.extend(fold_rows)
                metrics_rows.append({"subject": target, "test_run": int(run_id), **provenance,
                                     **report.scores(fold_rows, list(range(len(names))))})
                write_csv(output / ("per_trial_results.csv" if kind == "transformer" else "test_predictions.csv"), predictions)
                write_json(output / "splits.json", splits)
                write_csv(output / "per_run_results.csv", metrics_rows)
                print(f"  {target} run {run_id}: accuracy={measured['accuracy']:.4f}, checkpoint=last")
                del model, final_state
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del state, checkpoint
    subject_rows = [{"subject": s, **report.scores([r for r in predictions if r["subject"] == s], list(range(len(names))))}
                    for s in targets]
    write_csv(output / "per_subject_summary.csv", subject_rows)
    write_json(output / "summary.json", {"class_names": names, "adaptation_mode": "global_fold_checkpoint",
                                         "new_pretrain_calls": 0, "checkpoint_selection": "last",
                                         "n_target_subjects": len(targets), "n_test_trials": len(predictions)})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    return run(read_yaml(args.config), torch.device(args.device))


if __name__ == "__main__":
    raise SystemExit(main())
