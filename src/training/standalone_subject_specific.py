"""Independent adaptation jobs referencing, never modifying, an existing suite."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.training import paper_suite_results as report
from src.training import train_fold_checkpoint_finetune as reuse
from src.training.train_global_cross_subject_cv import ROOT, read_yaml, source_hash, write_csv, write_json
from src.baselines import mdm_baseline as mdm


def select_global(manifest, model, rest):
    jobs = [j for j in manifest["jobs"] if j["model"] == model and j["protocol"] == "global"
            and float(j["rest_length_s"]) == rest
            and (model != "transformer" or j["variant"] == "complete")]
    if len(jobs) != 1:
        raise ValueError(f"Need exactly one {model} global job at rest={rest}, found {len(jobs)}.")
    return copy.deepcopy(jobs[0])


def audit_global(job, n_splits):
    rows = report.load_predictions(job, n_splits)
    root = report.unique_file(job["output"], "test_predictions.csv").parent
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if len(summary.get("class_names", [])) < 2:
        raise ValueError(f"Missing completed global summary: {root}")
    if job["model"] in {"transformer", "spdnet"}:
        reuse.source_artifacts(job, n_splits)
    return rows, summary["class_names"]


def fine_config(manifest, model, rest, override=None):
    if override:
        path = Path(override).resolve()
    else:
        jobs = [j for j in manifest["jobs"] if j["model"] == model and j["protocol"] == "adaptation"
                and float(j["rest_length_s"]) == rest]
        if len(jobs) != 1:
            raise ValueError("No unique frozen adaptation config; supply --fine-tune-config YAML.")
        path = Path(jobs[0]["config"])
        if reuse.file_hash(path) != jobs[0]["config_sha256"]:
            raise ValueError(f"Frozen adaptation config changed: {path}")
    raw = read_yaml(path)
    if "fine_tune" not in raw:
        raise ValueError("Fine-tune YAML needs a fine_tune section.")
    return copy.deepcopy(raw["fine_tune"]), {"path": str(path), "sha256": reuse.file_hash(path)}


def prepare(model, args):
    directory = args.global_suite.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rest = float(args.rest_length if args.rest_length is not None else manifest["suite"]["comparison_rest"])
    source = select_global(manifest, model, rest)
    n_splits = int(manifest["suite"]["n_splits"])
    # A final job can finish writing artifacts immediately before Ctrl+C prevented
    # the parent from marking it completed. Audit artifacts, not the status alone.
    rows, names = audit_global(source, n_splits)
    output = (args.output_dir or directory / "fold_adaptation" / model).resolve()
    config = {"source_global": source, "source_n_splits": n_splits,
              "reuse_target_subjects": args.target_subjects,
              "output": {"dir": str(output / "results"), "save_fine_tuned_checkpoints": True}}
    provenance = {"global_suite_manifest": str(manifest_path),
                  "global_suite_manifest_sha256": reuse.file_hash(manifest_path),
                  "global_source_sha256": manifest.get("source_sha256"),
                  "adaptation_source_sha256": source_hash(),
                  "global_job_status_at_import": source.get("status"),
                  "global_config_sha256": source["config_sha256"], "class_names": names}
    if model != "mdm":
        config["fine_tune"], provenance["fine_tune_config"] = fine_config(manifest, model, rest, args.fine_tune_config)
        if args.epochs is not None:
            config["fine_tune"]["epochs"] = [args.epochs] if model == "transformer" else args.epochs
    elif args.epochs is not None or args.fine_tune_config:
        raise ValueError("Classical subject-specific MDM has no epochs or fine-tune checkpoint.")
    config["provenance"] = provenance
    job = {"id": f"{model}_global_folds_rest_{rest:g}_subject_specific", "model": model,
           "variant": "complete" if model == "transformer" else "baseline", "rest_length_s": rest,
           "protocol": "subject_specific" if model == "mdm" else "adaptation",
           "adaptation_mode": "target_run_refit" if model == "mdm" else "global_fold_checkpoint",
           "config": str(output / "config.yaml"), "output": config["output"]["dir"],
           "status": "pending", "source_global_job": source["id"]}
    print(f"{model}: import {source['id']}; {n_splits} global folds, {len(rows)} audited trials; "
          f"targets={args.target_subjects}; new neural pretrain calls=0.")
    print(f"New output: {output}; original manifest/configs/results remain unchanged.")
    return config, job, output


def validate_loaded(rows, names, y, subjects, runs, loaded_names):
    if names != loaded_names or len(rows) != len(y) or {r["trial_index"] for r in rows} != set(range(len(y))):
        raise ValueError("Loaded cohort/class order differs from the original global trial identities.")
    for row in rows:
        i = row["trial_index"]
        if row["subject"] != str(subjects[i]) or row["y_true"] != int(y[i]):
            raise ValueError(f"Global subject/label identity changed at trial {i}.")
        if row["run"] != "" and int(row["run"]) != int(runs[i]):
            raise ValueError(f"Global run identity changed at trial {i}.")


def run_mdm(config):
    from pyriemann.classification import MDM

    source = config["source_global"]
    rows, names = audit_global(source, int(config["source_n_splits"]))
    candidates = mdm.expand_mdm_experiments(read_yaml(source["config"]))
    if len(candidates) != 1:
        raise ValueError("Expected one frozen MDM global configuration.")
    cfg = candidates[0]
    if mdm.resolve_classifier_type(cfg["model"]) != "pyriemann":
        raise ValueError("This subject-specific entry supports classical pyriemann MDM only.")
    cache = Path(cfg["output"].get("dataset_cache_dir", "experiments/cache/mdm_preprocessed_datasets"))
    x, y, subjects, runs, loaded_names = mdm.load_or_preprocess_spd(
        cfg["data"], cache if cache.is_absolute() else ROOT / cache, {})
    validate_loaded(rows, names, y, subjects, runs, loaded_names)
    targets = reuse.resolve_targets(config, subjects, cfg["data"])
    local_indices = np.flatnonzero(np.isin(subjects, targets))
    local = reuse.utils.make_subject_specific_loro_splits(y[local_indices], subjects[local_indices], runs[local_indices],
                                                        seed=int(cfg["training"].get("seed", 42)), held_out_run_validation_size=0)
    x, _ = mdm.pool_spd_tokens(x, cfg["model"], eps=float(cfg["data"].get("eps", 1e-6)))
    metric = mdm.resolve_mdm_metric(cfg["model"], cli_metric=None, cli_mean_metric=None, cli_distance_metric=None)
    out = Path(config["output"]["dir"])
    out.mkdir(parents=True, exist_ok=False)
    source_fold = {r["subject"]: r["fold"] for r in rows}
    predictions, splits, metrics = [], [], []
    for subject, run, train, _, test in local:
        train, test = local_indices[train], local_indices[test]
        classifier = MDM(metric=metric, n_jobs=int(cfg["model"].get("n_jobs", 1)))
        classifier.fit(x[train], y[train])
        predicted = classifier.predict(x[test])
        fold_rows = [{"subject": subject, "run": int(run), "source_cv_fold": source_fold[subject],
                      "trial_index": int(i), "y_true": int(y[i]), "y_pred": int(p)}
                     for i, p in zip(test, predicted, strict=True)]
        predictions.extend(fold_rows)
        splits.append({"subject": subject, "test_run": int(run), "source_cv_fold": source_fold[subject],
                       "train_indices": train.tolist(), "test_indices": test.tolist(), "validation_indices": []})
        metrics.append({"subject": subject, "test_run": int(run), "source_cv_fold": source_fold[subject],
                        **report.scores(fold_rows, list(range(len(names))))})
        write_csv(out / "test_predictions.csv", predictions)
        write_csv(out / "per_run_results.csv", metrics)
        write_json(out / "splits.json", splits)
        print(f"MDM fold {source_fold[subject]} {subject} run {run}: target-only refit, accuracy={metrics[-1]['accuracy_pct']:.2f}%")
    write_csv(out / "per_subject_summary.csv", [{"subject": s, **report.scores(
        [r for r in predictions if r["subject"] == s], list(range(len(names))))} for s in targets])
    write_json(out / "summary.json", {"class_names": names, "adaptation_mode": "target_run_refit",
                                      "global_checkpoint_reused": False, "n_target_subjects": len(targets)})
    return 0


def audit_reuse(job):
    """Verify run-LOO predictions and their immutable source global folds."""
    cfg = read_yaml(job["config"])
    source, n = cfg["source_global"], int(cfg["source_n_splits"])
    global_rows, names = audit_global(source, n)
    records = report.load_predictions(job, n)
    index = {r["trial_index"]: r for r in global_rows}
    for row in records:
        ref = index.get(row["trial_index"])
        if ref is None or (ref["subject"], ref["y_true"]) != (row["subject"], row["y_true"]):
            raise ValueError("Adaptation predictions do not match the source trial identities.")
        if ref["run"] != "" and int(ref["run"]) != int(row["run"]):
            raise ValueError("Adaptation run does not match the source global run.")
    cohort = np.asarray(sorted({r["subject"] for r in global_rows}))
    source_cfg = read_yaml(source["config"])
    data = reuse.utils.expand_data_training_experiments(source_cfg)[0]["data"]
    targets = reuse.resolve_targets(cfg, cohort, data)
    if {r["trial_index"] for r in records} != {r["trial_index"] for r in global_rows if r["subject"] in targets}:
        raise ValueError("Incomplete adaptation target/trial coverage.")
    result_dir = Path(job["output"])
    if job["model"] != "mdm":
        _, _, artifacts = reuse.source_artifacts(source, n)
        for subject in targets:
            split = json.loads((result_dir / subject / "pretrain_split.json").read_text(encoding="utf-8"))
            fold = next(r["fold"] for r in global_rows if r["subject"] == subject)
            expected_train = {r["trial_index"] for r in global_rows if r["fold"] != fold}
            artifact = artifacts[fold]
            if (split.get("source_cv_fold") != fold or set(split["train_indices"]) != expected_train
                    or split.get("source_checkpoint_sha256") != artifact["sha256"]
                    or Path(split["source_checkpoint"]).resolve() != Path(artifact["path"]).resolve()):
                raise ValueError("Adaptation did not reuse exactly the corresponding global-fold checkpoint.")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("class_names") != names:
        raise ValueError("Adaptation class names/order differ from global results.")
    return records


def main(model, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-subjects", default="all")
    parser.add_argument("--rest-length", type=float)
    parser.add_argument("--fine-tune-config", type=Path)
    parser.add_argument("--epochs", type=int, help="Override fine-tune epochs only.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config, job, output = prepare(model, args)
    if args.dry_run:
        return 0
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Existing standalone results at {output}; use a new --output-dir, never overwrite them.")
    output.mkdir(parents=True, exist_ok=True)
    Path(job["config"]).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    job["config_sha256"] = reuse.file_hash(job["config"])
    job["provenance"] = config["provenance"]
    try:
        job["status"] = "running"
        write_json(output / "job.json", job)
        run_mdm(config) if model == "mdm" else reuse.run(config, torch.device(args.device))
        audit_reuse(job)
        job["status"] = "completed"
    except BaseException as error:
        job.update(status="failed", error=str(error) or type(error).__name__)
        raise
    finally:
        write_json(output / "job.json", job)
    return 0
