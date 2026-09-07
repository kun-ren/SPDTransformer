"""Run frozen PhysioNet main/ablation experiments and collect publication CSVs.

This branch's entry point orchestrates existing runners. It does not select
hyperparameters from test scores. Target-run adaptation is reported separately
because its train/test subjects overlap, unlike global subject-wise CV.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/physionet_paper_suite.yaml"
VARIANTS = {
    "complete": "Complete model",
    "fixed_logeuclidean": "Fixed Log-Euclidean attention metric",
    "linear_head": "Linear classification head",
    "single_band": "Single 8-30 Hz band",
}
MODELS = {"transformer": "SPD Transformer", "mdm": "MDM", "spdnet": "SPDNet", "csp": "CSP+LDA"}


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rest_tag(rest):
    return f"{rest:g}".replace(".", "p")


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def plan_jobs(suite, directory, *, subjects=None, include_adaptation=None):
    """Generate explicit configs; preserve singleton-grid syntax of each runner."""
    bases = {key: read_yaml(ROOT / suite[key]) for key in (
        "transformer_global", "transformer_finetune", "mdm_global", "mdm_subject_specific",
        "spdnet_global", "spdnet_finetune", "csp_global",
    )}
    ref = bases["transformer_global"]["data"]
    rests = [float(r) for r in suite["rest_lengths"]]
    if len(set(rests)) != len(rests) or set(rests) != {0., 1.5}:
        raise ValueError("This paper suite requires rest_lengths=[0.0, 1.5].")
    if int(suite["n_splits"]) < 2 or float(suite["epoch_end"]) <= 0:
        raise ValueError("Need >=2 subject folds and a positive epoch_end.")
    variants = suite["variants"]
    if len(variants) != len(set(variants)) or not set(variants) <= set(VARIANTS) or "complete" not in variants:
        raise ValueError("Variants must be unique supported names, including complete.")
    if float(suite["comparison_rest"]) not in rests:
        raise ValueError("comparison_rest must be one of rest_lengths.")
    jobs = []

    def add(model, variant, rest, mode, base_key, script):
        cfg = copy.deepcopy(bases[base_key])
        # Match every applicable preprocessing field, not only the epoch window.
        for key in list(cfg["data"]):
            if key in ref and key not in {"brain_region_mode", "pretrain_subjects"}:
                cfg["data"][key] = copy.deepcopy(ref[key])
        cfg["data"]["subjects"] = subjects or copy.deepcopy(ref["subjects"])
        cfg["data"]["epoch_slice"] = [[-rest, float(suite["epoch_end"])]]
        cfg["data"]["segment_slice"] = copy.deepcopy(ref["segment_slice"])
        cfg["data"]["filter_bank"] = copy.deepcopy(ref["filter_bank"])
        if mode == "global":
            tr = cfg["training"]
            tr["seed"] = [int(suite["seed"])]
            if model == "transformer":
                tr.update(cv_n_splits=[int(suite["n_splits"])], cv_seed=[int(suite["seed"])], cv_shuffle=[True])
                tr.update(use_validation=[False], checkpoint_selection=["last"])
                if variant == "fixed_logeuclidean":
                    cfg["model"]["metric"] = ["log-euclidean"]
                elif variant == "linear_head":
                    cfg["model"]["classifier_type"] = ["pooling"]
                    tr.update(prototype_intra_weight=[0.0], prototype_inter_weight=[0.0])
                elif variant == "single_band":
                    cfg["data"]["filter_bank"] = [[[8, 30]]]
            else:
                tr.update(n_splits=[int(suite["n_splits"])], cv_shuffle=[True])
                if model == "spdnet":
                    tr.update(protocol=["global"], use_validation=[False], checkpoint_selection=["last"])
                else:
                    tr.update(subject_specific=[False], allow_subject_overlap=[False],
                              subject_fold_method=["kfold"], val_size=[0.0], test_size=[1 / int(suite["n_splits"])])
        elif model == "transformer":
            cfg["data"]["pretrain_subjects"] = cfg["data"]["subjects"]
            targets = suite.get("target_subjects", "all")
            if str(targets).lower() != "all":
                cfg["data"]["subjects"] = targets
            cfg["pretrain"].update(seed=[int(suite["seed"])], use_validation=[False],
                                    validation_size=[0.0], test_size=[0.0], checkpoint_selection=["last"])
            cfg["fine_tune"].update(split_strategy=["leave_one_run_out"], use_validation=[False], checkpoint_selection=["last"])
        elif model == "spdnet":
            cfg["training"].update(protocol=["pretrain_finetune"], seed=[int(suite["seed"])])
            cfg["fine_tune"]["target_subjects"] = suite.get("target_subjects", "all")
        else:
            targets = suite.get("target_subjects", "all")
            if str(targets).lower() != "all":
                cfg["data"]["subjects"] = targets
            cfg["training"].update(subject_specific=[True], subject_split_strategy=["leave_one_run_out"],
                                      held_out_run_validation_size=[0.0], val_size=[0.0])
        job_id = f"{model}_{variant}_rest_{rest_tag(rest)}_{mode}"
        from src.training.config_grid import expand_data_grid, expand_grid
        for section, expand in (("data", expand_data_grid), ("training", expand_grid), ("pretrain", expand_grid)):
            if section in cfg and len(expand(cfg[section])) != 1:
                raise ValueError(f"{job_id}: {section} contains a hyperparameter grid; freeze it first.")
        if model in {"transformer", "mdm"} and len(expand_grid(cfg["model"])) != 1:
            raise ValueError(f"{job_id}: model contains multiple candidates; freeze hyperparameters first.")
        cfg["output"]["dir"] = str(directory / "runs" / job_id)
        config_path = directory / "configs" / f"{job_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        jobs.append({"id": job_id, "model": model, "variant": variant, "rest_length_s": rest,
                     "protocol": mode, "script": script, "config": str(config_path),
                     "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                     "output": cfg["output"]["dir"], "status": "pending"})

    for rest in rests:
        for variant in variants:
            add("transformer", variant, rest, "global", "transformer_global", "src/training/train_global_cross_subject.py")
        for model in ("mdm", "spdnet", "csp"):
            add(model, "baseline", rest, "global", model + "_global", f"src/baselines/{'csp_lda' if model == 'csp' else model}_baseline.py")
    adaptation = suite.get("include_adaptation", True) if include_adaptation is None else include_adaptation
    if adaptation:
        rest = float(suite["comparison_rest"])
        add("transformer", "complete", rest, "adaptation", "transformer_finetune", "src/training/train_pretrain_finetune_loro.py")
        add("spdnet", "baseline", rest, "adaptation", "spdnet_finetune", "src/baselines/spdnet_baseline.py")
        add("mdm", "baseline", rest, "subject_specific", "mdm_subject_specific", "src/baselines/mdm_baseline.py")
    return jobs


def execute_job(job, directory, device):
    config_path = Path(job["config"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != job["config_sha256"]:
        raise ValueError(f"Generated config changed: {config_path}; start a new suite.")
    output = Path(job["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to mix a retry with partial results at {output}. "
                         "Use a new suite directory for failed experiments.")
    command = [sys.executable, "-u", str(ROOT / job["script"]), "--config", str(config_path)]
    if job["model"] in {"transformer", "spdnet"}:
        command += ["--device", device]
    if job["model"] == "mdm":
        command += ["--fail-fast"]
    logfile = directory / "logs" / f"{job['id']}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    job.update(command=command, log=str(logfile))
    print(f"Running {job['id']}; log={logfile}", flush=True)
    with logfile.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--subjects", help="Override entire cohort for a smoke test, e.g. 1-10.")
    parser.add_argument("--target-subjects", help="Adaptation targets only; global CV cohort stays unchanged.")
    parser.add_argument("--global-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write all configs and manifest, without training.")
    parser.add_argument("--resume", action="store_true", help="Use frozen manifest; skip completed jobs.")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild tables from a completed manifest.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.resume or args.collect_only:
        if args.suite_dir is None:
            raise ValueError("--resume/--collect-only requires --suite-dir.")
        directory = args.suite_dir.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if args.subjects or args.target_subjects or args.global_only:
            raise ValueError("Resume uses frozen configs; overrides require a new suite directory.")
    else:
        suite = read_yaml(args.config)
        if args.target_subjects:
            suite["target_subjects"] = args.target_subjects
        directory = (args.suite_dir or ROOT / suite["output_dir"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Suite directory is not empty: {directory}; use --resume.")
        directory.mkdir(parents=True, exist_ok=True)
        jobs = plan_jobs(suite, directory, subjects=args.subjects,
                         include_adaptation=False if args.global_only else None)
        from importlib.metadata import PackageNotFoundError, version
        versions = {}
        for package in ("numpy", "scipy", "scikit-learn", "torch", "mne", "pyriemann", "geoopt"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not installed"
        manifest = {"suite": suite, "source_sha256": source_hash(), "jobs": jobs,
                    "python": sys.version, "package_versions": versions}
        write_json(directory / "manifest.json", manifest)
    for job in manifest["jobs"]:
        print(f"  {job['id']}: {job['status']}")
    print(f"{len(manifest['jobs'])} experiment jobs; global jobs each train {manifest['suite']['n_splits']} subject folds.")
    if args.dry_run:
        print(f"Plan saved: {directory}. No data loaded or models trained.")
        return 0
    if not args.collect_only:
        if source_hash() != manifest["source_sha256"]:
            raise ValueError("Source files changed since planning; start a new suite to avoid mixing model versions.")
        for job in manifest["jobs"]:
            if job["status"] == "completed":
                continue
            try:
                job["status"] = "running"
                write_json(directory / "manifest.json", manifest)
                execute_job(job, directory, args.device)
                job["status"] = "completed"
            except Exception as error:
                job.update(status="failed", error=str(error))
                raise
            finally:
                write_json(directory / "manifest.json", manifest)
    from src.training.paper_suite_results import collect_suite
    collect_suite(manifest, directory)
    print(f"Publication CSVs saved: {directory / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
