"""Collect old global results and new standalone adaptation without editing either."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import paper_suite_results as report
from src.training import standalone_subject_specific as standalone
from src.training.train_global_cross_subject_cv import write_json


def collect(global_suite, adaptation_dir, output):
    manifest_path = global_suite / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    global_jobs = [copy.deepcopy(j) for j in original["jobs"] if j["protocol"] == "global"]
    if not global_jobs:
        raise ValueError("No global jobs in the source suite.")
    n_splits = int(original["suite"]["n_splits"])
    for job in global_jobs:
        report.load_predictions(job, n_splits)
        root = report.unique_file(job["output"], "test_predictions.csv").parent
        if not (root / "summary.json").is_file():
            raise ValueError(f"Global job has no completed summary: {job['id']}")
        job["original_status"] = job["status"]
        job["status"] = "completed"
    new_jobs = []
    target_cohorts = []
    for model in ("transformer", "spdnet", "mdm"):
        job = json.loads((adaptation_dir / model / "job.json").read_text(encoding="utf-8"))
        if job["model"] != model or job["status"] != "completed":
            raise ValueError(f"Standalone {model} is incomplete or mislabeled.")
        cfg = standalone.read_yaml(job["config"])
        expected = standalone.select_global(original, model, float(job["rest_length_s"]))
        source = cfg["source_global"]
        for key in ("id", "output", "config", "config_sha256"):
            if source[key] != expected[key]:
                raise ValueError(f"{model}: standalone source is not this suite's global job.")
        if float(job["rest_length_s"]) != float(original["suite"]["comparison_rest"]):
            raise ValueError("Standalone rest length differs from the suite comparison setting.")
        records = standalone.audit_reuse(job)
        target_cohorts.append({r["subject"] for r in records})
        new_jobs.append(job)
    if not all(cohort == target_cohorts[0] for cohort in target_cohorts):
        raise ValueError("Standalone models used different target cohorts; do not compare their subject means.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Combined output is not empty: {output}; choose a new --output-dir.")
    output.mkdir(parents=True, exist_ok=True)
    merged = {"suite": original["suite"], "jobs": global_jobs + new_jobs,
              "global_source_sha256": original.get("source_sha256"),
              "global_manifest": str(manifest_path),
              "global_manifest_sha256": standalone.reuse.file_hash(manifest_path),
              "note": "Imported global artifacts; separate standalone adaptation, original manifest unmodified."}
    report.collect_suite(merged, output)
    write_json(output / "combined_manifest.json", merged)
    print(f"Combined CSVs: {output / 'tables'}; old suite remains unchanged.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-suite", type=Path, required=True)
    parser.add_argument("--adaptation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    source = args.global_suite.resolve()
    adaptation = (args.adaptation_dir or source / "fold_adaptation").resolve()
    return collect(source, adaptation, (args.output_dir or adaptation / "combined").resolve())


if __name__ == "__main__":
    raise SystemExit(main())
