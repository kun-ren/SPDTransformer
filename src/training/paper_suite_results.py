"""Audit paired held-out predictions and generate publication-ready CSVs."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)

from src.training.train_global_cross_subject_cv import MODELS, VARIANTS, write_csv, write_json

METRICS = ("accuracy_pct", "macro_f1_pct", "cohen_kappa", "balanced_accuracy_pct")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_file(root, name):
    files = list(Path(root).rglob(name))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}, found {len(files)}.")
    return files[0]


def load_predictions(job, n_splits):
    config = Path(job["config"])
    if hashlib.sha256(config.read_bytes()).hexdigest() != job["config_sha256"]:
        raise ValueError(f"Config changed after planning: {job['id']}")
    filename = ("per_trial_results.csv" if job["model"] == "transformer" and job["protocol"] != "global"
                else "test_predictions.csv")
    path = unique_file(job["output"], filename)
    rows = []
    for raw in read_csv(path):
        rows.append({"subject": raw.get("subject", raw.get("target_subject")),
                     "trial_index": int(raw.get("trial_index", raw.get("test_trial_index"))),
                     "fold": int(raw["fold"]) if job["protocol"] == "global" else None,
                     "run": raw.get("run", raw.get("test_run", "")),
                     "y_true": int(raw["y_true"]), "y_pred": int(raw["y_pred"])})
    if not rows or any(not row["subject"] for row in rows):
        raise ValueError(f"Empty predictions/missing subjects: {path}")
    indices = [r["trial_index"] for r in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Trials tested more than once in {path}.")
    if job["protocol"] == "global":
        by_index = {r["trial_index"]: r for r in rows}
        if set(r["fold"] for r in rows) != set(range(1, n_splits + 1)):
            raise ValueError(f"Incomplete fold coverage: {path}")
        subject_folds = defaultdict(set)
        for r in rows:
            subject_folds[r["subject"]].add(r["fold"])
        if any(len(folds) != 1 for folds in subject_folds.values()):
            raise ValueError(f"A subject appears in multiple test folds: {path}")
        if job["model"] == "transformer":
            split_paths = list(path.parent.glob("fold_*/split.json"))
            splits = [json.loads(p.read_text(encoding="utf-8")) for p in split_paths]
        else:
            splits = json.loads((path.parent / "splits.json").read_text(encoding="utf-8"))
        if len(splits) != n_splits or {int(s["fold"]) for s in splits} != set(range(1, n_splits + 1)):
            raise ValueError(f"Incomplete split metadata: {path}")
        for split in splits:
            train, test = set(split["train_indices"]), set(split["test_indices"])
            if split.get("validation_indices") or train & test or train | test != set(by_index):
                raise ValueError(f"Invalid train/test/validation partition: {path}")
            if {by_index[i]["subject"] for i in train} & {by_index[i]["subject"] for i in test}:
                raise ValueError(f"Subject leakage in {path}")
            if test != {r["trial_index"] for r in rows if r["fold"] == int(split["fold"])}:
                raise ValueError(f"Prediction fold differs from split: {path}")
    else:
        by_index = {r["trial_index"]: r for r in rows}
        if any(r["run"] == "" for r in rows):
            raise ValueError(f"Run metadata missing: {path}")
        if job["model"] == "transformer":
            splits = [json.loads(p.read_text(encoding="utf-8")) for p in path.parent.glob("*/test_run_*/split.json")]
        else:
            splits = json.loads((path.parent / "splits.json").read_text(encoding="utf-8"))
        tested = []
        for split in splits:
            subject = split.get("subject", split.get("target_subject"))
            run = int(split["test_run"])
            train = set(split.get("train_indices", split.get("fine_tune_indices")))
            test = set(split["test_indices"])
            expected_test = {r["trial_index"] for r in rows if r["subject"] == subject and int(r["run"]) == run}
            expected_train = {r["trial_index"] for r in rows if r["subject"] == subject and int(r["run"]) != run}
            if split.get("validation_indices") or not train or not test or train != expected_train or test != expected_test:
                raise ValueError(f"Run-level split mismatch/leakage: {path}")
            tested.extend(test)
        if sorted(tested) != sorted(by_index):
            raise ValueError(f"Incomplete or duplicate held-out run coverage: {path}")
        if job["protocol"] == "adaptation":
            for subject in {r["subject"] for r in rows}:
                split = json.loads((path.parent / subject / "pretrain_split.json").read_text(encoding="utf-8"))
                target = {r["trial_index"] for r in rows if r["subject"] == subject}
                if split.get("validation_indices") or split.get("test_indices") or target & set(split["train_indices"]):
                    raise ValueError(f"Target entered pretraining or source holdouts exist: {path}")
    return rows


def scores(rows, labels):
    truth, predicted = [r["y_true"] for r in rows], [r["y_pred"] for r in rows]
    if not set(truth + predicted) <= set(labels):
        raise ValueError("Prediction class IDs differ between experiments.")
    result = {"accuracy_pct": 100 * accuracy_score(truth, predicted),
              "balanced_accuracy_pct": 100 * balanced_accuracy_score(truth, predicted),
              "macro_f1_pct": 100 * f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
              "cohen_kappa": cohen_kappa_score(truth, predicted, labels=labels)}
    # Kappa is undefined if both arrays contain only the same single class.
    return {key: float(value) if np.isfinite(value) else None for key, value in result.items()}


def metadata(job):
    return {"experiment_id": job["id"], "model": MODELS[job["model"]],
            "model_configuration": VARIANTS.get(job["variant"], MODELS[job["model"]]),
            "rest_length_s": job["rest_length_s"]}


def grouped_scores(job, rows, unit, labels):
    groups = defaultdict(list)
    for row in rows:
        groups[row[unit]].append(row)
    result = []
    for key, records in sorted(groups.items()):
        entry = {**metadata(job), unit: key, "n_trials": len(records), **scores(records, labels)}
        if unit == "subject" and job["protocol"] == "global":
            entry["fold"] = records[0]["fold"]
        result.append(entry)
    return result


def summary_row(job, folds):
    result = {**metadata(job), "n_folds": len(folds), "sd_ddof": 1}
    for metric in METRICS:
        values = [r[metric] for r in folds]
        if any(v is None for v in values):
            result.update({metric + "_mean": None, metric + "_sd": None})
        else:
            result.update({metric + "_mean": float(np.mean(values)), metric + "_sd": float(np.std(values, ddof=1))})
    return result


def assert_paired(left, right):
    # A trial count alone cannot verify matching cohorts, labels and test folds.
    def identity(rows):
        return {(r["subject"], r["trial_index"]): (r["fold"], r["y_true"]) for r in rows}
    if identity(left) != identity(right):
        raise ValueError("Cannot pair experiments: test subjects/trials/labels/folds differ. "
                         "Use identical retained trials and subject folds; no silent intersection is allowed.")


def paired_permutation(differences, *, seed=42, resamples=99999):
    """Two-sided paired sign-flip test; exact for <=16 pairs, otherwise Monte Carlo."""
    diff = np.round(np.asarray(differences, dtype=float), 12)
    if len(diff) < 2 or not np.isfinite(diff).all():
        raise ValueError("Need at least two finite paired differences.")
    observed = abs(float(diff.mean()))
    if np.all(diff == 0):
        return 1.0, "all_ties", 1
    if len(diff) <= 16:
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(diff))))
        values = np.abs((signs * diff).mean(axis=1))
        return float(np.mean(values >= observed - 1e-12)), "exact", len(signs)
    if resamples < 1:
        raise ValueError("permutation_resamples must be positive.")
    rng, exceed = np.random.default_rng(seed), 0
    for start in range(0, resamples, 2048):
        count = min(2048, resamples - start)
        signs = rng.choice((-1, 1), size=(count, len(diff)))
        exceed += int(np.count_nonzero(np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12))
    return (exceed + 1) / (resamples + 1), "monte_carlo", resamples


def holm_adjust(rows):
    """Holm correction within a prespecified comparison family."""
    maximum = 0.0
    for rank, row in enumerate(sorted(rows, key=lambda r: r["p_value"])):
        maximum = max(maximum, (len(rows) - rank) * row["p_value"])
        row["p_value_holm"] = min(1.0, maximum)
        row["holm_family_size"] = len(rows)
        row["reject_holm_0p05"] = row["p_value_holm"] < .05


def collect_suite(manifest, directory):
    if any(j["status"] != "completed" for j in manifest["jobs"]):
        raise ValueError("Suite is incomplete. Refusing to fabricate full-fold tables/p-values.")
    suite, jobs = manifest["suite"], manifest["jobs"]
    data = {j["id"]: load_predictions(j, int(suite["n_splits"])) for j in jobs}
    global_jobs = [j for j in jobs if j["protocol"] == "global"]
    if not global_jobs:
        raise ValueError("No global experiments to summarize.")
    labels = sorted({r["y_true"] for r in data[global_jobs[0]["id"]]})
    if len(labels) < 2:
        raise ValueError("Classification requires at least two classes.")
    class_names = None
    for job in global_jobs:
        result_dir = unique_file(job["output"], "test_predictions.csv").parent
        names = json.loads((result_dir / "summary.json").read_text(encoding="utf-8")).get("class_names")
        if names is None or len(names) != len(labels) or (class_names is not None and names != class_names):
            raise ValueError("Missing/inconsistent class-name order; cannot compare predictions.")
        class_names = names
    if labels != list(range(len(class_names))):
        raise ValueError("Expected contiguous class IDs matching class_names.")
    references = {j["rest_length_s"]: j for j in global_jobs if j["model"] == "transformer" and j["variant"] == "complete"}
    for job in global_jobs:
        assert_paired(data[references[job["rest_length_s"]]["id"]], data[job["id"]])
    folds, subjects, summaries = {}, {}, {}
    class_rows, matrix_rows, prediction_rows = [], [], []
    for job in global_jobs:
        key, records = job["id"], data[job["id"]]
        folds[key] = grouped_scores(job, records, "fold", labels)
        subjects[key] = grouped_scores(job, records, "subject", labels)
        summaries[key] = summary_row(job, folds[key])
        prediction_rows.extend({**metadata(job), **r} for r in records)
        for fold in range(1, int(suite["n_splits"]) + 1):
            test = [r for r in records if r["fold"] == fold]
            truth, predicted = [r["y_true"] for r in test], [r["y_pred"] for r in test]
            precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
            confusion = confusion_matrix(truth, predicted, labels=labels)
            for i, label in enumerate(labels):
                class_rows.append({**metadata(job), "fold": fold, "class_id": label, "class_name": class_names[label],
                                   "precision_pct": 100 * precision[i], "recall_pct": 100 * recall[i],
                                   "f1_pct": 100 * f1[i], "support": int(support[i])})
                for k, predicted_label in enumerate(labels):
                    matrix_rows.append({**metadata(job), "fold": fold, "true_class": label,
                                        "predicted_class": predicted_label, "count": int(confusion[i, k])})
    ablations = []
    for job in global_jobs:
        if job["model"] != "transformer":
            continue
        reference = references[job["rest_length_s"]]["id"]
        differences = [a["macro_f1_pct"] - b["macro_f1_pct"] for a, b in zip(folds[job["id"]], folds[reference], strict=True)]
        ablations.append({**summaries[job["id"]], "delta_macro_f1_pp": float(np.mean(differences)),
                          "delta_macro_f1_pp_sd": float(np.std(differences, ddof=1))})

    reference = references[float(suite["comparison_rest"])]["id"]
    pvalues, pair_rows = [], []
    for unit, source in (("fold", folds), ("subject", subjects)):
        family = []
        for model in ("mdm", "spdnet"):
            baseline = next(j["id"] for j in global_jobs if j["model"] == model and j["rest_length_s"] == float(suite["comparison_rest"]))
            left = {r[unit]: r for r in source[reference]}
            right = {r[unit]: r for r in source[baseline]}
            if left.keys() != right.keys():
                raise ValueError("Mismatched paired metric units.")
            for metric in METRICS[:3]:
                pairs = [(k, left[k][metric], right[k][metric]) for k in sorted(left)]
                if any(a is None or b is None for _, a, b in pairs):
                    raise ValueError(f"Undefined {metric}; cannot perform paired inference.")
                differences = [a - b for _, a, b in pairs]
                pvalue, method, permutations = paired_permutation(
                    differences, seed=int(suite["seed"]), resamples=int(suite.get("permutation_resamples", 99999)),
                )
                family.append({"reference": reference, "baseline": baseline,
                               "rest_length_s": suite["comparison_rest"], "metric": metric,
                               "unit": unit, "n_pairs": len(pairs), "mean_difference": float(np.mean(differences)),
                               "difference_unit": "kappa" if metric == "cohen_kappa" else "percentage_points",
                               "test": "paired_sign_flip_two_sided", "permutation_method": method,
                               "n_permutations": permutations, "p_value": pvalue,
                               "interpretation": "CV fold dependence caveat" if unit == "fold" else "exploratory; subjects share fitted fold models"})
                pair_rows.extend({"reference": reference, "baseline": baseline, "unit": unit, "pair_id": k,
                                  "metric": metric, "reference_score": a, "baseline_score": b, "difference": a - b}
                                 for k, a, b in pairs)
        holm_adjust(family)
        pvalues.extend(family)

    adaptation_subjects, adaptation_runs, adaptation_summary = [], [], []
    for job in jobs:
        if job["protocol"] == "global":
            continue
        records = data[job["id"]]
        ss = grouped_scores(job, records, "subject", labels)
        adaptation_subjects.extend({**r, "protocol": job["protocol"]} for r in ss)
        for subject in sorted({r["subject"] for r in records}):
            runs = grouped_scores(job, [r for r in records if r["subject"] == subject], "run", labels)
            adaptation_runs.extend({**r, "subject": subject, "protocol": job["protocol"]} for r in runs)
        row = {**metadata(job), "protocol": job["protocol"], "aggregation_unit": "subject",
               "n_subjects": len(ss), "n_trials": len(records)}
        for metric in METRICS:
            values = [r[metric] for r in ss]
            row[metric + "_mean"] = float(np.mean(values)) if all(v is not None for v in values) else None
            row[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None
        adaptation_summary.append(row)

    output = Path(directory) / "tables"
    output.mkdir(exist_ok=True)
    write_csv(output / "global_fold_results.csv", [r for records in folds.values() for r in records])
    write_csv(output / "global_subject_results.csv", [r for records in subjects.values() for r in records])
    write_csv(output / "global_predictions.csv", prediction_rows)
    write_csv(output / "main_results.csv", [summaries[j["id"]] for j in global_jobs if j["model"] != "transformer" or j["variant"] == "complete"])
    write_csv(output / "ablation_results.csv", ablations)
    write_csv(output / "pvalues.csv", pvalues)
    write_csv(output / "comparison_pairs.csv", pair_rows)
    write_csv(output / "spdtransformer_rest_1p5_subject_results.csv", subjects[references[1.5]["id"]])
    write_csv(output / "per_class_metrics.csv", class_rows)
    write_csv(output / "confusion_matrices.csv", matrix_rows)
    write_csv(output / "adaptation_subject_results.csv", adaptation_subjects)
    write_csv(output / "adaptation_run_results.csv", adaptation_runs)
    write_csv(output / "adaptation_summary.csv", adaptation_summary)
    write_json(output / "report_metadata.json", {
        "global_aggregation": "arithmetic mean and sample SD (ddof=1) of held-out subject folds",
        "adaptation_aggregation": "mean and sample SD across subjects; not subject-independent",
        "rest_length_definition": "pre-event context; NOT baseline correction or a verified resting-state task",
        "epoch_windows": [[-float(r), suite["epoch_end"]] for r in suite["rest_lengths"]],
        "class_ids": labels, "class_names": class_names,
        "pvalue_note": "Two-sided paired sign flips; minimum nonzero exact p for five nonzero pairs is 2/32=0.0625. "
                       "Holm adjusts six comparisons per unit (two baselines x three metrics). "
                       "CV training sets overlap; fold tests are not unconditional independent replications. "
                       "Subject tests are exploratory conditional comparisons, not independent model-training runs.",
        "statistical_sources": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
                                "https://www.jmlr.org/papers/v5/grandvalet04a.html"],
    })
