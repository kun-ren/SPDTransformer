"""Run subject-specific four-class PhysioNet baselines and ablations.

Every model is evaluated on the same target subjects. A target run is held
out once, split approximately 50/50 into validation and test, and its test
predictions are pooled per subject before table statistics are calculated.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_CONFIG = (
    PROJECT_ROOT / "configs" / "train_physionet_pretrain_finetune_loro.yaml"
)
CSP_CONFIG = PROJECT_ROOT / "configs" / "csp_lda_physionet.yaml"
MDM_CONFIG = PROJECT_ROOT / "configs" / "mdm_physionet.yaml"
SPDNET_CONFIG = PROJECT_ROOT / "configs" / "spdnet_physionet.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "four_class"

TASK_TYPES_GRID = [["unilateral_fist", "both"]]
EXPECTED_CLASSES = {"left_hand", "right_hand", "hands", "feet"}
SUBJECT_COLUMNS = (
    "Subject",
    "Trials",
    "Accuracy (%)",
    "Balanced Accuracy (%)",
    "Macro-F1",
    "Cohen’s κ",
)


@dataclass(frozen=True)
class RunRecord:
    config: dict[str, Any]
    subjects: list[dict[str, Any]]
    class_names: list[str]
    source: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return value


def save_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def read_subject_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    missing = set(SUBJECT_COLUMNS) - set(raw_rows[0] if raw_rows else {})
    if missing:
        raise ValueError(
            f"{path} lacks publication columns {sorted(missing)}. Rerun its "
            "training function with the updated subject-level metric export."
        )
    rows = []
    for row in raw_rows:
        rows.append(
            {
                "Subject": str(row["Subject"]),
                "Trials": int(float(row["Trials"])),
                "Accuracy (%)": float(row["Accuracy (%)"]),
                "Balanced Accuracy (%)": float(row["Balanced Accuracy (%)"]),
                "Macro-F1": float(row["Macro-F1"]),
                "Cohen’s κ": float(row["Cohen’s κ"]),
            }
        )
    return rows


def make_four_class_config(
    base: dict[str, Any], output_dir: Path, subjects: str | None
) -> dict[str, Any]:
    config = deepcopy(base)
    config.setdefault("data", {})["task_types"] = deepcopy(TASK_TYPES_GRID)
    if subjects is not None:
        config["data"]["subjects"] = subjects
    config.setdefault("output", {})["dir"] = str(output_dir.resolve())
    return config


def make_transformer_config(
    base: dict[str, Any],
    output_dir: Path,
    subjects: str | None,
    *,
    resting_length: float,
    metric: str = "learnable-metric",
    classifier_type: str = "mdm",
    single_band: bool = False,
) -> dict[str, Any]:
    config = make_four_class_config(base, output_dir, subjects)
    config["data"]["epoch_slice"] = [[-float(resting_length), 4.0]]
    if single_band:
        config["data"]["filter_bank"] = [[[8, 30]]]
    config["model"]["metric"] = [metric]
    config["model"]["classifier_type"] = [classifier_type]
    return config


def build_configs(
    campaign_dir: Path, subjects: str | None
) -> dict[str, tuple[Path, Path, bool]]:
    transformer_base = load_yaml(TRANSFORMER_CONFIG)
    if subjects is None and transformer_base.get("data", {}).get("subjects") in {
        None,
        "",
    }:
        raise ValueError(
            "No target subjects configured. Use --subjects 1 or --subjects 1-10, "
            "or set data.subjects in train_physionet_pretrain_finetune_loro.yaml."
        )
    config_dir = campaign_dir / "configs"
    raw_dir = campaign_dir / "raw"
    transformer_runner = (
        PROJECT_ROOT / "src" / "training" / "train_pretrain_finetune_loro.py"
    )
    specs: dict[str, tuple[dict[str, Any], Path, bool]] = {}

    for rest in (0.0, 1.0, 2.0):
        suffix = str(rest).replace(".", "p")
        name = f"spd_transformer_complete_rest_{suffix}"
        specs[name] = (
            make_transformer_config(
                transformer_base,
                raw_dir / name,
                subjects,
                resting_length=rest,
            ),
            transformer_runner,
            True,
        )
        single_name = f"spd_transformer_single_band_rest_{suffix}"
        specs[single_name] = (
            make_transformer_config(
                transformer_base,
                raw_dir / single_name,
                subjects,
                resting_length=rest,
                single_band=True,
            ),
            transformer_runner,
            True,
        )

    specs["spd_transformer_fixed_log_euclidean"] = (
        make_transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_fixed_log_euclidean",
            subjects,
            resting_length=1.0,
            metric="log-euclidean",
        ),
        transformer_runner,
        True,
    )
    specs["spd_transformer_linear_head"] = (
        make_transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_linear_head",
            subjects,
            resting_length=1.0,
            classifier_type="pooling",
        ),
        transformer_runner,
        True,
    )

    baseline_specs = {
        "csp_lda": (CSP_CONFIG, "csp_lda_baseline.py", False),
        "mdm": (MDM_CONFIG, "mdm_baseline.py", True),
        "spdnet": (SPDNET_CONFIG, "spdnet_baseline.py", True),
    }
    for name, (base_path, filename, uses_device) in baseline_specs.items():
        config = make_four_class_config(load_yaml(base_path), raw_dir / name, subjects)
        config.setdefault("training", {})["subject_specific"] = [True]
        config["training"]["held_out_run_validation_size"] = [0.5]
        specs[name] = (
            config,
            PROJECT_ROOT / "src" / "baselines" / filename,
            uses_device,
        )

    commands: dict[str, tuple[Path, Path, bool]] = {}
    manifest: dict[str, Any] = {
        "protocol": "subject-specific leave-one-run-out; held-out run split 50/50 validation/test",
        "task_types": TASK_TYPES_GRID[0],
        "expected_classes": sorted(EXPECTED_CLASSES),
        "subject_aggregation": "pooled held-out test-trial predictions",
        "p_value": "two-sided paired Wilcoxon signed-rank test on subject accuracies",
        "runs": {},
    }
    for name, (config, executable, uses_device) in specs.items():
        path = config_dir / f"{name}.yaml"
        save_yaml(path, config)
        commands[name] = (path, executable, uses_device)
        manifest["runs"][name] = {
            "config": str(path.resolve()),
            "output_dir": config["output"]["dir"],
        }
    with (campaign_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return commands


def run_configs(
    commands: dict[str, tuple[Path, Path, bool]], device: str | None
) -> None:
    for index, (name, (config_path, runner, uses_device)) in enumerate(
        commands.items(), start=1
    ):
        command = [sys.executable, str(runner), "--config", str(config_path)]
        if device and uses_device:
            command.extend(["--device", device])
        print(f"\n[{index}/{len(commands)}] Running {name}", flush=True)
        print("  " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def only_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one completed {description}, found {len(paths)}.")
    return paths[0]


def load_transformer_record(root: Path) -> RunRecord:
    table = only_path(list(root.glob("*/per_subject_summary.csv")), str(root))
    run_dir = table.parent
    overall = read_json(run_dir / "overall_summary.json")
    return RunRecord(
        config=load_yaml(run_dir / "config.yaml"),
        subjects=read_subject_table(table),
        class_names=list(overall.get("class_names", [])),
        source=table,
    )


def load_baseline_record(root: Path) -> RunRecord:
    table = only_path(list(root.glob("run_*/per_subject_summary.csv")), str(root))
    summary = read_json(table.parent / "summary.json")
    return RunRecord(
        config=dict(summary.get("config", {})),
        subjects=read_subject_table(table),
        class_names=list(summary.get("class_names", [])),
        source=table,
    )


def validate_record(record: RunRecord) -> None:
    if set(record.class_names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {sorted(EXPECTED_CLASSES)} in {record.source}, got "
            f"{record.class_names}."
        )
    identifiers = [row["Subject"] for row in record.subjects]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Invalid or duplicate subject rows in {record.source}.")


def subject_map(record: RunRecord) -> dict[str, dict[str, Any]]:
    return {str(row["Subject"]): row for row in record.subjects}


def paired_p_value(record: RunRecord, reference: RunRecord) -> float:
    from scipy.stats import wilcoxon

    left = subject_map(record)
    right = subject_map(reference)
    if set(left) != set(right):
        raise ValueError(
            f"Paired comparison needs identical subjects: {record.source} versus "
            f"{reference.source}."
        )
    subjects = sorted(left)
    x = np.asarray([left[s]["Accuracy (%)"] for s in subjects], dtype=float)
    y = np.asarray([right[s]["Accuracy (%)"] for s in subjects], dtype=float)
    if len(subjects) < 2:
        return float("nan")
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y, alternative="two-sided", zero_method="zsplit").pvalue)


def mean_sd(values: list[float], decimals: int) -> str:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def median_iqr(values: list[float]) -> str:
    q1, median, q3 = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return f"{median:.2f} [{q1:.2f}–{q3:.2f}]"


def format_p_value(value: float | None) -> str:
    if value is None:
        return "—"
    if np.isnan(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def result_row(
    label_key: str,
    label: str,
    record: RunRecord,
    reference: RunRecord | None,
    **extra: Any,
) -> dict[str, Any]:
    accuracies = [float(row["Accuracy (%)"]) for row in record.subjects]
    macro_f1 = [float(row["Macro-F1"]) for row in record.subjects]
    kappas = [float(row["Cohen’s κ"]) for row in record.subjects]
    return {
        label_key: label,
        **extra,
        "Accuracy (%) ↑": mean_sd(accuracies, 2),
        "Median [IQR] (%)": median_iqr(accuracies),
        "Macro-F1 ↑": mean_sd(macro_f1, 3),
        "Cohen’s κ ↑": mean_sd(kappas, 3),
        "p-value": format_p_value(
            None if reference is None else paired_p_value(record, reference)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_campaign(campaign_dir: Path) -> tuple[Path, Path, Path]:
    raw = campaign_dir / "raw"
    complete = {
        rest: load_transformer_record(
            raw / f"spd_transformer_complete_rest_{str(rest).replace('.', 'p')}"
        )
        for rest in (0.0, 1.0, 2.0)
    }
    single = {
        rest: load_transformer_record(
            raw / f"spd_transformer_single_band_rest_{str(rest).replace('.', 'p')}"
        )
        for rest in (0.0, 1.0, 2.0)
    }
    fixed = load_transformer_record(raw / "spd_transformer_fixed_log_euclidean")
    linear = load_transformer_record(raw / "spd_transformer_linear_head")
    csp = load_baseline_record(raw / "csp_lda")
    mdm = load_baseline_record(raw / "mdm")
    spdnet = load_baseline_record(raw / "spdnet")
    records = [*complete.values(), *single.values(), fixed, linear, csp, mdm, spdnet]
    for record in records:
        validate_record(record)

    proposed = complete[1.0]
    baseline_rows = [
        result_row("Model", "CSP+LDA", csp, proposed),
        result_row("Model", "MDM", mdm, proposed),
        result_row("Model", "SPDNet", spdnet, proposed),
        result_row("Model", "Proposed SPD Transformer", proposed, None),
    ]
    ablation_specs = [
        ("Complete model", rest, complete[rest], complete[rest])
        for rest in (0.0, 1.0, 2.0)
    ] + [
        ("Fixed Log-Euclidean metric", 1.0, fixed, complete[1.0]),
        ("Linear classification head", 1.0, linear, complete[1.0]),
    ] + [
        ("Single 8–30 Hz band", rest, single[rest], complete[rest])
        for rest in (0.0, 1.0, 2.0)
    ]
    ablation_rows = [
        result_row(
            "Model configuration",
            label,
            record,
            None if record is reference else reference,
            **{"Resting-state length (s)": rest},
        )
        for label, rest, record, reference in ablation_specs
    ]
    per_subject_rows = []
    for label, record in (
        ("CSP+LDA", csp),
        ("MDM", mdm),
        ("SPDNet", spdnet),
        ("Proposed SPD Transformer", proposed),
    ):
        per_subject_rows.extend(
            {"Model": label, **{key: row[key] for key in SUBJECT_COLUMNS}}
            for row in record.subjects
        )

    baseline_path = campaign_dir / "baseline_results.csv"
    main_path = campaign_dir / "main_results.csv"
    ablation_path = campaign_dir / "ablation_results.csv"
    subject_path = campaign_dir / "per_subject_results.csv"
    write_csv(baseline_path, baseline_rows)
    write_csv(main_path, baseline_rows)
    write_csv(ablation_path, ablation_rows)
    write_csv(subject_path, per_subject_rows)
    return baseline_path, ablation_path, subject_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Training device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--subjects",
        help="Target subjects, e.g. 1 or 1-10; overrides each generated config.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", type=Path, metavar="CAMPAIGN_DIR")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize_only is not None:
        paths = summarize_campaign(args.summarize_only.expanduser().resolve())
        for path in paths:
            print(f"Wrote {path}")
        return 0

    campaign_dir = (
        args.output_dir.expanduser().resolve()
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    campaign_dir.mkdir(parents=True, exist_ok=False)
    commands = build_configs(campaign_dir, args.subjects)
    print(f"Four-class subject-specific campaign: {campaign_dir}")
    for config_path, runner, uses_device in commands.values():
        command = [sys.executable, str(runner), "--config", str(config_path)]
        if args.device and uses_device:
            command.extend(["--device", args.device])
        print("  " + subprocess.list2cmdline(command))
    if args.dry_run:
        print("Dry run complete; no training was started.")
        return 0

    run_configs(commands, args.device)
    paths = summarize_campaign(campaign_dir)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
