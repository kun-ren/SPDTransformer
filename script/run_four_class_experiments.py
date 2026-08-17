"""Run the thesis experiment tables as four-class PhysioNet experiments.

The four classes are left hand, right hand, both hands, and both feet. This
script derives configs from the current two-class configs, runs the complete
SPD Transformer ablation matrix plus CSP+LDA and SPDNet, and writes table-ready
CSV files from the five fold-level results.

Examples:
    python script/run_four_class_experiments.py --device cuda:0
    python script/run_four_class_experiments.py --dry-run
    python script/run_four_class_experiments.py \
        --summarize-only experiments/results/four_class/20260817_120000
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

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.config_grid import expand_data_grid, expand_grid


TRANSFORMER_CONFIG = PROJECT_ROOT / "configs" / "train_grid.yaml"
CSP_CONFIG = PROJECT_ROOT / "configs" / "csp_lda_physionet.yaml"
SPDNET_CONFIG = PROJECT_ROOT / "configs" / "spdnet_physionet.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "four_class"

TASK_TYPES_GRID = [["unilateral_fist", "both"]]
EXPECTED_CLASSES = {"left_hand", "right_hand", "hands", "feet"}


@dataclass(frozen=True)
class RunRecord:
    config: dict[str, Any]
    folds: list[dict[str, Any]]
    class_names: list[str]
    source: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def make_four_class_config(base: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = deepcopy(base)
    config.setdefault("data", {})["task_types"] = deepcopy(TASK_TYPES_GRID)
    config.setdefault("output", {})["dir"] = str(output_dir.resolve())
    return config


def transformer_config(
    base: dict[str, Any],
    output_dir: Path,
    *,
    resting_lengths: list[float],
    subject_independent: list[bool],
    metric: str = "learnable-metric",
    classifier_type: str = "mdm",
    single_band: bool = False,
) -> dict[str, Any]:
    config = make_four_class_config(base, output_dir)
    data = config["data"]
    model = config["model"]

    data["epoch_slice"] = [[-float(length), 4.0] for length in resting_lengths]
    data["allow_subject_overlap"] = [not value for value in subject_independent]
    if single_band:
        data["filter_bank"] = [[[8, 30]]]

    model["metric"] = [metric]
    model["classifier_type"] = [classifier_type]
    return config


def count_transformer_runs(config: dict[str, Any]) -> int:
    return (
        len(expand_data_grid(config.get("data", {})))
        * len(expand_grid(config.get("model", {})))
        * len(expand_grid(config.get("training", {})))
    )


def count_baseline_runs(config: dict[str, Any]) -> int:
    return len(expand_data_grid(config.get("data", {}))) * len(
        expand_grid(config.get("training", {}))
    )


def build_configs(campaign_dir: Path) -> dict[str, tuple[Path, list[str]]]:
    """Create all configs and return config paths with their runner commands."""
    transformer_base = load_yaml(TRANSFORMER_CONFIG)
    csp_base = load_yaml(CSP_CONFIG)
    spdnet_base = load_yaml(SPDNET_CONFIG)
    config_dir = campaign_dir / "configs"
    raw_dir = campaign_dir / "raw"

    configs: dict[str, tuple[dict[str, Any], Path, list[str], int, bool]] = {}

    transformer_specs = {
        "spd_transformer_complete": transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_complete",
            resting_lengths=[0.0, 1.0, 2.0],
            subject_independent=[False, True],
        ),
        "spd_transformer_fixed_log_euclidean": transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_fixed_log_euclidean",
            resting_lengths=[1.0],
            subject_independent=[False],
            metric="log-euclidean",
        ),
        "spd_transformer_linear_head": transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_linear_head",
            resting_lengths=[1.0],
            subject_independent=[False, True],
            classifier_type="pooling",
        ),
        "spd_transformer_single_band": transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_single_band",
            resting_lengths=[0.0, 1.0, 2.0],
            subject_independent=[False, True],
            single_band=True,
        ),
    }
    expected_transformer_runs = {
        "spd_transformer_complete": 6,
        "spd_transformer_fixed_log_euclidean": 1,
        "spd_transformer_linear_head": 2,
        "spd_transformer_single_band": 6,
    }
    for name, config in transformer_specs.items():
        configs[name] = (
            config,
            config_dir / f"{name}.yaml",
            [str(PROJECT_ROOT / "src" / "training" / "train.py")],
            expected_transformer_runs[name],
            True,
        )

    csp = make_four_class_config(csp_base, raw_dir / "csp_lda")
    csp["data"]["allow_subject_overlap"] = [True, False]
    configs["csp_lda"] = (
        csp,
        config_dir / "csp_lda.yaml",
        [str(PROJECT_ROOT / "src" / "baselines" / "csp_lda_baseline.py")],
        2,
        False,
    )

    spdnet = make_four_class_config(spdnet_base, raw_dir / "spdnet")
    spdnet["data"]["allow_subject_overlap"] = [True, False]
    configs["spdnet"] = (
        spdnet,
        config_dir / "spdnet.yaml",
        [str(PROJECT_ROOT / "src" / "baselines" / "spdnet_baseline.py")],
        2,
        False,
    )

    commands: dict[str, tuple[Path, list[str]]] = {}
    manifest: dict[str, Any] = {
        "campaign_dir": str(campaign_dir.resolve()),
        "task_types": TASK_TYPES_GRID[0],
        "expected_classes": sorted(EXPECTED_CLASSES),
        "fold_standard_deviation": "population (ddof=0)",
        "runs": {},
    }
    for name, (config, path, runner, expected_count, is_transformer) in configs.items():
        actual_count = (
            count_transformer_runs(config)
            if is_transformer
            else count_baseline_runs(config)
        )
        if actual_count != expected_count:
            raise ValueError(
                f"{name} expands to {actual_count} runs, expected {expected_count}. "
                "Check the base config for additional grid-valued settings."
            )
        save_yaml(path, config)
        commands[name] = (path, runner)
        manifest["runs"][name] = {
            "config": str(path.resolve()),
            "expected_run_count": expected_count,
            "output_dir": config["output"]["dir"],
        }

    with (campaign_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return commands


def run_configs(
    commands: dict[str, tuple[Path, list[str]]],
    device: str | None,
) -> None:
    for index, (name, (config_path, runner)) in enumerate(commands.items(), start=1):
        command = [sys.executable, *runner, "--config", str(config_path)]
        if device and name != "csp_lda":
            command.extend(["--device", device])
        print(f"\n[{index}/{len(commands)}] Running {name}", flush=True)
        print("  " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_transformer_records(root: Path) -> list[RunRecord]:
    records = []
    for summary_path in sorted(root.glob("*/run_*/five_fold_summary.json")):
        summary = _read_json(summary_path)
        config = load_yaml(summary_path.with_name("config.yaml"))
        folds = summary.get("folds", [])
        class_names = list(folds[0].get("class_names", [])) if folds else []
        records.append(RunRecord(config, folds, class_names, summary_path))
    return records


def load_baseline_records(root: Path, *, config_name: str) -> list[RunRecord]:
    records = []
    for summary_path in sorted(root.glob("*/run_*/summary.json")):
        summary = _read_json(summary_path)
        if "config" in summary:
            config = summary["config"]
        else:
            config = _read_json(summary_path.with_name(config_name))
        records.append(
            RunRecord(
                config=config,
                folds=list(summary.get("folds", [])),
                class_names=list(summary.get("class_names", [])),
                source=summary_path,
            )
        )
    return records


def require_records(name: str, records: list[RunRecord], expected: int) -> None:
    if len(records) != expected:
        raise ValueError(
            f"Expected {expected} completed {name} runs, found {len(records)}."
        )
    for record in records:
        if len(record.folds) != 5:
            raise ValueError(
                f"Expected five folds in {record.source}, found {len(record.folds)}."
            )
        if set(record.class_names) != EXPECTED_CLASSES:
            raise ValueError(
                f"Expected four classes {sorted(EXPECTED_CLASSES)} in {record.source}, "
                f"found {record.class_names}."
            )


def subject_independent(record: RunRecord) -> bool:
    return not bool(record.config["data"]["allow_subject_overlap"])


def resting_length(record: RunRecord) -> float:
    epoch_start = float(record.config["data"]["epoch_slice"][0])
    return max(0.0, -epoch_start)


def metric_values(record: RunRecord, metric: str) -> list[float]:
    baseline_key = metric
    transformer_key = f"test_{metric}"
    values = []
    for fold in record.folds:
        if transformer_key in fold:
            values.append(float(fold[transformer_key]))
        elif baseline_key in fold:
            values.append(float(fold[baseline_key]))
        else:
            raise KeyError(f"Metric {metric!r} is missing from {record.source}")
    return values


def metric_stats(
    record: RunRecord,
    metric: str,
    scale: float = 1.0,
) -> tuple[float, float]:
    values = metric_values(record, metric)
    return statistics.fmean(values) * scale, statistics.pstdev(values) * scale


def formatted(mean: float, std: float, decimals: int = 2) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def ablation_row(label: str, record: RunRecord) -> dict[str, Any]:
    accuracy_mean, accuracy_std = metric_stats(record, "accuracy", scale=100.0)
    macro_f1_mean, macro_f1_std = metric_stats(record, "macro_f1", scale=100.0)
    return {
        "model_configuration": label,
        "resting_state_length_s": resting_length(record),
        "subject_independent": subject_independent(record),
        "accuracy_mean_pct": round(accuracy_mean, 6),
        "accuracy_std_pct": round(accuracy_std, 6),
        "accuracy_mean_std": formatted(accuracy_mean, accuracy_std),
        "macro_f1_mean_pct": round(macro_f1_mean, 6),
        "macro_f1_std_pct": round(macro_f1_std, 6),
        "macro_f1_mean_std": formatted(macro_f1_mean, macro_f1_std),
        "delta_macro_f1_pp": 0.0,
    }


def main_result_row(label: str, record: RunRecord) -> dict[str, Any]:
    accuracy_mean, accuracy_std = metric_stats(record, "accuracy", scale=100.0)
    macro_f1_mean, macro_f1_std = metric_stats(record, "macro_f1", scale=100.0)
    kappa_mean, kappa_std = metric_stats(record, "cohen_kappa")
    return {
        "model": label,
        "subject_independent": subject_independent(record),
        "accuracy_mean_pct": round(accuracy_mean, 6),
        "accuracy_std_pct": round(accuracy_std, 6),
        "accuracy_mean_std": formatted(accuracy_mean, accuracy_std),
        "macro_f1_mean_pct": round(macro_f1_mean, 6),
        "macro_f1_std_pct": round(macro_f1_std, 6),
        "macro_f1_mean_std": formatted(macro_f1_mean, macro_f1_std),
        "cohen_kappa_mean": round(kappa_mean, 6),
        "cohen_kappa_std": round(kappa_std, 6),
        "cohen_kappa_mean_std": formatted(kappa_mean, kappa_std, decimals=3),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty result table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_campaign(campaign_dir: Path) -> tuple[Path, Path]:
    raw_dir = campaign_dir / "raw"
    complete = load_transformer_records(raw_dir / "spd_transformer_complete")
    fixed = load_transformer_records(raw_dir / "spd_transformer_fixed_log_euclidean")
    linear = load_transformer_records(raw_dir / "spd_transformer_linear_head")
    single_band = load_transformer_records(raw_dir / "spd_transformer_single_band")
    csp = load_baseline_records(raw_dir / "csp_lda", config_name="config.json")
    spdnet = load_baseline_records(raw_dir / "spdnet", config_name="config.json")

    require_records("complete-model", complete, 6)
    require_records("fixed Log-Euclidean", fixed, 1)
    require_records("linear-head", linear, 2)
    require_records("single-band", single_band, 6)
    require_records("CSP+LDA", csp, 2)
    require_records("SPDNet", spdnet, 2)

    ordered_groups = [
        ("Complete model", complete),
        ("Fixed Log-Euclidean metric", fixed),
        ("Linear classification head", linear),
        ("Single 8--30 Hz band", single_band),
    ]
    ablation_rows = [
        ablation_row(label, record)
        for label, records in ordered_groups
        for record in sorted(
            records,
            key=lambda item: (subject_independent(item), resting_length(item)),
        )
    ]
    complete_mf1 = {
        (row["resting_state_length_s"], row["subject_independent"]): row[
            "macro_f1_mean_pct"
        ]
        for row in ablation_rows
        if row["model_configuration"] == "Complete model"
    }
    for row in ablation_rows:
        key = (row["resting_state_length_s"], row["subject_independent"])
        if key not in complete_mf1:
            raise ValueError(f"No matching complete-model result for {key}")
        row["delta_macro_f1_pp"] = round(
            row["macro_f1_mean_pct"] - complete_mf1[key], 6
        )

    complete_main = [record for record in complete if resting_length(record) == 1.0]
    if len(complete_main) != 2:
        raise ValueError("Expected two complete-model runs with 1 s resting state")
    main_groups = [
        ("CSP+LDA 4-s Epoch", csp),
        ("SPDNet", spdnet),
        ("Proposed SPD Transformer", complete_main),
    ]
    main_rows = [
        main_result_row(label, record)
        for label, records in main_groups
        for record in sorted(records, key=subject_independent)
    ]

    ablation_path = campaign_dir / "ablation_results.csv"
    main_path = campaign_dir / "main_results.csv"
    write_csv(ablation_path, ablation_rows)
    write_csv(main_path, main_rows)
    return ablation_path, main_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two thesis result tables with four PhysioNet motor-imagery "
            "classes and export fold aggregates as CSV."
        )
    )
    parser.add_argument("--device", help="training device, e.g. cuda or cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"parent directory for new campaigns (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write configs and show commands without starting training",
    )
    parser.add_argument(
        "--summarize-only",
        type=Path,
        metavar="CAMPAIGN_DIR",
        help="skip training and rebuild CSVs from a completed campaign",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize_only is not None:
        campaign_dir = args.summarize_only.expanduser().resolve()
        ablation_path, main_path = summarize_campaign(campaign_dir)
        print(f"Wrote {ablation_path}")
        print(f"Wrote {main_path}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = args.output_dir.expanduser().resolve() / timestamp
    campaign_dir.mkdir(parents=True, exist_ok=False)
    commands = build_configs(campaign_dir)

    print(f"Four-class campaign: {campaign_dir}")
    for name, (config_path, runner) in commands.items():
        command = [sys.executable, *runner, "--config", str(config_path)]
        if args.device and name != "csp_lda":
            command.extend(["--device", args.device])
        print("  " + subprocess.list2cmdline(command))

    if args.dry_run:
        print("Dry run complete; no training was started.")
        return 0

    run_configs(commands, args.device)
    ablation_path, main_path = summarize_campaign(campaign_dir)
    print(f"\nWrote {ablation_path}")
    print(f"Wrote {main_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
