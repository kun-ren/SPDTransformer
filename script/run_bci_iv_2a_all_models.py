"""Run every BCI Competition IV-2a model and build unified result tables.

The campaign includes CSP+LDA, classical MDM, SPDNet, and the proposed SPD
Transformer. Each model is evaluated with both stratified sample-level folds
and stratified subject-disjoint folds.

Examples:
    python script/run_bci_iv_2a_all_models.py --device cuda:0
    python script/run_bci_iv_2a_all_models.py --dry-run
    python script/run_bci_iv_2a_all_models.py --models csp_lda,mdm
    python script/run_bci_iv_2a_all_models.py --summarize-only CAMPAIGN_DIR
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

from src.baselines.baseline_utils import expand_data_training_experiments
from src.baselines.mdm_baseline import expand_mdm_experiments
from src.training.train import expand_experiments as expand_transformer_experiments


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "bci_iv_2a_all_models"
)
EXPECTED_CLASSES = {"left_hand", "right_hand", "feet", "tongue"}
EXPECTED_FOLDS = 5
REPORT_METRICS = ("accuracy", "macro_f1", "cohen_kappa")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config_path: Path
    runner_path: Path
    runner_kind: str
    accepts_device: bool
    input_representation: str


@dataclass(frozen=True)
class RunRecord:
    spec: ModelSpec
    config: dict[str, Any]
    folds: list[dict[str, Any]]
    class_names: list[str]
    evaluation: dict[str, Any]
    source: Path


MODEL_SPECS = (
    ModelSpec(
        key="csp_lda",
        label="CSP + LDA",
        config_path=PROJECT_ROOT / "configs" / "csp_lda_bci_iv_2a.yaml",
        runner_path=PROJECT_ROOT / "src" / "baselines" / "csp_lda_baseline.py",
        runner_kind="baseline",
        accepts_device=False,
        input_representation="8-30 Hz full-trial EEG",
    ),
    ModelSpec(
        key="mdm",
        label="MDM",
        config_path=PROJECT_ROOT / "configs" / "mdm_bci_iv_2a.yaml",
        runner_path=PROJECT_ROOT / "src" / "baselines" / "mdm_baseline.py",
        runner_kind="mdm",
        accepts_device=True,
        input_representation="8-30 Hz full-trial 22x22 covariance",
    ),
    ModelSpec(
        key="spdnet",
        label="SPDNet",
        config_path=PROJECT_ROOT / "configs" / "spdnet_bci_iv_2a.yaml",
        runner_path=PROJECT_ROOT / "src" / "baselines" / "spdnet_baseline.py",
        runner_kind="baseline",
        accepts_device=True,
        input_representation="8-30 Hz full-trial 22x22 covariance",
    ),
    ModelSpec(
        key="spd_transformer",
        label="SPD Transformer",
        config_path=PROJECT_ROOT / "configs" / "train_bci_iv_2a.yaml",
        runner_path=PROJECT_ROOT / "src" / "training" / "train.py",
        runner_kind="transformer",
        accepts_device=True,
        input_representation="three bands, temporal segments, 3x7 motor regions",
    ),
)
SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_models(raw: str | None) -> list[ModelSpec]:
    if raw is None or not raw.strip():
        return list(MODEL_SPECS)
    keys = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(keys) - set(SPEC_BY_KEY))
    if unknown:
        valid = ", ".join(SPEC_BY_KEY)
        raise argparse.ArgumentTypeError(
            f"unknown model(s): {', '.join(unknown)}; valid models: {valid}"
        )
    selected = set(keys)
    return [spec for spec in MODEL_SPECS if spec.key in selected]


def count_runs(spec: ModelSpec, config: dict[str, Any]) -> int:
    if spec.runner_kind == "transformer":
        return len(expand_transformer_experiments(config))
    if spec.runner_kind == "mdm":
        return len(expand_mdm_experiments(config))
    return len(expand_data_training_experiments(config))


def make_campaign_config(
    spec: ModelSpec,
    campaign_dir: Path,
) -> dict[str, Any]:
    config = deepcopy(load_yaml(spec.config_path))
    data = config.setdefault("data", {})
    data["dataset"] = ["bnci2014_001"]
    data["subjects"] = "1-9"
    data["events"] = "left_hand,right_hand,feet,tongue"
    data["sessions"] = "all"
    data["test_size"] = [0.2]
    data["val_size"] = [0.0]
    data["allow_subject_overlap"] = [True, False]
    config.setdefault("training", {})["n_splits"] = [EXPECTED_FOLDS]
    config.setdefault("output", {})["dir"] = str(
        (campaign_dir / "raw" / spec.key).resolve()
    )
    return config


def build_campaign(
    campaign_dir: Path,
    specs: list[ModelSpec],
) -> dict[str, tuple[ModelSpec, Path]]:
    commands: dict[str, tuple[ModelSpec, Path]] = {}
    manifest: dict[str, Any] = {
        "campaign_dir": str(campaign_dir.resolve()),
        "dataset": "MOABB BNCI2014_001 / BCI Competition IV Dataset 2a",
        "subjects": list(range(1, 10)),
        "classes": sorted(EXPECTED_CLASSES),
        "evaluation": ["stratified_kfold", "stratified_group_kfold"],
        "n_splits": EXPECTED_FOLDS,
        "models": {},
    }

    for spec in specs:
        config = make_campaign_config(spec, campaign_dir)
        run_count = count_runs(spec, config)
        if run_count != 2:
            raise ValueError(
                f"{spec.key} expands to {run_count} runs; expected exactly "
                "two (sample-level and subject-disjoint)."
            )
        config_path = campaign_dir / "configs" / f"{spec.key}.yaml"
        save_yaml(config_path, config)
        commands[spec.key] = (spec, config_path)
        manifest["models"][spec.key] = {
            "label": spec.label,
            "config": str(config_path.resolve()),
            "runner": str(spec.runner_path.resolve()),
            "output_dir": config["output"]["dir"],
            "expected_run_count": run_count,
            "input_representation": spec.input_representation,
        }

    write_json(campaign_dir / "manifest.json", manifest)
    return commands


def command_for(
    spec: ModelSpec,
    config_path: Path,
    device: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(spec.runner_path),
        "--config",
        str(config_path),
    ]
    if device and spec.accepts_device:
        command.extend(["--device", device])
    return command


def run_campaign(
    commands: dict[str, tuple[ModelSpec, Path]],
    device: str | None,
) -> None:
    total = len(commands)
    for index, (spec, config_path) in enumerate(commands.values(), start=1):
        command = command_for(spec, config_path, device)
        print(f"\n[{index}/{total}] Running {spec.label}", flush=True)
        print("  " + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_transformer_records(spec: ModelSpec, root: Path) -> list[RunRecord]:
    records = []
    for summary_path in sorted(root.glob("*/run_*/five_fold_summary.json")):
        summary = read_json(summary_path)
        folds = list(summary.get("folds", []))
        class_names = list(folds[0].get("class_names", [])) if folds else []
        records.append(
            RunRecord(
                spec=spec,
                config=load_yaml(summary_path.with_name("config.yaml")),
                folds=folds,
                class_names=class_names,
                evaluation=dict(summary.get("evaluation", {})),
                source=summary_path,
            )
        )
    return records


def load_baseline_records(spec: ModelSpec, root: Path) -> list[RunRecord]:
    records = []
    for summary_path in sorted(root.glob("*/run_*/summary.json")):
        summary = read_json(summary_path)
        config = summary.get("config")
        if not isinstance(config, dict):
            config = read_json(summary_path.with_name("config.json"))
        records.append(
            RunRecord(
                spec=spec,
                config=config,
                folds=list(summary.get("folds", [])),
                class_names=list(summary.get("class_names", [])),
                evaluation=dict(summary.get("evaluation", {})),
                source=summary_path,
            )
        )
    return records


def selected_specs_from_manifest(campaign_dir: Path) -> list[ModelSpec]:
    manifest = read_json(campaign_dir / "manifest.json")
    models = manifest.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Campaign manifest does not contain any models")
    unknown = sorted(set(models) - set(SPEC_BY_KEY))
    if unknown:
        raise ValueError(f"Campaign manifest contains unknown models: {unknown}")
    return [spec for spec in MODEL_SPECS if spec.key in models]


def load_campaign_records(
    campaign_dir: Path,
    specs: list[ModelSpec],
) -> list[RunRecord]:
    records = []
    for spec in specs:
        root = campaign_dir / "raw" / spec.key
        model_records = (
            load_transformer_records(spec, root)
            if spec.runner_kind == "transformer"
            else load_baseline_records(spec, root)
        )
        if len(model_records) != 2:
            raise ValueError(
                f"Expected two completed {spec.label} runs under {root}, "
                f"found {len(model_records)}."
            )
        records.extend(model_records)
    return records


def allow_subject_overlap(record: RunRecord) -> bool:
    data_cfg = record.config.get("data", {})
    return bool(data_cfg.get("allow_subject_overlap", True))


def strategy_name(record: RunRecord) -> str:
    return "sample_level" if allow_subject_overlap(record) else "subject_disjoint"


def fold_metric(fold: dict[str, Any], metric: str) -> float:
    transformer_key = f"test_{metric}"
    if transformer_key in fold:
        return float(fold[transformer_key])
    if metric in fold:
        return float(fold[metric])
    raise KeyError(f"Fold does not contain metric {metric!r}: {fold.keys()}")


def validate_record(record: RunRecord) -> None:
    if len(record.folds) != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected {EXPECTED_FOLDS} folds in {record.source}, "
            f"found {len(record.folds)}."
        )
    if set(record.class_names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected classes {sorted(EXPECTED_CLASSES)} in {record.source}, "
            f"found {record.class_names}."
        )
    for fold in record.folds:
        for metric in REPORT_METRICS:
            fold_metric(fold, metric)


def aggregate_record(record: RunRecord) -> dict[str, Any]:
    validate_record(record)
    row: dict[str, Any] = {
        "model_key": record.spec.key,
        "model": record.spec.label,
        "split_strategy": strategy_name(record),
        "allow_subject_overlap": allow_subject_overlap(record),
        "input_representation": record.spec.input_representation,
        "n_folds": len(record.folds),
        "n_classes": len(record.class_names),
        "class_names": ",".join(record.class_names),
        "source": str(record.source.resolve()),
    }
    for metric in REPORT_METRICS:
        values = [fold_metric(fold, metric) for fold in record.folds]
        row[f"{metric}_mean"] = statistics.fmean(values)
        row[f"{metric}_std"] = statistics.pstdev(values)
        row[f"{metric}_min"] = min(values)
        row[f"{metric}_max"] = max(values)
    return row


def fold_rows(record: RunRecord) -> list[dict[str, Any]]:
    validate_record(record)
    rows = []
    for fold_index, fold in enumerate(record.folds, start=1):
        rows.append(
            {
                "model_key": record.spec.key,
                "model": record.spec.label,
                "split_strategy": strategy_name(record),
                "fold": int(fold.get("fold", fold_index)),
                "n_train": int(fold["n_train"]),
                "n_test": int(fold["n_test"]),
                "accuracy": fold_metric(fold, "accuracy"),
                "macro_f1": fold_metric(fold, "macro_f1"),
                "cohen_kappa": fold_metric(fold, "cohen_kappa"),
                "source": str(record.source.resolve()),
            }
        )
    return rows


def summarize_campaign(campaign_dir: Path) -> tuple[Path, Path, Path]:
    specs = selected_specs_from_manifest(campaign_dir)
    records = load_campaign_records(campaign_dir, specs)
    model_order = {spec.key: index for index, spec in enumerate(MODEL_SPECS)}
    records.sort(
        key=lambda record: (
            model_order[record.spec.key],
            0 if allow_subject_overlap(record) else 1,
        )
    )

    aggregate_rows = [aggregate_record(record) for record in records]
    all_fold_rows = [row for record in records for row in fold_rows(record)]
    aggregate_path = campaign_dir / "all_models_results.csv"
    fold_path = campaign_dir / "all_models_fold_results.csv"
    json_path = campaign_dir / "all_models_results.json"
    write_csv(aggregate_path, aggregate_rows)
    write_csv(fold_path, all_fold_rows)
    write_json(
        json_path,
        {
            "campaign_dir": str(campaign_dir.resolve()),
            "dataset": "bnci2014_001",
            "classes": sorted(EXPECTED_CLASSES),
            "fold_standard_deviation": "population (ddof=0)",
            "results": aggregate_rows,
        },
    )
    return aggregate_path, fold_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CSP+LDA, MDM, SPDNet, and SPD Transformer on BCI "
            "Competition IV Dataset 2a and export unified result tables."
        )
    )
    parser.add_argument("--device", help="training device, e.g. cuda or cuda:0")
    parser.add_argument(
        "--models",
        help="comma-separated subset: csp_lda,mdm,spdnet,spd_transformer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"campaign parent directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write campaign configs and commands without training",
    )
    parser.add_argument(
        "--summarize-only",
        type=Path,
        metavar="CAMPAIGN_DIR",
        help="rebuild result tables from an existing completed campaign",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarize_only is not None:
        campaign_dir = args.summarize_only.expanduser().resolve()
        outputs = summarize_campaign(campaign_dir)
        for path in outputs:
            print(f"Wrote {path}")
        return 0

    specs = parse_models(args.models)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = args.output_dir.expanduser().resolve() / timestamp
    campaign_dir.mkdir(parents=True, exist_ok=False)
    commands = build_campaign(campaign_dir, specs)

    print(f"BCI IV-2a all-model campaign: {campaign_dir}")
    for spec, config_path in commands.values():
        print("  " + subprocess.list2cmdline(command_for(spec, config_path, args.device)))

    if args.dry_run:
        print("Dry run complete; no training was started.")
        return 0

    run_campaign(commands, args.device)
    outputs = summarize_campaign(campaign_dir)
    for path in outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
