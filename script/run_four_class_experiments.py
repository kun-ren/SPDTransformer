"""Run subject-specific four-class BCI baselines and ablations.

Every model is evaluated on the same target subjects. Runs are pooled inside
each target subject and trials are shuffled into 70% train / 30% test with no
validation dataset. EEGNet additionally reports other-subject pretraining plus
target-subject fine-tuning. Use ``--dataset physionet_mi`` or
``--dataset bci_iv_2a``.
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
EEGNET_CONFIG = PROJECT_ROOT / "configs" / "eegnet_physionet.yaml"
BCI_TRANSFORMER_CONFIG = (
    PROJECT_ROOT / "configs" / "train_bci_iv_2a_pretrain_finetune_loro.yaml"
)
BCI_CSP_CONFIG = PROJECT_ROOT / "configs" / "csp_lda_bci_iv_2a.yaml"
BCI_MDM_CONFIG = PROJECT_ROOT / "configs" / "mdm_bci_iv_2a.yaml"
BCI_SPDNET_CONFIG = PROJECT_ROOT / "configs" / "spdnet_bci_iv_2a.yaml"
BCI_EEGNET_CONFIG = PROJECT_ROOT / "configs" / "eegnet_bci_iv_2a.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "four_class"

DEFAULT_TASK_TYPES = ("unilateral_fist", "both")
TASK_CLASS_NAMES = {
    ("unilateral_fist", "both"): {
        "left_hand",
        "right_hand",
        "hands",
        "feet",
    },
    ("unilateral_fist",): {"left_hand", "right_hand"},
    ("both",): {"hands", "feet"},
}
BCI_IV_2A_CLASS_NAMES = {"left_hand", "right_hand", "feet", "tongue"}
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


def normalize_campaign_dataset(value: Any) -> str:
    normalized = str(value or "physionet_mi").strip().lower().replace("-", "_")
    aliases = {
        "physionet": "physionet_mi",
        "physionet_mi": "physionet_mi",
        "eegbci": "physionet_mi",
        "bci_iv_2a": "bnci2014_001",
        "bci_competition_iv_2a": "bnci2014_001",
        "bnci2014_001": "bnci2014_001",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            "dataset must be physionet_mi or bci_iv_2a"
        )
    return aliases[normalized]


def dataset_config_paths(
    dataset: str,
) -> tuple[Path, dict[str, tuple[Path, str, bool]]]:
    dataset = normalize_campaign_dataset(dataset)
    if dataset == "bnci2014_001":
        return BCI_TRANSFORMER_CONFIG, {
            "csp_lda": (BCI_CSP_CONFIG, "csp_lda_baseline.py", False),
            "mdm": (BCI_MDM_CONFIG, "mdm_baseline.py", True),
            "spdnet": (BCI_SPDNET_CONFIG, "spdnet_baseline.py", True),
            "eegnet_subject_wise": (
                BCI_EEGNET_CONFIG,
                "eegnet_baseline.py",
                True,
            ),
            "eegnet_transfer": (
                BCI_EEGNET_CONFIG,
                "eegnet_baseline.py",
                True,
            ),
        }
    return TRANSFORMER_CONFIG, {
        "csp_lda": (CSP_CONFIG, "csp_lda_baseline.py", False),
        "mdm": (MDM_CONFIG, "mdm_baseline.py", True),
        "spdnet": (SPDNET_CONFIG, "spdnet_baseline.py", True),
        "eegnet_subject_wise": (EEGNET_CONFIG, "eegnet_baseline.py", True),
        "eegnet_transfer": (EEGNET_CONFIG, "eegnet_baseline.py", True),
    }


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


def parse_task_types(value: Any) -> tuple[str, ...]:
    """Accept a short CLI form or the equivalent train-grid YAML form."""

    if isinstance(value, tuple):
        parsed = list(value)
    elif isinstance(value, list):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise argparse.ArgumentTypeError("task types cannot be empty")
        loaded = yaml.safe_load(text)
        parsed = loaded if isinstance(loaded, list) else text.split(",")
    if len(parsed) == 1 and isinstance(parsed[0], list):
        parsed = parsed[0]
    normalized = tuple(str(item).strip() for item in parsed if str(item).strip())
    if normalized not in TASK_CLASS_NAMES:
        choices = "unilateral_fist,both; unilateral_fist; or both"
        raise argparse.ArgumentTypeError(
            f"unsupported task types {normalized!r}; choose {choices}"
        )
    return normalized


def make_four_class_config(
    base: dict[str, Any],
    output_dir: Path,
    subjects: str | None,
    task_types: tuple[str, ...],
    dataset: str = "physionet_mi",
) -> dict[str, Any]:
    config = deepcopy(base)
    data_cfg = config.setdefault("data", {})
    dataset = normalize_campaign_dataset(dataset)
    if dataset == "physionet_mi":
        data_cfg["dataset"] = ["physionet_mi"]
        data_cfg["task_types"] = [list(task_types)]
    else:
        data_cfg["dataset"] = ["bnci2014_001"]
        data_cfg["events"] = "left_hand,right_hand,feet,tongue"
        data_cfg["sessions"] = "all"
    if subjects is not None:
        data_cfg["subjects"] = subjects
    config.setdefault("output", {})["dir"] = str(output_dir.resolve())
    return config


def resolve_campaign_subjects(
    subjects: str | None,
    dataset: str = "physionet_mi",
) -> Any:
    """Resolve ``--subjects all`` to the configured pretraining cohort."""

    if subjects is None or subjects.strip().lower() != "all":
        return subjects
    transformer_path, _ = dataset_config_paths(dataset)
    transformer_config = load_yaml(transformer_path)
    pretrain_subjects = transformer_config.get("data", {}).get(
        "pretrain_subjects"
    )
    if pretrain_subjects is None or str(pretrain_subjects).strip() == "":
        raise ValueError(
            "--subjects all requires data.pretrain_subjects in "
            f"{transformer_path}."
        )
    return pretrain_subjects


def make_transformer_config(
    base: dict[str, Any],
    output_dir: Path,
    subjects: str | None,
    task_types: tuple[str, ...],
    *,
    resting_length: float,
    metric: str = "learnable-metric",
    classifier_type: str = "mdm",
    single_band: bool = False,
    dataset: str = "physionet_mi",
) -> dict[str, Any]:
    config = make_four_class_config(
        base,
        output_dir,
        subjects,
        task_types,
        dataset=dataset,
    )
    config["data"]["epoch_slice"] = [[-float(resting_length), 4.0]]
    if single_band:
        config["data"]["filter_bank"] = [[[8, 30]]]
    config["model"]["metric"] = [metric]
    config["model"]["classifier_type"] = [classifier_type]
    return config


def build_configs(
    campaign_dir: Path,
    subjects: str | None,
    task_types: tuple[str, ...] = DEFAULT_TASK_TYPES,
    dataset: str = "physionet_mi",
) -> dict[str, tuple[Path, Path, bool]]:
    dataset = normalize_campaign_dataset(dataset)
    task_types = parse_task_types(task_types)
    transformer_path, baseline_specs = dataset_config_paths(dataset)
    transformer_base = load_yaml(transformer_path)
    if subjects is None and transformer_base.get("data", {}).get("subjects") in {
        None,
        "",
    }:
        raise ValueError(
            "No target subjects configured. Use --subjects 1 or --subjects 1-10, "
            f"or set data.subjects in {transformer_path.name}."
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
                task_types,
                resting_length=rest,
                dataset=dataset,
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
                task_types,
                resting_length=rest,
                single_band=True,
                dataset=dataset,
            ),
            transformer_runner,
            True,
        )

    specs["spd_transformer_fixed_log_euclidean"] = (
        make_transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_fixed_log_euclidean",
            subjects,
            task_types,
            resting_length=1.0,
            metric="log-euclidean",
            dataset=dataset,
        ),
        transformer_runner,
        True,
    )
    specs["spd_transformer_linear_head"] = (
        make_transformer_config(
            transformer_base,
            raw_dir / "spd_transformer_linear_head",
            subjects,
            task_types,
            resting_length=1.0,
            classifier_type="pooling",
            dataset=dataset,
        ),
        transformer_runner,
        True,
    )

    for name, (base_path, filename, uses_device) in baseline_specs.items():
        config = make_four_class_config(
            load_yaml(base_path),
            raw_dir / name,
            subjects,
            task_types,
            dataset=dataset,
        )
        config.setdefault("data", {})["allow_subject_overlap"] = [True]
        training_cfg = config.setdefault("training", {})
        training_cfg["train_size"] = [0.7]
        training_cfg["test_size"] = [0.3]
        training_cfg.pop("held_out_run_validation_size", None)
        if name.startswith("eegnet_"):
            training_cfg["protocol"] = [
                "transfer" if name == "eegnet_transfer" else "subject_wise"
            ]
            training_cfg["subject_wise_n_splits"] = [1]
            config["data"]["pretrain_subjects"] = transformer_base["data"][
                "pretrain_subjects"
            ]
        else:
            training_cfg["subject_specific"] = [True]
        specs[name] = (
            config,
            PROJECT_ROOT / "src" / "baselines" / filename,
            uses_device,
        )

    commands: dict[str, tuple[Path, Path, bool]] = {}
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "protocol": (
            "subject-specific stratified trial-level train/test split; "
            "all runs pooled; no validation dataset; EEGNet transfer excludes "
            "the target from other-subject pretraining"
        ),
        "task_types": list(task_types),
        "expected_classes": sorted(
            BCI_IV_2A_CLASS_NAMES
            if dataset == "bnci2014_001"
            else TASK_CLASS_NAMES[task_types]
        ),
        "subject_aggregation": "one pooled trial-random test partition per subject",
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
    # Baseline runners create ``<timestamp>/run_<grid-id>/`` beneath the
    # configured output directory. Keep the legacy direct ``run_*`` layout
    # readable as well, so existing campaigns can be summarized in place.
    tables = [
        *root.glob("*/run_*/per_subject_summary.csv"),
        *root.glob("run_*/per_subject_summary.csv"),
    ]
    table = only_path(tables, str(root))
    summary = read_json(table.parent / "summary.json")
    return RunRecord(
        config=dict(summary.get("config", {})),
        subjects=read_subject_table(table),
        class_names=list(summary.get("class_names", [])),
        source=table,
    )


def validate_record(record: RunRecord, expected_classes: set[str]) -> None:
    if set(record.class_names) != expected_classes:
        raise ValueError(
            f"Expected {sorted(expected_classes)} in {record.source}, got "
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
    manifest = read_json(campaign_dir / "manifest.json")
    expected_classes = set(manifest.get("expected_classes", []))
    if not expected_classes:
        task_types = parse_task_types(
            manifest.get("task_types", DEFAULT_TASK_TYPES)
        )
        expected_classes = TASK_CLASS_NAMES[task_types]
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
    eegnet_subject_wise = load_baseline_record(raw / "eegnet_subject_wise")
    eegnet_transfer = load_baseline_record(raw / "eegnet_transfer")
    records = [
        *complete.values(),
        *single.values(),
        fixed,
        linear,
        csp,
        mdm,
        spdnet,
        eegnet_subject_wise,
        eegnet_transfer,
    ]
    for record in records:
        validate_record(record, expected_classes)

    proposed = complete[1.0]
    baseline_rows = [
        result_row("Model", "CSP+LDA", csp, proposed),
        result_row("Model", "MDM", mdm, proposed),
        result_row("Model", "SPDNet", spdnet, proposed),
        result_row(
            "Model",
            "EEGNet (subject-wise)",
            eegnet_subject_wise,
            proposed,
        ),
        result_row(
            "Model",
            "EEGNet (pretrain + fine-tune)",
            eegnet_transfer,
            proposed,
        ),
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
        ("EEGNet (subject-wise)", eegnet_subject_wise),
        ("EEGNet (pretrain + fine-tune)", eegnet_transfer),
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
    parser.add_argument(
        "--dataset",
        type=normalize_campaign_dataset,
        default="physionet_mi",
        metavar="DATASET",
        help="physionet_mi (default) or bci_iv_2a",
    )
    parser.add_argument("--device", help="Training device, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--subjects",
        help=(
            "Target subjects, e.g. 1 or 1-10; use 'all' to target every "
            "subject in Transformer data.pretrain_subjects. Overrides each "
            "generated config."
        ),
    )
    parser.add_argument(
        "--task-types",
        type=parse_task_types,
        default=DEFAULT_TASK_TYPES,
        metavar="TASKS",
        help=(
            "PhysioNet task selection: unilateral_fist,both (four classes), "
            "unilateral_fist (left/right), or both (hands/feet). Ignored for "
            "BCI IV-2a, whose four classes are fixed."
        ),
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
    campaign_subjects = resolve_campaign_subjects(args.subjects, args.dataset)
    commands = build_configs(
        campaign_dir,
        campaign_subjects,
        args.task_types,
        dataset=args.dataset,
    )
    class_text = (
        "left_hand,right_hand,feet,tongue"
        if args.dataset == "bnci2014_001"
        else ",".join(args.task_types)
    )
    print(
        f"Subject-specific {args.dataset} campaign "
        f"({class_text}): {campaign_dir}"
    )
    if args.subjects is not None:
        print(f"Target subjects: {campaign_subjects}")
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
