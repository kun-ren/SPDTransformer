from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from script.run_bci_iv_2a_all_models import (
    EXPECTED_CLASSES,
    MODEL_SPECS,
    PROJECT_ROOT,
    build_campaign,
    count_runs,
    load_yaml,
    summarize_campaign,
)
from src.baselines.baseline_utils import expand_data_training_experiments
from src.baselines.mdm_baseline import expand_mdm_experiments
from src.training.train import expand_experiments as expand_transformer_experiments


def expanded_configs(spec, config):
    if spec.runner_kind == "transformer":
        return expand_transformer_experiments(config)
    if spec.runner_kind == "mdm":
        return expand_mdm_experiments(config)
    return expand_data_training_experiments(config)


def write_fake_run(run_dir: Path, spec, config: dict) -> None:
    run_dir.mkdir(parents=True)
    folds = []
    for fold in range(1, 6):
        metrics = {
            "accuracy": 0.60 + fold * 0.01,
            "macro_f1": 0.55 + fold * 0.01,
            "cohen_kappa": 0.40 + fold * 0.01,
        }
        row = {"fold": fold, "n_train": 80, "n_test": 20}
        if spec.runner_kind == "transformer":
            row.update({f"test_{key}": value for key, value in metrics.items()})
            row["class_names"] = sorted(EXPECTED_CLASSES)
        else:
            row.update(metrics)
        folds.append(row)

    if spec.runner_kind == "transformer":
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        summary_name = "five_fold_summary.json"
        summary = {"folds": folds, "evaluation": {"n_splits": 5}}
    else:
        with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle)
        summary_name = "summary.json"
        summary = {
            "class_names": sorted(EXPECTED_CLASSES),
            "folds": folds,
            "evaluation": {"n_splits": 5},
        }

    with (run_dir / summary_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle)


def test_campaign_configs_and_unified_summary():
    with TemporaryDirectory(dir=PROJECT_ROOT / "tmp") as temporary_dir:
        run_campaign_summary_test(Path(temporary_dir))


def run_campaign_summary_test(campaign_dir: Path) -> None:
    commands = build_campaign(campaign_dir, list(MODEL_SPECS))
    assert list(commands) == [spec.key for spec in MODEL_SPECS]

    for spec, config_path in commands.values():
        campaign_config = load_yaml(config_path)
        assert count_runs(spec, campaign_config) == 2
        experiments = expanded_configs(spec, campaign_config)
        assert {
            bool(experiment["data"]["allow_subject_overlap"])
            for experiment in experiments
        } == {True, False}

        timestamp_dir = campaign_dir / "raw" / spec.key / "20260822_120000"
        for index, experiment in enumerate(experiments, start=1):
            write_fake_run(
                timestamp_dir / f"run_{index:03d}_fake",
                spec,
                experiment,
            )

    aggregate_path, fold_path, json_path = summarize_campaign(campaign_dir)
    assert aggregate_path.exists()
    assert fold_path.exists()
    assert json_path.exists()

    with aggregate_path.open("r", encoding="utf-8", newline="") as handle:
        aggregate_rows = list(csv.DictReader(handle))
    with fold_path.open("r", encoding="utf-8", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))

    assert len(aggregate_rows) == len(MODEL_SPECS) * 2
    assert len(fold_rows) == len(MODEL_SPECS) * 2 * 5
    assert {row["split_strategy"] for row in aggregate_rows} == {
        "sample_level",
        "subject_disjoint",
    }
    assert {row["model_key"] for row in aggregate_rows} == {
        spec.key for spec in MODEL_SPECS
    }
