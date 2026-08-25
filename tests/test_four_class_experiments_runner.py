from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from script.run_four_class_experiments import (
    build_configs,
    load_baseline_record,
    resolve_campaign_subjects,
)


def test_all_subjects_resolves_to_physionet_pretrain_cohort():
    assert resolve_campaign_subjects("all") == (
        "1-87,90-91,93-99,101-103,105,107-109"
    )


def test_explicit_subjects_are_unchanged():
    assert resolve_campaign_subjects("1-3,8") == "1-3,8"
    assert resolve_campaign_subjects(None) is None


def test_bci_all_subjects_resolves_to_full_nine_subject_cohort():
    assert resolve_campaign_subjects("all", "bci_iv_2a") == "1-9"


def test_bci_campaign_generates_dataset_aligned_configs():
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        commands = build_configs(
            root,
            "1-2",
            dataset="bci_iv_2a",
        )

        assert len(commands) == 11
        for name, (config_path, _runner, _uses_device) in commands.items():
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            assert config["data"]["dataset"] == ["bnci2014_001"]
            assert config["data"]["subjects"] == "1-2"
            assert config["data"]["events"] == (
                "left_hand,right_hand,feet,tongue"
            )
            if name in {"csp_lda", "mdm", "spdnet"}:
                assert config["training"]["subject_specific"] == [True]
                assert config["data"]["allow_subject_overlap"] == [True]
            else:
                assert config["data"]["pretrain_subjects"] == "1-9"


def test_baseline_record_supports_timestamp_run_layout():
    with TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        run_dir = root / "20260824_032518" / "run_001_98b6c4a79"
        run_dir.mkdir(parents=True)
        with (run_dir / "per_subject_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "Subject",
                    "Trials",
                    "Accuracy (%)",
                    "Balanced Accuracy (%)",
                    "Macro-F1",
                    "Cohen’s κ",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Subject": "S001",
                    "Trials": 24,
                    "Accuracy (%)": 58.33,
                    "Balanced Accuracy (%)": 58.33,
                    "Macro-F1": 0.4958,
                    "Cohen’s κ": 0.1667,
                }
            )
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": {"data": {"dataset": ["physionet_mi"]}},
                    "class_names": ["feet", "hands", "left_hand", "right_hand"],
                },
                handle,
            )

        record = load_baseline_record(root)

        assert record.source == run_dir / "per_subject_summary.csv"
        assert record.subjects[0]["Subject"] == "S001"
