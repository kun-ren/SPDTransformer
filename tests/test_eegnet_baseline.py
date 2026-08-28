from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.baselines.baseline_utils import expand_data_training_experiments, load_yaml
from src.baselines.eegnet_baseline import (
    EEGNet,
    extract_single_trial_eeg,
    resolve_target_labels,
    split_data_config_for_protocol,
    subject_wise_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_eegnet_forward_shape_and_official_max_norm_constraints():
    model = EEGNet(
        num_classes=4,
        channels=8,
        samples=320,
        kernel_length=40,
        f1=8,
        depth_multiplier=2,
        f2=16,
    )
    logits = model(torch.randn(3, 1, 8, 320))
    assert logits.shape == (3, 4)

    with torch.no_grad():
        model.depthwise_conv.weight.fill_(10.0)
        model.classifier.weight.fill_(10.0)
    model.apply_max_norm_()
    depthwise_norms = model.depthwise_conv.weight.flatten(1).norm(dim=1)
    classifier_norms = model.classifier.weight.norm(dim=1)
    assert torch.all(depthwise_norms <= 1.0 + 1.0e-6)
    assert torch.all(classifier_norms <= 0.25 + 1.0e-6)


def test_eegnet_requires_one_segment_and_one_band():
    x = np.zeros((5, 1, 1, 8, 320), dtype=np.float32)
    assert extract_single_trial_eeg(x).shape == (5, 8, 320)


def test_subject_wise_default_matches_trial_random_70_30_protocol():
    y = np.tile(np.arange(2, dtype=np.int64), 20)
    subjects = np.asarray(["S001"] * 20 + ["S002"] * 20)
    splits = subject_wise_splits(
        y,
        subjects,
        n_splits=1,
        train_size=0.7,
        test_size=0.3,
        seed=42,
    )
    assert len(splits) == 2
    for subject, _fold, train_idx, validation_idx, test_idx in splits:
        assert len(train_idx) == 14
        assert len(validation_idx) == 0
        assert len(test_idx) == 6
        assert not np.intersect1d(train_idx, test_idx).size
        assert np.all(subjects[train_idx] == subject)
        assert np.all(subjects[test_idx] == subject)


def test_optional_subject_wise_four_fold_covers_each_trial_once_as_test():
    y = np.tile(np.arange(2, dtype=np.int64), 20)
    subjects = np.asarray(["S001"] * 20 + ["S002"] * 20)
    splits = subject_wise_splits(
        y,
        subjects,
        n_splits=4,
        train_size=0.7,
        test_size=0.3,
        seed=42,
    )
    assert len(splits) == 8
    for subject in ("S001", "S002"):
        subject_test = np.concatenate(
            [test_idx for item, _fold, _train, _val, test_idx in splits if item == subject]
        )
        expected = np.flatnonzero(subjects == subject)
        assert np.array_equal(np.sort(subject_test), expected)


def test_transfer_loads_pretraining_cohort_and_resolves_target():
    load_cfg, targets = split_data_config_for_protocol(
        {"subjects": "1", "pretrain_subjects": "1-3"},
        "transfer",
    )
    assert targets == [1]
    assert load_cfg["subjects"] == [1, 2, 3]
    labels = np.asarray(["S001", "S002", "S003"])
    assert resolve_target_labels(targets, labels) == ["S001"]
    assert set(labels[labels != "S001"]) == {"S002", "S003"}


def test_paper_protocol_resolves_targets_after_author_exclusions():
    load_cfg, targets = split_data_config_for_protocol(
        {
            "subjects": "1-109",
            "eegnet_excluded_subjects": "88,92,100,104",
        },
        "paper_global_ss_tl",
    )
    available = np.asarray(
        [
            f"S{subject:03d}"
            for subject in range(1, 110)
            if subject not in {88, 92, 100, 104}
        ]
    )

    assert load_cfg["subjects"] == "1-109"
    assert targets is None
    assert resolve_target_labels(targets, available) == available.tolist()


def test_eegnet_physionet_config_matches_trial_pooled_transfer_protocol():
    config = load_yaml(PROJECT_ROOT / "configs" / "eegnet_physionet.yaml")
    experiments = expand_data_training_experiments(config)
    assert [item["training"]["protocol"] for item in experiments] == ["transfer"]
    assert experiments[0]["training"]["pretrain_train_size"] == 0.7
    assert experiments[0]["training"]["pretrain_validation_size"] == 0.15
    assert experiments[0]["training"]["pretrain_test_size"] == 0.15
    assert experiments[0]["training"]["train_size"] == 0.7
    assert experiments[0]["training"]["test_size"] == 0.3
    assert experiments[0]["data"]["task_types"] == ["unilateral_fist"]
    assert experiments[0]["data"]["epoch_slice"] == [-2.0, 4.0]
    assert experiments[0]["data"]["segment_slice"] == [6.0, None]
    assert config["model"]["F1"] == 8
    assert config["model"]["D"] == 2
    assert config["model"]["F2"] == 16
    assert config["model"]["within_subject_dropout"] == 0.5
    assert config["model"]["cross_subject_dropout"] == 0.25
