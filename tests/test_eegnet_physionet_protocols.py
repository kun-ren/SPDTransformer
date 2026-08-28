from pathlib import Path

import numpy as np
import torch

from src.baselines.baseline_utils import expand_data_training_experiments, load_yaml
from src.baselines.eegnet_baseline import (
    EEGNet,
    global_subject_folds,
    parse_learning_rate_schedule,
    stratified_train_validation_test_split,
)
from src.training.train_pretrain_finetune_loro import make_pretrain_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_paper_configuration_expands_to_two_three_and_four_classes():
    config = load_yaml(PROJECT_ROOT / "configs" / "eegnet_physionet_paper.yaml")
    experiments = expand_data_training_experiments(config)

    assert [item["data"]["eegnet_num_classes"] for item in experiments] == [2, 3, 4]
    assert {item["training"]["protocol"] for item in experiments} == {
        "paper_global_ss_tl"
    }


def test_paper_global_fold_is_subject_held_out_and_non_shuffled():
    excluded = {88, 92, 100, 104}
    subjects = [f"S{subject:03d}" for subject in range(1, 110) if subject not in excluded]
    subject_labels = np.repeat(subjects, 4)

    folds = global_subject_folds(subject_labels, n_splits=5)

    assert len(folds) == 5
    for _fold, train_idx, test_idx, train_subjects, test_subjects in folds:
        assert len(train_subjects) == 84
        assert len(test_subjects) == 21
        assert len(train_idx) == 84 * 4
        assert len(test_idx) == 21 * 4
        assert not set(train_subjects) & set(test_subjects)
    assert folds[0][4] == subjects[:21]


def test_transfer_pretrain_split_is_pooled_stratified_70_15_15():
    y = np.repeat(np.arange(2), 100)
    indices = np.arange(len(y))

    train_idx, validation_idx, test_idx = stratified_train_validation_test_split(
        y,
        indices,
        train_size=0.7,
        validation_size=0.15,
        test_size=0.15,
        seed=42,
    )
    expected = make_pretrain_split(
        indices,
        y,
        validation_size=0.15,
        test_size=0.15,
        seed=42,
    )

    # Match the SPDTransformer helper exactly, including sklearn's rounding.
    assert (len(train_idx), len(validation_idx), len(test_idx)) == tuple(
        len(split) for split in expected
    )
    assert len(train_idx) + len(validation_idx) + len(test_idx) == len(indices)
    assert all(
        np.array_equal(actual, expected_split)
        for actual, expected_split in zip(
            (train_idx, validation_idx, test_idx),
            expected,
        )
    )
    assert set(train_idx).isdisjoint(validation_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(validation_idx).isdisjoint(test_idx)
    for split in (train_idx, validation_idx, test_idx):
        counts = np.bincount(y[split], minlength=2)
        assert counts.max() - counts.min() <= 1


def test_paper_learning_rate_schedule_and_model_shape():
    schedule = parse_learning_rate_schedule(
        "0:0.01,20:0.001,50:0.0001",
        default_learning_rate=0.01,
    )
    assert schedule == [(0, 0.01), (20, 0.001), (50, 0.0001)]

    model = EEGNet(
        num_classes=4,
        channels=64,
        samples=480,
        kernel_length=128,
        pool1_length=8,
        pool2_length=8,
        dropout_rate=0.2,
    )
    assert model(torch.zeros(2, 1, 64, 480)).shape == (2, 4)
