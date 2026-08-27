from __future__ import annotations

import numpy as np
import pytest

from src.baselines.baseline_utils import make_subject_specific_trial_splits
from src.training.train_pretrain_finetune_loro import split_target_train_test


def test_baseline_subject_trial_splits_pool_runs_and_have_no_validation():
    y = np.tile(np.asarray([0, 1], dtype=np.int64), 10)
    subjects = np.asarray(["S001"] * 10 + ["S002"] * 10)

    splits = make_subject_specific_trial_splits(
        y,
        subjects,
        train_size=0.7,
        test_size=0.3,
        seed=42,
    )

    assert len(splits) == 2
    for subject, split_id, train_idx, validation_idx, test_idx in splits:
        assert split_id == 1
        assert len(train_idx) == 7
        assert len(validation_idx) == 0
        assert len(test_idx) == 3
        assert not set(train_idx) & set(test_idx)
        assert set(train_idx) | set(test_idx) == set(np.flatnonzero(subjects == subject))
        assert set(np.unique(y[train_idx])) == {0, 1}
        assert set(np.unique(y[test_idx])) == {0, 1}


def test_transformer_target_split_uses_every_trial_once():
    y = np.tile(np.arange(4, dtype=np.int64), 10)
    indices = np.arange(len(y), dtype=np.int64)

    train_idx, test_idx = split_target_train_test(
        indices,
        y,
        train_size=0.7,
        test_size=0.3,
        seed=42,
    )

    assert len(train_idx) == 28
    assert len(test_idx) == 12
    assert not set(train_idx) & set(test_idx)
    assert set(train_idx) | set(test_idx) == set(indices)
    assert set(np.unique(y[train_idx])) == {0, 1, 2, 3}
    assert set(np.unique(y[test_idx])) == {0, 1, 2, 3}


def test_trial_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_target_train_test(
            np.arange(10),
            np.tile([0, 1], 5),
            train_size=0.7,
            test_size=0.2,
            seed=42,
        )
