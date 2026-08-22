from __future__ import annotations

import numpy as np

from src.datasets.PhysioNetMI_subject_specific import group_trials_by_subject
from src.training.train_subject_specific import make_subject_folds, parse_subject_spec


def test_group_trials_by_subject_keeps_subjects_separate():
    x = np.stack([np.eye(3) * (index + 1) for index in range(24)])
    y = np.tile(np.arange(4), 6)
    subjects = np.asarray(["S001"] * 12 + ["S002"] * 12)

    datasets, skipped = group_trials_by_subject(
        x,
        y,
        subjects,
        ["feet", "hands", "left_hand", "right_hand"],
        min_trials_per_class=3,
    )

    assert skipped == []
    assert [dataset.subject_id for dataset in datasets] == ["S001", "S002"]
    assert all(dataset.n_trials == 12 for dataset in datasets)
    assert all(dataset.class_counts == (3, 3, 3, 3) for dataset in datasets)


def test_incomplete_subject_can_be_skipped_with_audit_record():
    x = np.stack([np.eye(2) for _ in range(11)])
    y = np.asarray([0, 1, 2, 3] * 2 + [0, 1, 2])
    subjects = np.asarray(["S001"] * 8 + ["S002"] * 3)

    datasets, skipped = group_trials_by_subject(
        x,
        y,
        subjects,
        ["feet", "hands", "left_hand", "right_hand"],
        min_trials_per_class=2,
        on_incomplete_subject="skip",
    )

    assert [dataset.subject_id for dataset in datasets] == ["S001"]
    assert skipped[0]["subject_id"] == "S002"
    assert "missing classes" in skipped[0]["reason"]


def test_nested_subject_folds_are_disjoint_and_cover_each_trial_once():
    y = np.tile(np.arange(4), 10)
    folds = make_subject_folds(
        y,
        outer_splits=5,
        validation_size=0.25,
        seed=42,
    )

    test_indices = []
    for train_idx, val_idx, test_idx in folds:
        assert not (set(train_idx) & set(val_idx))
        assert not (set(train_idx) & set(test_idx))
        assert not (set(val_idx) & set(test_idx))
        assert set(np.unique(y[train_idx])) == {0, 1, 2, 3}
        assert set(np.unique(y[val_idx])) == {0, 1, 2, 3}
        assert set(np.unique(y[test_idx])) == {0, 1, 2, 3}
        test_indices.extend(test_idx.tolist())

    assert sorted(test_indices) == list(range(len(y)))


def test_parse_subject_spec_supports_ranges_and_prefixed_ids():
    assert parse_subject_spec("1-3,S005") == [1, 2, 3, 5]
