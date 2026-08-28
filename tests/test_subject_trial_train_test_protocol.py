from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from src.baselines.baseline_utils import make_subject_specific_trial_splits
import src.training.train_pretrain_finetune_loro as protocol_runner
from src.training.train_pretrain_finetune_loro import (
    make_global_train_validation_split,
    split_global_subjects,
    split_target_train_test,
    validate_global_protocol,
)


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


def test_global_subject_split_is_deterministic_disjoint_and_complete():
    subjects = [f"S{subject:03d}" for subject in range(1, 11)]

    train_subjects, test_subjects = split_global_subjects(
        subjects,
        train_size=0.7,
        test_size=0.3,
        seed=42,
    )
    repeated = split_global_subjects(
        subjects,
        train_size=0.7,
        test_size=0.3,
        seed=42,
    )

    assert (train_subjects, test_subjects) == repeated
    assert len(train_subjects) == 7
    assert len(test_subjects) == 3
    assert not set(train_subjects) & set(test_subjects)
    assert set(train_subjects) | set(test_subjects) == set(subjects)


def test_global_validation_contains_only_global_train_subject_trials():
    subjects = np.repeat([f"S{subject:03d}" for subject in range(1, 7)], 8)
    y = np.tile(np.repeat([0, 1], 4), 6)
    train_subjects, test_subjects = split_global_subjects(
        np.unique(subjects).tolist(),
        train_size=2 / 3,
        test_size=1 / 3,
        seed=7,
    )
    global_train_all = np.flatnonzero(np.isin(subjects, train_subjects))
    train_idx, validation_idx = make_global_train_validation_split(
        global_train_all,
        y,
        validation_size=0.25,
        seed=7,
    )

    validate_global_protocol(
        y,
        subjects,
        train_subjects,
        test_subjects,
        num_classes=2,
    )
    assert not set(train_idx) & set(validation_idx)
    assert set(train_idx) | set(validation_idx) == set(global_train_all)
    assert set(subjects[train_idx]) <= set(train_subjects)
    assert set(subjects[validation_idx]) <= set(train_subjects)
    assert not set(subjects[train_idx]) & set(test_subjects)
    assert not set(subjects[validation_idx]) & set(test_subjects)


def test_nested_protocol_trains_one_global_and_restarts_for_each_subject(
    tmp_path,
    monkeypatch,
):
    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x, return_aux=False):
            del return_aux
            return self.linear(x.flatten(1)), {}

    rng = np.random.default_rng(7)
    x = rng.normal(size=(48, 1, 1, 2, 2)).astype(np.float32)
    y = np.tile(np.repeat([0, 1], 4), 6).astype(np.int64)
    subjects = np.repeat([f"S{subject:03d}" for subject in range(1, 7)], 8)
    runs = np.tile(np.asarray([4, 4, 8, 8, 12, 12, 4, 8]), 6)
    monkeypatch.setattr(
        protocol_runner,
        "load_or_preprocess_spd_with_runs",
        lambda _data_cfg, _cache_dir: (x, y, subjects, runs, ["left", "right"]),
    )
    monkeypatch.setattr(
        protocol_runner,
        "make_model",
        lambda _cfg, _x, _classes, *, device, dtype: TinyClassifier().to(
            device=device,
            dtype=dtype,
        ),
    )
    initial_fine_tune_states = []
    original_fixed_epochs = protocol_runner.train_fixed_epochs

    def recording_fixed_epochs(model, *args, **kwargs):
        initial_fine_tune_states.append(
            {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        )
        return original_fixed_epochs(model, *args, **kwargs)

    monkeypatch.setattr(protocol_runner, "train_fixed_epochs", recording_fixed_epochs)
    output_dir = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "dataset": ["physionet_mi"],
            "root_dir": ["unused"],
            "subjects": "1-6",
            "filter_bank": [[[8, 30]]],
            "task_types": [["unilateral_fist"]],
            "epoch_slice": [[0.0, 1.0]],
            "segment_slice": [[1.0, None]],
            "imaged": [True],
            "executed": [False],
        },
        "model": {"dummy": [1]},
        "pretrain": {
            "subject_train_size": [2 / 3],
            "subject_test_size": [1 / 3],
            "subject_split_seed": [42],
            "subject_split_shuffle": [True],
            "validation_size": [0.25],
            "epochs": [1],
            "batch_size": [8],
            "precision": ["float32"],
            "learning_rate": [0.01],
            "stiefel_learning_rate": [0.001],
            "lr_scheduler": ["none"],
            "weight_decay": [0.0],
            "gradient_clip_norm": [1.0],
            "early_stopping_patience": [2],
            "early_stopping_min_delta": [0.0],
            "condition_regularization_weight": [0.0],
            "seed": [42],
            "num_workers": [0],
            "pin_memory": [False],
            "allow_tf32": [False],
        },
        "fine_tune": {
            "train_size": [0.5],
            "test_size": [0.5],
            "epochs": [1],
            "batch_size": [4],
            "precision": ["float32"],
            "learning_rate": [0.01],
            "stiefel_learning_rate": [0.001],
            "lr_scheduler": ["none"],
            "weight_decay": [0.0],
            "gradient_clip_norm": [1.0],
            "condition_regularization_weight": [0.0],
            "num_workers": [0],
            "pin_memory": [False],
            "allow_tf32": [False],
        },
        "output": {
            "dir": str(output_dir),
            "dataset_cache_dir": str(tmp_path / "cache"),
            "save_pretrained_checkpoints": True,
            "save_fine_tuned_checkpoints": False,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert protocol_runner.main(["--config", str(config_path), "--device", "cpu"]) == 0

    run_dir = next(output_dir.iterdir())
    checkpoint = torch.load(
        run_dir / "global" / "global_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    split = json.loads((run_dir / "global" / "split.json").read_text(encoding="utf-8"))
    assert len(split["global_train_subjects"]) == 4
    assert len(split["global_test_subjects"]) == 2
    assert not set(split["global_train_subjects"]) & set(split["global_test_subjects"])
    assert len(initial_fine_tune_states) == 2
    for initial_state in initial_fine_tune_states:
        for name, expected in checkpoint["model_state_dict"].items():
            assert torch.equal(initial_state[name], expected)
