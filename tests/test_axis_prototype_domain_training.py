import csv
import json

import numpy as np
import torch

import src.training.train as trainer
import src.training.train_pretrain_finetune_loro as adaptation


def tiny_model_config():
    return {
        "head_nums": 2, "attention_dim": "3,3", "depth": 2,
        "stage_transition": True, "metric": "learnable-metric",
        "learnable_metric_rank": 2, "classifier_type": "mdm",
        "pooling": "weighted", "use_position_bias": False,
        "layer_norm_affine": False, "add_norm_type": "sequence_add_norm",
        "share_metric_across_layers": True,
        "domain_adversarial": True, "domain_hidden_dim": 4, "domain_dropout": 0.0,
    }


def tiny_training_config():
    return {
        "epochs": 1, "batch_size": 32, "precision": "float32",
        "learning_rate": 0.001, "stiefel_learning_rate": 0.0001,
        "lr_scheduler": "none", "weight_decay": 0.01,
        "early_stopping_patience": 2, "condition_regularization_weight": 0.0,
        "prototype_intra_weight": 0.001, "prototype_inter_weight": 0.01,
        "prototype_margin": 1.0, "domain_adversarial_max_weight": 0.03,
        "domain_adversarial_schedule": "constant",
        "domain_adversarial_warmup_epochs": 0, "seed": 42,
    }


def tiny_dataset():
    generator = np.random.default_rng(23)
    subjects = np.repeat(["S001", "S002", "S003"], 24)
    runs = np.tile(np.repeat([4, 8, 12], 8), 3)
    targets = np.tile([0, 1], len(subjects) // 2)
    factors = 0.1 * generator.normal(size=(len(subjects), 2, 2, 2, 3, 3))
    x = (factors @ factors.swapaxes(-1, -2) + np.eye(3)).astype(np.float32)
    return x, targets, subjects, runs, ["left", "right"]


def test_train_fold_saves_source_mapping_and_auxiliary_metrics(tmp_path):
    x, y, subjects, _, names = tiny_dataset()
    train_idx = np.flatnonzero(subjects != "S001")
    test_idx = np.flatnonzero(subjects == "S001")
    config = {
        "data": {"allow_subject_overlap": False, "seed": 42},
        "model": tiny_model_config(), "training": tiny_training_config(),
    }
    trainer.train_fold(
        1, 1, 3, config, x, y, subjects, names, tmp_path,
        train_idx, test_idx, torch.device("cpu"),
    )
    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
    assert checkpoint["domain_subject_mapping"] == {"S002": 0, "S003": 1}
    with (tmp_path / "history.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["train_domain_loss"]) > 0
    assert float(row["train_domain_adversarial_coefficient"]) > 0
    assert float(row["train_prototype_intra_loss"]) > 0
    assert 0 <= float(row["test_accuracy"]) <= 1


def test_pretrain_excludes_target_and_finetuning_restarts_with_frozen_domain_head(
        tmp_path, monkeypatch,
):
    config = {
        "data": {
            "dataset": "physionet_mi", "subjects": "1", "pretrain_subjects": "1-3",
            "filter_bank": [[[8, 13]]], "epoch_slice": [[-2.0, 4.0]],
            "segment_slice": [[0.5, 0.25]], "task_types": [["unilateral_fist"]],
        },
        "model": tiny_model_config(),
        "pretrain": tiny_training_config(),
        "fine_tune": tiny_training_config(),
        "output": {
            "dir": str(tmp_path / "results"),
            "dataset_cache_dir": str(tmp_path / "cache"),
            "save_pretrained_checkpoints": True,
            "save_fine_tuned_checkpoints": True,
        },
    }
    monkeypatch.setattr(adaptation, "load_config", lambda _path: config)
    monkeypatch.setattr(adaptation, "load_or_preprocess_spd_with_runs", lambda *_: tiny_dataset())
    real_train = adaptation.train_with_early_stopping
    stages = []
    pretrained = None

    def checked_train(model, train_loader, validation_loader, cfg, **kwargs):
        nonlocal pretrained
        stage = kwargs["stage_name"]
        if stage == "pretrain":
            assert all(p.requires_grad for p in model.domain_head.parameters())
            domain_ids = torch.cat([batch[2] for batch in train_loader])
            assert set(domain_ids.tolist()) == {0, 1}
            subset = train_loader.dataset
            np.testing.assert_array_equal(
                subset.dataset.domain_labels[:24].numpy(), np.full(24, -1),
            )
        else:
            assert not any(p.requires_grad for p in model.domain_head.parameters())
            assert pretrained is not None
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor.cpu(), pretrained[name], rtol=0, atol=0)
        result = real_train(model, train_loader, validation_loader, cfg, **kwargs)
        if stage == "pretrain":
            pretrained = result[0]
        else:
            for name, tensor in model.state_dict().items():
                if name.startswith("domain_head."):
                    torch.testing.assert_close(tensor.cpu(), pretrained[name], rtol=0, atol=0)
        stages.append(stage)
        return result

    monkeypatch.setattr(adaptation, "train_with_early_stopping", checked_train)
    assert adaptation.main(["--config", str(tmp_path / "config.yaml"), "--device", "cpu"]) == 0
    assert stages == ["pretrain", "fine-tune", "fine-tune", "fine-tune"]
    split_path = next((tmp_path / "results").rglob("pretrain_split.json"))
    split = json.loads(split_path.read_text())
    assert split["excluded_target_subject"] == "S001"
    assert split["domain_subject_mapping"] == {"S002": 0, "S003": 1}
    histories = list((tmp_path / "results").rglob("fine_tune_history.csv"))
    assert len(histories) == 3
    for path in histories:
        with path.open() as handle:
            row = next(csv.DictReader(handle))
        assert float(row["train_domain_loss"]) == 0.0
        assert float(row["train_prototype_intra_loss"]) > 0.0
