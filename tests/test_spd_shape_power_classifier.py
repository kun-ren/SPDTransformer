from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.models.SPDShapePowerClassifier import SPDShapePowerClassifier
from src.models.SPDTransformerClassifier import SPDTransformerClassifier
from src.training.train import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_spd(*shape: int, dimension: int) -> torch.Tensor:
    factor = torch.randn(*shape, dimension, dimension)
    identity = torch.eye(dimension)
    return factor @ factor.transpose(-1, -2) + 0.5 * identity


def make_classifier(
        *,
        fusion_classifier: str = "cosine",
        power_center_log: bool = True,
) -> SPDShapePowerClassifier:
    return SPDShapePowerClassifier(
        num_heads=2,
        spd_in_dim=3,
        attention_dim=[3],
        num_classes=2,
        stage_transition=False,
        time_sequence_length=2,
        frequency_sequence_length=2,
        brain_region_sequence_length=2,
        depth=1,
        metric="learnable-metric",
        learnable_metric_mode="low-rank",
        learnable_metric_score="distance",
        learnable_metric_rank=2,
        use_position_bias=False,
        layer_norm_affine=False,
        add_norm_type="sequence_add_norm",
        dropout=0.0,
        attention_dropout=0.0,
        power_hidden_dim=4,
        power_feature_dim=5,
        power_kernel_size=3,
        power_dropout=0.0,
        power_center_log=power_center_log,
        fusion_classifier=fusion_classifier,
        fusion_dropout=0.0,
    )


def test_covariance_decomposition_is_trace_normalized_and_reconstructable():
    model = make_classifier(power_center_log=False)
    base = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    scales = torch.tensor([2.0, 5.0]).reshape(1, 2, 1, 1, 1, 1)
    x = scales * base.reshape(1, 1, 1, 1, 3, 3)

    shape, log_power = model.decompose_covariance(x)

    assert torch.allclose(
        torch.diagonal(shape, dim1=-2, dim2=-1).sum(dim=-1),
        torch.full_like(log_power, 3.0),
    )
    reconstructed = shape * log_power.exp()[..., None, None]
    assert torch.allclose(reconstructed, x)


@pytest.mark.parametrize("fusion_classifier", ["cosine", "linear"])
def test_shape_power_forward_and_backward(fusion_classifier: str):
    model = make_classifier(fusion_classifier=fusion_classifier)
    x = make_spd(2, 2, 2, 2, dimension=3)

    logits, aux = model(x, return_aux=True)
    logits.square().mean().backward()

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert aux
    metric_gradient = (
        model.shape_encoder.layers[0].shared_metric.metric_low_rank.grad
    )
    power_gradient = model.power_encoder.network[0].weight.grad
    assert metric_gradient is not None and torch.isfinite(metric_gradient).all()
    assert power_gradient is not None and torch.isfinite(power_gradient).all()


def test_build_model_selects_shape_power_classifier():
    model = build_model(
        {
            "head_nums": 1,
            "attention_dim": "3",
            "depth": 1,
            "stage_transition": False,
            "classifier_type": "shape_power",
            "power_hidden_dim": 4,
            "power_feature_dim": 5,
            "power_kernel_size": 3,
            "power_dropout": 0.1,
            "power_center_log": True,
            "fusion_classifier": "linear",
            "fusion_dropout": 0.1,
        },
        spd_in_dim=3,
        num_classes=2,
        time_sequence_length=2,
        frequency_sequence_length=2,
        brain_region_sequence_length=2,
    )

    assert isinstance(model, SPDTransformerClassifier)
    assert isinstance(model.model, SPDShapePowerClassifier)
    assert model.model.fusion_classifier == "linear"


def test_physionet_configs_enable_shape_power_fusion():
    config_names = (
        "train_grid.yaml",
        "train_physionet_global_cv_hparam.yaml",
        "train_physionet_pretrain_finetune_loro.yaml",
    )
    for config_name in config_names:
        with (PROJECT_ROOT / "configs" / config_name).open(
            "r", encoding="utf-8"
        ) as handle:
            model = yaml.safe_load(handle)["model"]
        assert model["classifier_type"] == ["shape_power"]
        assert model["pooling"] == ["mean"]
        assert model["power_center_log"] == [True]
        assert model["fusion_classifier"] == ["cosine"]
