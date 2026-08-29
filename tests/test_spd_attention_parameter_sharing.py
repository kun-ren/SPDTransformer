from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.models.MultiHeadEncoder import SPDMultiHeadEncoder
from src.models.SPDAttention import SingleHeadAttention


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_spd(*shape: int, dimension: int) -> torch.Tensor:
    factor = torch.randn(*shape, dimension, dimension)
    identity = torch.eye(dimension)
    return factor @ factor.transpose(-1, -2) + 0.5 * identity


def test_single_head_uses_one_projection_for_query_and_key():
    attention = SingleHeadAttention(
        spd_in_dim=3,
        attention_dim=3,
        metric="log-euclidean",
        stage_transition=False,
    )
    x = make_spd(1, 2, dimension=3)

    _output, aux = attention(x, return_aux=True)

    assert attention.key is attention.query
    assert torch.equal(aux["P_q"], aux["P_k"])
    assert not any(name.startswith("key.") for name in attention.state_dict())


def test_multi_head_layer_registers_one_metric_for_every_axis_and_head():
    encoder = SPDMultiHeadEncoder(
        spd_in_dim=3,
        attention_dim=3,
        time_sequence_length=2,
        frequency_sequence_length=2,
        brain_region_sequence_length=2,
        num_heads=4,
        stage_transition=False,
        metric="learnable-metric",
        learnable_metric_mode="low-rank",
        learnable_metric_score="distance",
        learnable_metric_rank=2,
        use_position_bias=False,
        layer_norm_affine=False,
    )
    all_heads = [
        *encoder.time_attention,
        *encoder.frequency_attention,
        *encoder.region_attention,
    ]

    metric_parameter_names = [
        name for name, _parameter in encoder.named_parameters()
        if "metric_low_rank" in name
    ]
    assert metric_parameter_names == ["shared_metric.metric_low_rank"]
    assert all(head.metric_parameters is None for head in all_heads)
    assert all(head.key is head.query for head in all_heads)

    x = make_spd(1, 2, 2, 2, dimension=3)
    output, _aux = encoder(x, return_log=True, return_aux=False)
    output.square().mean().backward()

    gradient = encoder.shared_metric.metric_low_rank.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_physionet_configs_use_shallow_mean_pooling_regularization():
    with (PROJECT_ROOT / "configs" / "train_physionet_global_cv_hparam.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        global_model = yaml.safe_load(handle)["model"]
    with (
        PROJECT_ROOT / "configs" / "train_physionet_pretrain_finetune_loro.yaml"
    ).open("r", encoding="utf-8") as handle:
        transfer_model = yaml.safe_load(handle)["model"]

    assert global_model["depth"] == [1]
    assert global_model["head_nums"] == [4]
    assert global_model["attention_dim"] == ["8"]
    assert global_model["stage_transition"] == [False]
    assert global_model["pooling"] == ["mean"]
    assert global_model["dropout"] == [0.1, 0.2, 0.3]
    assert global_model["attention_dropout"] == [0.1, 0.2]

    assert transfer_model["depth"] == [1]
    assert transfer_model["head_nums"] == [4]
    assert transfer_model["attention_dim"] == ["8"]
    assert transfer_model["stage_transition"] == [False]
    assert transfer_model["pooling"] == ["mean"]
    assert transfer_model["dropout"] == [0.2]
    assert transfer_model["attention_dropout"] == [0.1]
