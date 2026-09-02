from __future__ import annotations

import torch

from src.models.SPDAttention import SingleHeadAttention, spd_exp


def test_metric_rank_is_capped_by_each_layers_tangent_dimension():
    attention = SingleHeadAttention(
        spd_in_dim=2,
        attention_dim=2,
        metric="learnable-metric",
        stage_transition=False,
        learnable_metric_mode="low-rank",
        learnable_metric_score="distance",
        learnable_metric_rank=4,
    )

    assert attention.tangent_feature_dim == 3
    assert attention.requested_learnable_metric_rank == 4
    assert attention.learnable_metric_rank == 3
    assert attention.metric_low_rank.shape == (3, 3)


def test_spd_exp_compresses_extreme_finite_log_spectrum():
    log_eigenvalues = torch.tensor(
        [[1.0e4, -1.0e4, 25.0]],
        requires_grad=True,
    )
    log_spd = torch.diag_embed(log_eigenvalues)

    spd = spd_exp(log_spd, eps=1.0e-6)

    assert torch.isfinite(spd).all()
    assert torch.linalg.eigvalsh(spd).min() > 0
    spd.log1p().mean().backward()
    assert log_eigenvalues.grad is not None
    assert torch.isfinite(log_eigenvalues.grad).all()


def test_spd_exp_rejects_non_finite_log_input():
    log_spd = torch.diag_embed(torch.tensor([[0.0, float("inf")]]))

    try:
        spd_exp(log_spd)
    except RuntimeError as error:
        assert "before matrix_exp" in str(error)
    else:
        raise AssertionError("Non-finite log-SPD input must not be repaired.")


def test_extreme_finite_low_rank_metric_does_not_overflow_attention_score():
    attention = SingleHeadAttention(
        spd_in_dim=4,
        attention_dim=4,
        metric="learnable-metric",
        stage_transition=False,
        learnable_metric_mode="low-rank",
        learnable_metric_score="distance",
        learnable_metric_rank=2,
        eps=1.0e-6,
    )
    with torch.no_grad():
        attention.metric_low_rank.fill_(1.0e20)

    diagonal = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
    )
    x = torch.diag_embed(diagonal).unsqueeze(0)
    output, _aux = attention(x, return_aux=False)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert attention.metric_low_rank.grad is not None
    assert torch.isfinite(attention.metric_low_rank.grad).all()


def test_non_finite_attention_input_is_not_silently_repaired():
    attention = SingleHeadAttention(
        spd_in_dim=3,
        attention_dim=3,
        metric="learnable-metric",
        stage_transition=False,
        learnable_metric_mode="low-rank",
        learnable_metric_score="distance",
        learnable_metric_rank=2,
    )
    score = torch.tensor([[[0.0, float("inf")]]])
    try:
        attention._stabilize_attention_score(score)
    except RuntimeError as error:
        assert "finite=1/2" in str(error)
    else:
        raise AssertionError("A genuinely non-finite score must still fail.")


def test_attention_dropout_renormalizes_each_query_row():
    attention_layer = SingleHeadAttention(
        spd_in_dim=3,
        attention_dim=3,
        metric="log-euclidean",
        stage_transition=False,
        attention_dropout=0.5,
    )
    attention_layer.train()
    attention = torch.softmax(torch.randn(4, 5, 6), dim=-1)

    torch.manual_seed(11)
    regularized = attention_layer._dropout_and_renormalize_attention(attention)

    assert torch.isfinite(regularized).all()
    assert torch.allclose(
        regularized.sum(dim=-1),
        torch.ones_like(regularized[..., 0]),
    )
