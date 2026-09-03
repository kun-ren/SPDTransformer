from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch


PoolingWeightMode = Literal["full", "factorized"]


def validate_probability(value: float, name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}.")
    return probability


def validate_unit_interval(value: float, name: str) -> float:
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return fraction


def normalize_pooling_weight_mode(value: str) -> PoolingWeightMode:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"full", "dense", "joint"}:
        return "full"
    if normalized in {"factorized", "factorised", "axis", "axis_factorized"}:
        return "factorized"
    raise ValueError(
        "pooling_weight_mode must be 'full' or 'factorized', "
        f"got {value!r}."
    )


def factorized_logits(
        axis_logits: Sequence[torch.Tensor],
        token_shape: tuple[int, ...],
) -> torch.Tensor:
    """Build additive time/frequency/region logits without a full token grid."""
    if len(axis_logits) < len(token_shape):
        raise ValueError(
            f"Need {len(token_shape)} axis logits, got {len(axis_logits)}."
        )

    logits = None
    for axis, size in enumerate(token_shape):
        values = axis_logits[axis]
        if size > values.numel():
            raise ValueError(
                f"Token axis {axis} has size {size}, but pooling was initialized "
                f"for at most {values.numel()}."
            )
        view_shape = [1] * len(token_shape)
        view_shape[axis] = size
        contribution = values[:size].reshape(view_shape)
        logits = contribution if logits is None else logits + contribution

    if logits is None:
        raise ValueError("factorized_logits requires at least one token axis.")
    return logits


def regularize_simplex_weights(
        weights: torch.Tensor,
        *,
        sample_count: int,
        dropout: float,
        uniform_mix: float = 0.0,
        training: bool,
) -> torch.Tensor:
    """Apply per-sample dropout to simplex weights and renormalize survivors."""
    if weights.ndim != 1 or weights.numel() < 1:
        raise ValueError(
            f"weights must be a non-empty vector, got shape {tuple(weights.shape)}."
        )
    if sample_count < 1:
        raise ValueError(f"sample_count must be positive, got {sample_count}.")

    dropout = validate_probability(dropout, "dropout")
    uniform_mix = validate_unit_interval(uniform_mix, "uniform_mix")
    normalized = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    if uniform_mix:
        normalized = (
            (1.0 - uniform_mix) * normalized
            + uniform_mix / float(normalized.numel())
        )

    if not training or dropout == 0.0 or normalized.numel() == 1:
        return normalized.unsqueeze(0)

    expanded = normalized.unsqueeze(0).expand(sample_count, -1)
    keep_mask = torch.rand_like(expanded).ge(dropout).to(dtype=expanded.dtype)
    dropped = expanded * keep_mask
    totals = dropped.sum(dim=-1, keepdim=True)
    renormalized = dropped / totals.clamp_min(torch.finfo(weights.dtype).tiny)
    return torch.where(totals > 0, renormalized, expanded)
