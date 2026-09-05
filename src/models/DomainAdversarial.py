from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            inputs: torch.Tensor,
            coefficient: float,
    ) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(
        inputs: torch.Tensor,
        coefficient: float,
) -> torch.Tensor:
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(
            "gradient-reversal coefficient must be finite and non-negative, "
            f"got {coefficient}."
        )
    return _GradientReversal.apply(inputs, coefficient)


class DomainAdversarialHead(nn.Module):
    """Predict the source subject from a reversed pooled log-SPD feature."""

    def __init__(
            self,
            spd_dim: int,
            num_domains: int,
            hidden_dim: int = 32,
            dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if spd_dim < 1:
            raise ValueError(f"spd_dim must be positive, got {spd_dim}.")
        if num_domains < 2:
            raise ValueError(
                "Domain adversarial training requires at least two source "
                f"domains, got {num_domains}."
            )
        if hidden_dim < 1:
            raise ValueError(
                f"domain hidden_dim must be positive, got {hidden_dim}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"domain dropout must be in [0, 1), got {dropout}.")

        self.spd_dim = int(spd_dim)
        self.num_domains = int(num_domains)
        feature_dim = self.spd_dim * (self.spd_dim + 1) // 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_domains),
        )

        row, col = torch.triu_indices(self.spd_dim, self.spd_dim)
        self.register_buffer("triu_row", row, persistent=False)
        self.register_buffer("triu_col", col, persistent=False)
        scale = torch.ones(row.numel())
        scale[row != col] = math.sqrt(2.0)
        self.register_buffer("triu_scale", scale, persistent=False)

    def vectorize(self, pooled_log: torch.Tensor) -> torch.Tensor:
        if pooled_log.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                "Expected pooled log-SPD shape "
                f"(..., {self.spd_dim}, {self.spd_dim}), got "
                f"{tuple(pooled_log.shape)}."
            )
        symmetric = 0.5 * (pooled_log + pooled_log.transpose(-1, -2))
        features = symmetric[..., self.triu_row, self.triu_col]
        return features * self.triu_scale.to(
            device=features.device,
            dtype=features.dtype,
        )

    def forward(
            self,
            pooled_log: torch.Tensor,
            reversal_coefficient: float,
    ) -> torch.Tensor:
        features = self.vectorize(pooled_log)
        reversed_features = gradient_reverse(features, reversal_coefficient)
        return self.classifier(reversed_features)
