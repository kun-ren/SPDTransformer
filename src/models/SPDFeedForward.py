from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
    spd_dim = x.shape[-1]
    row, col = torch.triu_indices(spd_dim, spd_dim, device=x.device)
    return x[..., row, col]


def _upper_triangular_unvectorize(
        vector: torch.Tensor,
        spd_dim: int,
) -> torch.Tensor:
    expected_dim = spd_dim * (spd_dim + 1) // 2
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected vector feature dimension {expected_dim}, "
            f"got {vector.shape[-1]}."
        )

    row, col = torch.triu_indices(spd_dim, spd_dim, device=vector.device)
    matrix = vector.new_zeros(*vector.shape[:-1], spd_dim, spd_dim)
    matrix[..., row, col] = vector
    matrix[..., col, row] = vector
    return _sym(matrix)


def _apply_activation(
        x: torch.Tensor,
        activation: Literal["relu", "gelu", "softplus"],
) -> torch.Tensor:
    if activation == "relu":
        return F.relu(x)
    if activation == "gelu":
        return F.gelu(x)
    if activation == "softplus":
        return F.softplus(x)
    raise ValueError(
        f"activation must be 'relu', 'gelu', or 'softplus', got {activation!r}."
    )


class SPDFeedForward(nn.Module):
    """
    Log-domain vector feed-forward block for SPD Transformer.

    The encoder already keeps the block input in the tangent/log domain, so this
    module implements the stable part of the requested pipeline:

        log(SPD) -> upper triangular vector
            -> Linear -> GELU -> Dropout -> Linear
            -> symmetric matrix -> residual in log domain

    The returned matrix is symmetric log-domain output. TraceAddNorm consumes
    this log-domain output directly and keeps the block in the tangent domain.
    """

    def __init__(
            self,
            spd_dim: int,
            hidden_spd_dim: int | None = None,
            activation: Literal["relu", "gelu", "softplus"] = "gelu",
            dropout: float = 0.0,
            eps: float = 1e-4,
            debug_tensor_stats: bool = False,
    ) -> None:
        super().__init__()

        if activation not in {"relu", "gelu", "softplus"}:
            raise ValueError(
                f"activation must be 'relu', 'gelu', or 'softplus', got {activation!r}"
            )

        self.spd_dim = spd_dim
        self.feature_dim = spd_dim * (spd_dim + 1) // 2
        self.hidden_feature_dim = int(hidden_spd_dim or self.feature_dim * 2)
        if self.hidden_feature_dim < 1:
            raise ValueError(
                f"hidden_spd_dim must be positive, got {hidden_spd_dim!r}."
            )

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.eps = eps
        self.debug_tensor_stats = debug_tensor_stats

        self.linear_in = nn.Linear(self.feature_dim, self.hidden_feature_dim)
        self.linear_out = nn.Linear(self.hidden_feature_dim, self.feature_dim)

        # Start close to identity: the residual branch exists, but initially
        # contributes almost nothing until the final projection learns.
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)
        self.raw_gate = nn.Parameter(torch.logit(torch.tensor(0.1)))

    def forward(self, x_log: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_log: (..., spd_dim, spd_dim), symmetric log-domain matrix.

        Returns:
            (..., spd_dim, spd_dim), symmetric log-domain residual output.
        """
        if x_log.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected x_log shape (..., {self.spd_dim}, {self.spd_dim}), "
                f"got {tuple(x_log.shape)}."
            )

        #x_log = _sym(x_log)
        vector = _upper_triangular_vectorize(x_log)

        hidden = self.linear_in(vector)
        hidden = _apply_activation(hidden, self.activation)
        hidden = self.dropout(hidden)
        delta_vector = self.linear_out(hidden)

        delta_log = _upper_triangular_unvectorize(delta_vector, self.spd_dim)
        gate = torch.sigmoid(self.raw_gate)
        out_log = x_log + gate * delta_log

        return out_log
