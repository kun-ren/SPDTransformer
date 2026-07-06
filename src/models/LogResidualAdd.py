from __future__ import annotations

import torch
from torch import nn


class LogResidualAdd(nn.Module):
    """
    Log-domain residual merge without normalization.

    This module has the same call pattern as TraceAddNorm, but it deliberately
    avoids trace normalization and spectral normalization. Both inputs are
    interpreted as complete symmetric log-domain matrices:

        output = (1 - eta) * residual_log + eta * sublayer_output_log

    where eta is a learned scalar in (0, 1). The convex merge is used because
    current attention/FFN blocks return full log matrices, not pure residual
    deltas.
    """

    def __init__(
            self,
            spd_in_dim: int,
            sequence_length: int | None = None,
            tau: float = 1.0,
            eps: float = 1e-6,
            affine: bool = False,
            position_axis: int = -3,
            sublayer_weight_init: float = 0.1,
    ) -> None:
        super().__init__()
        if spd_in_dim < 1:
            raise ValueError(f"spd_in_dim must be positive, got {spd_in_dim}.")
        if not 0.0 < sublayer_weight_init < 1.0:
            raise ValueError(
                "sublayer_weight_init must be in (0, 1), "
                f"got {sublayer_weight_init}."
            )

        self.spd_in_dim = spd_in_dim
        self.sequence_length = sequence_length
        self.tau = tau
        self.eps = eps
        self.affine = affine
        self.position_axis = position_axis
        self.residual_weight = nn.Parameter(
            torch.logit(torch.tensor(float(sublayer_weight_init)))
        )

    def forward(
            self,
            residual_log: torch.Tensor,
            sublayer_output_log: torch.Tensor,
    ) -> torch.Tensor:
        if residual_log.shape != sublayer_output_log.shape:
            raise ValueError(
                "residual_log and sublayer_output_log must have the same "
                f"shape, got {tuple(residual_log.shape)} and "
                f"{tuple(sublayer_output_log.shape)}."
            )
        if residual_log.shape[-2:] != (self.spd_in_dim, self.spd_in_dim):
            raise ValueError(
                f"Expected last two dimensions ({self.spd_in_dim}, "
                f"{self.spd_in_dim}), got {tuple(residual_log.shape[-2:])}."
            )

        eta = torch.sigmoid(self.residual_weight).to(
            device=residual_log.device,
            dtype=residual_log.dtype,
        )
        output_log = (1.0 - eta) * residual_log + eta * sublayer_output_log
        return 0.5 * (output_log + output_log.transpose(-1, -2))
