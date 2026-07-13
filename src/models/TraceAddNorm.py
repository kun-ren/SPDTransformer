import torch
from torch import nn
import torch.nn.functional as F


class TraceAddNorm(nn.Module):
    """
    Trace Add & Norm block using a Log-Euclidean residual connection.

    Both inputs are symmetric log-domain representations in the same channel
    basis. Different feature matrices are expected; only their shapes and
    channel semantics must agree. The residual merge is

        S = (1 - alpha) * residual_log + alpha * sublayer_output_log

    followed by a direct log-domain trace shift:

        S_norm = S + ((C - tr(S)) / C) * I

    where C is the SPD matrix dimension. This gives tr(S_norm) = C without
    dividing by tr(S), which can be close to zero. It also avoids matrix
    exponential and logarithm operations in the residual path.

    When affine=True, a learnable per-position scale is applied in the log
    domain after trace normalization:

        S'_l = gamma_l * S_l + beta_l * I

    This preserves SPD structure because S'_l remains symmetric, so
    exp(S'_l) is SPD. position_axis is interpreted as a full tensor dimension;
    the default position_axis=-3 means the dimension immediately before the two
    SPD matrix dimensions. For 5D inputs, all SPD matrices at the same
    frequency-band position share one affine parameter pair across batch and
    time.
    """

    def __init__(
            self,
            spd_in_dim: int,
            sequence_length: int,
            tau: float = 1.0,
            eps: float = 1e-5,
            affine: bool = True,
            position_axis: int = -3,
    ):
        super().__init__()
        self.eps = eps
        self.spd_in_dim = spd_in_dim
        self.sequence_length = sequence_length
        self.affine = affine
        self.position_axis = position_axis
        self.residual_weight = nn.Parameter(
            torch.logit(torch.tensor(0.1))
        )

        if affine:
            inverse_softplus_one = torch.log(torch.expm1(torch.tensor(1.0)))
            self.affine_weight_raw = nn.Parameter(
                torch.full((sequence_length,), inverse_softplus_one.item())
            )
            self.affine_bias = nn.Parameter(torch.zeros(sequence_length))
        else:
            self.register_parameter("affine_weight_raw", None)
            self.register_parameter("affine_bias", None)

    def forward(
            self,
            residual_log: torch.Tensor,
            sublayer_output_log: torch.Tensor,
    ) -> torch.Tensor:
        # Constrain the sublayer contribution to (0, 1).
        eta = torch.sigmoid(self.residual_weight)
        S_res = (1.0 - eta) * residual_log + eta * sublayer_output_log
        S_res = 0.5 * (
                S_res + S_res.transpose(-1, -2)
        )

        n_channels = S_res.shape[-1]
        traces = torch.diagonal(S_res, dim1=-2, dim2=-1).sum(dim=-1)
        trace_shift = (
            float(n_channels) - traces
        ) / float(n_channels)
        identity = torch.eye(
            n_channels,
            device=S_res.device,
            dtype=S_res.dtype,
        )
        output_log = S_res + trace_shift[..., None, None] * identity

        output_log = self._apply_log_domain_affine(output_log)
        return 0.5 * (output_log + output_log.transpose(-1, -2))

    def _apply_log_domain_affine(self, output_log: torch.Tensor) -> torch.Tensor:
        if not self.affine:
            return output_log

        position_axis = self._canonical_position_axis(output_log)
        position_length = output_log.shape[position_axis]
        if position_length > self.sequence_length:
            raise ValueError(
                f"Input position length {position_length} exceeds configured "
                f"sequence_length {self.sequence_length}."
            )

        gamma = F.softplus(
            self.affine_weight_raw[:position_length]
        )
        beta = self.affine_bias[:position_length]

        # For [B, T, F, R, C, C], axis-specific parameter views are:
        # time [1, T, 1, 1, 1, 1], frequency [1, 1, F, 1, 1, 1],
        # and brain region [1, 1, 1, R, 1, 1].
        parameter_shape = [1] * output_log.ndim
        parameter_shape[position_axis] = position_length
        gamma = gamma.reshape(parameter_shape)
        beta = beta.reshape(parameter_shape)

        identity = torch.eye(
            output_log.shape[-1],
            device=output_log.device,
            dtype=output_log.dtype,
        )
        return gamma * output_log + beta * identity

    def _canonical_position_axis(self, output_log: torch.Tensor) -> int:
        if output_log.ndim < 3:
            raise ValueError(
                "output_log must have at least one leading position dimension "
                "followed by two SPD matrix dimensions, "
                f"got shape {tuple(output_log.shape)}."
            )

        if output_log.shape[-2:] != (self.spd_in_dim, self.spd_in_dim):
            raise ValueError(
                f"Expected trailing SPD shape ({self.spd_in_dim}, "
                f"{self.spd_in_dim}), got {tuple(output_log.shape[-2:])}."
            )

        position_axis = self.position_axis
        if position_axis < 0:
            position_axis += output_log.ndim
        if not 0 <= position_axis < output_log.ndim - 2:
            raise ValueError(
                "position_axis must refer to a dimension before the two SPD "
                f"matrix dimensions, got position_axis={self.position_axis} "
                f"for shape {tuple(output_log.shape)}."
            )
        return position_axis
