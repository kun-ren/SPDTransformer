import torch
from torch import nn
import torch.nn.functional as F


class TraceAddNorm(nn.Module):
    """
    Trace Add & Norm block using a Log-Euclidean residual connection.

    The residual merge is a two-point Log-Euclidean barycenter:
        merged = exp((1 - alpha) log(residual) + alpha log(sublayer_output))
    followed by trace normalization. The trace normalization is computed
    directly in the log domain:

        log(exp(S) / tr(exp(S))) = S - log(tr(exp(S))) I

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

    def forward(self, residual_log: torch.Tensor, sublayer_output_log: torch.Tensor) -> torch.Tensor:


        #Constrain the residual scale to (0, 1).
        eta = torch.sigmoid(self.residual_weight)

        S_res = (1.0 - eta) * residual_log + eta * sublayer_output_log



        S_res = 0.5 * (
                S_res + S_res.transpose(-1, -2)
        )

        C = S_res.shape[-1]
        I = torch.eye(C, device=S_res.device, dtype=S_res.dtype)
        log_trace = self._log_trace_exp(S_res)
        output_log = S_res - log_trace[..., None, None] * I
        output_log = self._apply_log_domain_affine(output_log)
        return 0.5 * (output_log + output_log.transpose(-1, -2))

    def _log_trace_exp(self, symmetric_log: torch.Tensor) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(symmetric_log)
        return torch.logsumexp(eigenvalues, dim=-1)

    def _apply_log_domain_affine(self, output_log: torch.Tensor) -> torch.Tensor:
        if not self.affine:
            return output_log

        position_axis = self.position_axis
        if position_axis < 0:
            position_axis += output_log.ndim
        if not 0 <= position_axis < output_log.ndim - 2:
            raise ValueError(
                "position_axis must refer to a non-SPD tensor dimension, "
                f"got position_axis={self.position_axis} for shape "
                f"{tuple(output_log.shape)}."
            )

        current_length = output_log.shape[position_axis]
        if current_length > self.sequence_length:
            raise ValueError(
                f"Input position length {current_length} exceeds configured "
                f"sequence_length {self.sequence_length}."
            )

        view_shape = [1] * output_log.ndim
        view_shape[position_axis] = current_length

        gamma = F.softplus(self.affine_weight_raw[:current_length]).view(view_shape)
        beta = self.affine_bias[:current_length].view(view_shape)

        eye = torch.eye(
            output_log.shape[-1],
            device=output_log.device,
            dtype=output_log.dtype,
        )
        return gamma * output_log + beta * eye
