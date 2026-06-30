import torch
from torch import nn

from src.models.SPDAttention import spd_log


class TraceAddNorm(nn.Module):
    """
    Trace Add & Norm block using a Log-Euclidean residual connection.

    The residual merge is a two-point Log-Euclidean barycenter:
        merged = exp((1 - alpha) log(residual) + alpha log(sublayer_output))
    followed by Riemannian layer normalization.
    """

    def __init__(
            self,
            spd_in_dim: int,
            sequence_length: int,
            tau: float = 1.0,
            eps: float = 1e-5,
            affine: bool = True,
    ):
        super().__init__()

        self.spd_in_dim = spd_in_dim
        self.residual_weight = nn.Parameter(
            torch.logit(torch.tensor(0.1))
        )


    def forward(self, residual_log: torch.Tensor, sublayer_output_log: torch.Tensor) -> torch.Tensor:

        #Constrain the residual scale to (0, 1).
        eta = torch.sigmoid(self.residual_weight)

        S_res = residual_log + eta * sublayer_output_log


        # Protect against small floating-point asymmetry.
        S_res = 0.5 * (
                S_res + S_res.transpose(-1, -2)
        )

        output = torch.matrix_exp(S_res)

        output = 0.5 * (
                output + output.transpose(-1, -2)
        )

        C = output.shape[-1]
        I = torch.eye(C, device=output.device, dtype=output.dtype)

        output = output + self.eps * I

        trace = torch.diagonal(output, dim1=-2, dim2=-1).sum(dim=-1)
        trace = trace.clamp_min(self.eps)

        output = output / trace[..., None, None]

        return spd_log(output)