import torch
from torch import nn

from src.models.RiemannianLayerNorm import RiemannianLayerNorm


class SPDAddNorm(nn.Module):
    """
    SPD Add & Norm block using a Log-Euclidean residual connection.

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
            torch.tensor(-2.0)
        )

        self.norm = RiemannianLayerNorm(
            spd_dim=spd_in_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            preserve_log_mean=False,
        )

    def forward(self, residual_log: torch.Tensor, sublayer_output_log: torch.Tensor) -> torch.Tensor:

        #Constrain the residual scale to (0, 1).
        eta = torch.sigmoid(self.residual_weight)

        S_res = (
                residual_log
                + eta * sublayer_output_log
        )

        # Protect against small floating-point asymmetry.
        S_res = 0.5 * (
                S_res + S_res.transpose(-1, -2)
        )

        output_log = self.norm(S_res)
        return output_log