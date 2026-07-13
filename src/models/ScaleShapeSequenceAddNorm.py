import torch
from torch import nn
import torch.nn.functional as F


class ScaleShapeSequenceAddNorm(nn.Module):
    """
    Sequence Add & Norm that separates log-volume from covariance shape.

    For each merged symmetric log-domain matrix S_l, decompose

        m_l = tr(S_l) / C
        A_l = S_l - m_l * I

    where A_l is traceless. Along ``position_axis``, normalize only the shared
    RMS scale of A_l and center m_l with one group mean:

        r = sqrt(mean_{l,i,j}(A_l[i,j] ** 2) + eps)
        S_norm_l = A_l / r + (m_l - mean_l(m_l)) * I

    The same r and mean are used for every position in a group. Consequently,
    pairwise log-volume differences are preserved exactly, while the traceless
    covariance-shape component has a controlled numerical scale. Dimensions
    other than ``position_axis`` define independent groups.
    """

    def __init__(
            self,
            spd_in_dim: int,
            sequence_length: int,
            tau: float = 1.0,
            eps: float = 1e-5,
            affine: bool = True,
            position_axis: int = -3,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

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
        if residual_log.shape != sublayer_output_log.shape:
            raise ValueError(
                "residual_log and sublayer_output_log must have the same "
                f"shape, got {tuple(residual_log.shape)} and "
                f"{tuple(sublayer_output_log.shape)}."
            )

        position_axis = self._canonical_position_axis(residual_log)
        position_length = residual_log.shape[position_axis]
        if position_length > self.sequence_length:
            raise ValueError(
                f"Input position length {position_length} exceeds configured "
                f"sequence_length {self.sequence_length}."
            )

        eta = torch.sigmoid(self.residual_weight)
        output_log = (
            (1.0 - eta) * residual_log
            + eta * sublayer_output_log
        )
        output_log = 0.5 * (
            output_log + output_log.transpose(-1, -2)
        )

        identity = torch.eye(
            self.spd_in_dim,
            device=output_log.device,
            dtype=output_log.dtype,
        )
        log_volume = torch.diagonal(
            output_log, dim1=-2, dim2=-1
        ).sum(dim=-1) / float(self.spd_in_dim)
        traceless = output_log - log_volume[..., None, None] * identity

        shape_rms = traceless.square().mean(
            dim=(position_axis, output_log.ndim - 2, output_log.ndim - 1),
            keepdim=True,
        ).add(self.eps).sqrt()
        centered_log_volume = log_volume - log_volume.mean(
            dim=position_axis,
            keepdim=True,
        )

        output_log = (
            traceless / shape_rms
            + centered_log_volume[..., None, None] * identity
        )
        output_log = self._apply_log_domain_affine(output_log)
        return 0.5 * (output_log + output_log.transpose(-1, -2))

    def _apply_log_domain_affine(
            self,
            output_log: torch.Tensor,
    ) -> torch.Tensor:
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
