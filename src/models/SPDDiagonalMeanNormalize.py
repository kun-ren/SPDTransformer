import math

import torch
from torch import nn


class SPDDiagonalMeanNormalize(nn.Module):
    """
    Normalize each batch sample by the mean diagonal value of all SPD matrices.

    For input shape (batch, ..., channels, channels), all diagonal entries from
    the non-batch SPD matrices are pooled per batch sample. The output is scaled
    so each sample has diagonal sum channels * num_matrices, i.e. average
    diagonal value 1.
    """

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(
                "Expected SPD tensor with shape (batch, ..., channels, channels), "
                f"got {tuple(x.shape)}."
            )

        n_channels = x.shape[-1]
        if x.shape[-2] != n_channels:
            raise ValueError(
                "The last two dimensions must be a square SPD matrix, "
                f"got {tuple(x.shape[-2:])}."
            )

        num_matrices = math.prod(x.shape[1:-2])
        if num_matrices < 1:
            raise ValueError(
                "Expected at least one SPD matrix per batch sample, "
                f"got shape {tuple(x.shape)}."
            )

        diagonals = torch.diagonal(x, dim1=-2, dim2=-1)
        diagonal_sum = diagonals.sum(dim=tuple(range(1, diagonals.ndim)))
        diagonal_sum = diagonal_sum.clamp_min(self.eps)

        target_sum = float(n_channels * num_matrices)
        scale = target_sum / diagonal_sum
        view_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        return x * scale.reshape(view_shape)
