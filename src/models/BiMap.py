import torch
from torch import nn


class BiMap(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input
        x: (..., in_dim, in_dim)
        :return: output
        y: (..., out_dim, out_dim)
        """
        w = self.weight

        y = w @ x @ w.transpose(-1, -2)

        # numerical symmetry correction
        y = 0.5 * (y + y.transpose(-1, -2))

        # optional jitter for numerical stability
        eye = torch.eye(
            self.out_dim,
            device=x.device,
            dtype=x.dtype,
        )
        y = y + self.eps * eye

        return y