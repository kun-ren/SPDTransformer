import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal


class BiMap(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim > in_dim:
            raise ValueError(
                "BiMap with row-orthogonal weight requires "
                f"out_dim <= in_dim, got out_dim={out_dim} and in_dim={in_dim}."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.reset_parameters()
        orthogonal(
            self,
            name="weight",
            use_trivialization=False,
        )

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

        return y
