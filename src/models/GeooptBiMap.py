import torch
import torch.nn as nn
import geoopt


class GeooptBiMap(nn.Module):
    """
    SPDNet-style BiMap layer with Stiefel manifold constraint.

    P_out = W^T P W

    Input:
        P: (..., in_dim, in_dim)

    Output:
        P_out: (..., out_dim, out_dim)

    Constraint:
        W^T W = I
    """

    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-9):
        super().__init__()

        if out_dim > in_dim:
            raise ValueError("For Stiefel BiMap, out_dim must be <= in_dim.")

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.manifold = geoopt.Stiefel()

        # Initialize W on Stiefel manifold
        W = torch.eye(in_dim, out_dim)

        self.weight = geoopt.ManifoldParameter(
            W,
            manifold=self.stiefel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        P: (..., in_dim, in_dim)
        """
        W = self.weight
        X_out = W.transpose(-1, -2) @ x @ W
        X_out = 0.5 * (X_out + X_out.transpose(-1, -2))
        return X_out