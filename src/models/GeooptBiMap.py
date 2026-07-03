import torch
import torch.nn as nn
import geoopt
from typing import Literal, Sequence


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

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            eps: float = 1e-9,
            init: Literal["identity", "random"] = "identity",
            identity_indices: Sequence[int] | torch.Tensor | None = None,
    ):
        super().__init__()
        if out_dim > in_dim:
            raise ValueError("For Stiefel BiMap, out_dim must be <= in_dim.")
        if init not in {"identity", "random"}:
            raise ValueError(
                f"GeooptBiMap init must be 'identity' or 'random', got {init!r}."
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps
        self.identity_indices = None

        self.manifold = geoopt.Stiefel()

        # Initialize W on the Stiefel manifold.
        if init == "identity":
            if identity_indices is None:
                W = torch.eye(in_dim, out_dim)
                self.identity_indices = tuple(range(out_dim))
            else:
                indices = torch.as_tensor(identity_indices, dtype=torch.long).flatten()
                if indices.numel() != out_dim:
                    raise ValueError(
                        "identity_indices length must match out_dim, "
                        f"got {indices.numel()} and out_dim={out_dim}."
                    )
                if indices.unique().numel() != out_dim:
                    raise ValueError("identity_indices must not contain duplicates.")
                if int(indices.min()) < 0 or int(indices.max()) >= in_dim:
                    raise ValueError(
                        "identity_indices must be in [0, in_dim), "
                        f"got min={int(indices.min())}, max={int(indices.max())}, "
                        f"in_dim={in_dim}."
                    )

                W = torch.zeros(in_dim, out_dim)
                W[indices, torch.arange(out_dim)] = 1.0
                self.identity_indices = tuple(int(index) for index in indices.tolist())
        else:
            W, _ = torch.linalg.qr(
                torch.randn(in_dim, out_dim),
                mode="reduced",
            )

        self.weight = geoopt.ManifoldParameter(
            W,
            manifold=self.manifold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        P: (..., in_dim, in_dim)
        """
        W = self.weight
        X_out = W.transpose(-1, -2) @ x @ W

        eye = torch.eye(
            self.out_dim,
            device=X_out.device,
            dtype=X_out.dtype,
        )
        X_out = X_out + self.eps * eye
        X_out = 0.5 * (X_out + X_out.transpose(-1, -2))
        
        return X_out
