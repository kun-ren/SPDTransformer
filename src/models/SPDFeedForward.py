from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from src.models.SPDAttention import spd_log


def _sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _safe_eigh(
        x: torch.Tensor,
        eps: float = 1e-4,
):
    """
    Stable eigen-decomposition for symmetric SPD-like matrices.
    """
    x = _sym(x)

    eigvals, eigvecs = torch.linalg.eigh(x)
    eigvals = torch.clamp(eigvals, min=eps)

    return eigvals, eigvecs


class SPDActivation(nn.Module):
    """
    SPD-safe activation applied in the eigenvalue domain.

    Input:
        x: (..., C, C), SPD matrix

    Output:
        y: (..., C, C), SPD matrix
    """

    def __init__(
            self,
            activation: Literal["relu", "gelu", "softplus"] = "gelu",
            eps: float = 1e-4,
    ) -> None:
        super().__init__()

        if activation not in {"relu", "gelu", "softplus"}:
            raise ValueError(
                f"activation must be 'relu', 'gelu', or 'softplus', got {activation!r}"
            )

        self.activation = activation
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sym(x)

        eigvals, eigvecs = _safe_eigh(x, eps=self.eps)

        if self.activation == "relu":
            eigvals = eigvals.clamp_min(self.eps)

        elif self.activation == "gelu":
            eigvals = F.gelu(eigvals).clamp_min(self.eps)

        elif self.activation == "softplus":
            eigvals = F.softplus(eigvals).clamp_min(self.eps)

        y = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
        y = _sym(y)

        return y


class GeooptSquareBiMap(nn.Module):
    """
    Same-dimension BiMap:

        Y = W^T X W

    where:
        W in St(C, C)

    If W is initialized as identity, then initially:

        Y = X
    """

    def __init__(
            self,
            spd_dim: int,
            init: Literal["identity", "random"] = "identity",
    ) -> None:
        super().__init__()

        self.spd_dim = spd_dim
        self.stiefel = geoopt.Stiefel()

        if init == "identity":
            W = torch.eye(spd_dim)

        elif init == "random":
            W = torch.randn(spd_dim, spd_dim)
            W = self.stiefel.projx(W)

        else:
            raise ValueError(f"Unknown init: {init!r}")

        self.W = geoopt.ManifoldParameter(
            W,
            manifold=self.stiefel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., C, C), SPD matrix
        """
        if x.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected x shape (..., {self.spd_dim}, {self.spd_dim}), "
                f"got {tuple(x.shape)}."
            )

        x = _sym(x)

        W = self.W
        y = W.transpose(-1, -2) @ x @ W
        y = _sym(y)

        return y


class SPDFeedForward(nn.Module):
    """
    SPD-valued feed-forward block for SPD Transformer.

    This version respects the SPD manifold by avoiding ordinary Euclidean
    vector FFN operations on vectorized matrices.

    Pipeline:
        X
        -> BiMap: W1^T X W1
        -> SPDActivation in eigenvalue domain
        -> Dropout-like SPD noise is not used directly
        -> BiMap: W2^T X W2
        -> SPDActivation in eigenvalue domain
        -> SPD output

    Input:
        x: (..., spd_dim, spd_dim), SPD matrix

    Output:
        out: (..., spd_dim, spd_dim), SPD matrix
    """

    def __init__(
            self,
            spd_dim: int,
            hidden_spd_dim: int | None = None,
            activation: Literal["relu", "gelu", "softplus"] = "gelu",
            dropout: float = 0.0,
            eps: float = 1e-4,
            debug_tensor_stats: bool = False,
    ) -> None:
        super().__init__()

        if hidden_spd_dim is not None and hidden_spd_dim != spd_dim:
            raise ValueError(
                "Strict SPDFeedForward keeps the SPD matrix dimension unchanged. "
                "Use hidden_spd_dim=None or hidden_spd_dim=spd_dim. "
                "A pure BiMap cannot safely do C -> hidden -> C when hidden != C."
            )

        self.spd_dim = spd_dim
        self.eps = eps
        self.dropout = dropout
        self.debug_tensor_stats = debug_tensor_stats

        self.bimap1 = GeooptSquareBiMap(
            spd_dim=spd_dim,
            init="identity",
        )

        self.act1 = SPDActivation(
            activation=activation,
            eps=eps,
        )

        self.bimap2 = GeooptSquareBiMap(
            spd_dim=spd_dim,
            init="identity",
        )

        self.act2 = SPDActivation(
            activation=activation,
            eps=eps,
        )

        # Learnable residual scale inside FFN.
        # Initialized close to zero, so the FFN starts near identity if used
        # inside Log-Euclidean residual block.
        self.raw_gate = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., spd_dim, spd_dim), SPD matrix

        return:
            out: (..., spd_dim, spd_dim), SPD matrix
        """
        if x.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected x shape (..., {self.spd_dim}, {self.spd_dim}), "
                f"got {tuple(x.shape)}."
            )

        x = _sym(x)

        x = torch.matrix_exp(x)

        y = self.bimap1(x)
        y = self.act1(y)

        y = self.bimap2(y)
        y = self.act2(y)

        # Optional SPD-safe residual interpolation in Log-Euclidean form.
        # This keeps the output SPD and starts close to identity.
        gate = torch.sigmoid(self.raw_gate)

        x_log = spd_log(x, eps=self.eps)
        y_log = spd_log(y, eps=self.eps)

        out_log = (1.0 - gate) * x_log + gate * y_log

        return out_log
