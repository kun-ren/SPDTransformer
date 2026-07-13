from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn

from src.models.SPDMDMClassifier import LogEuclideanMDMHead


PoolingMode = Literal["mean", "weighted"]


class TangentTransformerMDMClassifier(nn.Module):
    """
    Euclidean Transformer baseline for a sequence of SPD matrices.

    Each SPD token is mapped to the Log-Euclidean tangent space at the
    identity and isometrically vectorized. A native ``nn.TransformerEncoder``
    processes the resulting token sequence. Its output is projected back to a
    symmetric tangent-space matrix, pooled over tokens, and classified with
    the same differentiable Log-Euclidean MDM head used by SPDTransformer.
    """

    def __init__(
            self,
            spd_dim: int,
            num_classes: int,
            token_shape: tuple[int, ...],
            d_model: int,
            output_spd_dim: int | None = None,
            nhead: int = 1,
            num_layers: int = 1,
            dim_feedforward: int | None = None,
            dropout: float = 0.1,
            activation: Literal["relu", "gelu"] = "gelu",
            norm_first: bool = False,
            pooling: PoolingMode = "weighted",
            use_position_embedding: bool = True,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if spd_dim < 1:
            raise ValueError(f"spd_dim must be positive, got {spd_dim}.")
        if output_spd_dim is None:
            output_spd_dim = spd_dim
        if output_spd_dim < 1:
            raise ValueError(
                f"output_spd_dim must be positive, got {output_spd_dim}."
            )
        if not 1 <= len(token_shape) <= 3:
            raise ValueError(
                "token_shape must contain time, time/frequency, or "
                f"time/frequency/region dimensions, got {token_shape}."
            )
        if any(int(size) < 1 for size in token_shape):
            raise ValueError(f"token_shape must be positive, got {token_shape}.")
        if d_model < 1:
            raise ValueError(f"d_model must be positive, got {d_model}.")
        if nhead < 1 or d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        pooling = self._normalize_pooling(pooling)
        token_shape = tuple(int(size) for size in token_shape)
        tangent_dim = spd_dim * (spd_dim + 1) // 2
        output_tangent_dim = output_spd_dim * (output_spd_dim + 1) // 2
        num_tokens = math.prod(token_shape)
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        if dim_feedforward < 1:
            raise ValueError(
                f"dim_feedforward must be positive, got {dim_feedforward}."
            )

        self.spd_dim = int(spd_dim)
        self.output_spd_dim = int(output_spd_dim)
        self.num_classes = int(num_classes)
        self.token_shape = token_shape
        self.num_tokens = num_tokens
        self.tangent_dim = tangent_dim
        self.output_tangent_dim = output_tangent_dim
        self.d_model = int(d_model)
        self.pooling = pooling
        self.eps = float(eps)

        upper_indices = torch.triu_indices(spd_dim, spd_dim)
        upper_rows, upper_cols = upper_indices[0], upper_indices[1]
        vector_scales = torch.ones(tangent_dim)
        vector_scales[upper_rows != upper_cols] = math.sqrt(2.0)
        output_upper_indices = torch.triu_indices(output_spd_dim, output_spd_dim)
        output_upper_rows = output_upper_indices[0]
        output_upper_cols = output_upper_indices[1]
        output_vector_scales = torch.ones(output_tangent_dim)
        output_vector_scales[output_upper_rows != output_upper_cols] = math.sqrt(2.0)
        inverse_basis = torch.zeros(
            output_tangent_dim,
            output_spd_dim,
            output_spd_dim,
        )
        basis_indices = torch.arange(output_tangent_dim)
        inverse_basis[
            basis_indices,
            output_upper_rows,
            output_upper_cols,
        ] = 1.0 / output_vector_scales
        off_diagonal = output_upper_rows != output_upper_cols
        inverse_basis[
            basis_indices[off_diagonal],
            output_upper_cols[off_diagonal],
            output_upper_rows[off_diagonal],
        ] = 1.0 / output_vector_scales[off_diagonal]

        self.register_buffer("_upper_rows", upper_rows, persistent=False)
        self.register_buffer("_upper_cols", upper_cols, persistent=False)
        self.register_buffer("_vector_scales", vector_scales, persistent=False)
        self.register_buffer("_inverse_basis", inverse_basis, persistent=False)

        self.input_projection: nn.Module
        if tangent_dim == d_model:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(tangent_dim, d_model)

        if use_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.empty(1, num_tokens, d_model)
            )
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("position_embedding", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.output_projection: nn.Module
        if d_model == output_tangent_dim:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(d_model, output_tangent_dim)

        if pooling == "weighted":
            self.token_weight_logits = nn.Parameter(torch.zeros(token_shape))
        else:
            self.register_parameter("token_weight_logits", None)

        self.mdm_head = LogEuclideanMDMHead(
            spd_dim=output_spd_dim,
            num_classes=num_classes,
            eps=eps,
        )

    @staticmethod
    def _normalize_pooling(pooling: str) -> PoolingMode:
        normalized = str(pooling).strip().lower().replace("-", "_")
        if normalized in {"mean", "mdm_mean"}:
            return "mean"
        if normalized in {
            "weighted",
            "weight",
            "mdm_weighted",
            "learned_weighted",
        }:
            return "weighted"
        raise ValueError(
            "TangentTransformerMDMClassifier pooling must be 'mean' or "
            f"'weighted', got {pooling!r}."
        )

    def spd_log_map(self, x: torch.Tensor) -> torch.Tensor:
        """Map SPD matrices to symmetric matrices in the tangent space at I."""
        self._validate_matrix_shape(x)
        x = 0.5 * (x + x.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(x)
        log_eigenvalues = eigenvalues.clamp_min(self.eps).log()
        x_log = (
            eigenvectors * log_eigenvalues.unsqueeze(-2)
        ) @ eigenvectors.transpose(-1, -2)
        return 0.5 * (x_log + x_log.transpose(-1, -2))

    def symmetric_matrix_to_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Isometric vech: off-diagonal entries are multiplied by sqrt(2)."""
        self._validate_matrix_shape(x)
        x = 0.5 * (x + x.transpose(-1, -2))
        return (
            x[..., self._upper_rows, self._upper_cols]
            * self._vector_scales
        )

    def vector_to_symmetric_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`symmetric_matrix_to_vector`."""
        if x.shape[-1] != self.output_tangent_dim:
            raise ValueError(
                f"Expected output tangent vector dimension "
                f"{self.output_tangent_dim}, "
                f"got {tuple(x.shape)}."
            )
        return torch.einsum("...k,kij->...ij", x, self._inverse_basis)

    def token_weights(
            self,
            actual_token_shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if actual_token_shape is None:
            actual_token_shape = self.token_shape
        if self.token_weight_logits is None:
            return torch.full(
                actual_token_shape,
                1.0 / self.num_tokens,
                device=self._inverse_basis.device,
                dtype=self._inverse_basis.dtype,
            )
        index = tuple(slice(0, size) for size in actual_token_shape)
        index += (0,) * (len(self.token_shape) - len(actual_token_shape))
        logits = self.token_weight_logits[index]
        return torch.softmax(logits.reshape(-1), dim=0).reshape(actual_token_shape)

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        self._validate_input_shape(x)
        batch_size = x.shape[0]
        actual_token_shape = tuple(x.shape[1:-2])
        actual_num_tokens = math.prod(actual_token_shape)

        x_log = self.spd_log_map(x)
        tangent_tokens = self.symmetric_matrix_to_vector(x_log).reshape(
            batch_size,
            actual_num_tokens,
            self.tangent_dim,
        )
        encoded_tokens = self.input_projection(tangent_tokens)
        if self.position_embedding is not None:
            encoded_tokens = encoded_tokens + self.position_embedding
        encoded_tokens = self.encoder(encoded_tokens)

        output_vectors = self.output_projection(encoded_tokens)
        output_log_matrices = self.vector_to_symmetric_matrix(output_vectors)
        weights = self.token_weights(actual_token_shape).reshape(
            1,
            actual_num_tokens,
            1,
            1,
        )
        pooled_log = (output_log_matrices * weights).sum(dim=1)
        logits = self.mdm_head(pooled_log)

        if not return_aux:
            return logits
        return logits, {
            "token_weights": weights.reshape(actual_token_shape),
            "pooled_log": pooled_log,
            "encoded_log_tokens": output_log_matrices.reshape(
                batch_size,
                *actual_token_shape,
                self.output_spd_dim,
                self.output_spd_dim,
            ),
        }

    def _validate_matrix_shape(self, x: torch.Tensor) -> None:
        if x.ndim < 2 or x.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected trailing matrix shape ({self.spd_dim}, {self.spd_dim}), "
                f"got {tuple(x.shape)}."
            )

    def _validate_input_shape(self, x: torch.Tensor) -> None:
        if x.ndim not in {4, 5, 6}:
            raise ValueError(
                "Expected input shape with time, time/frequency, or "
                "time/frequency/region token dimensions, "
                f"got {tuple(x.shape)}."
            )
        self._validate_matrix_shape(x)
        actual_token_shape = tuple(x.shape[1:-2])
        expected_prefix = self.token_shape[:len(actual_token_shape)]
        omitted_shape = self.token_shape[len(actual_token_shape):]
        if actual_token_shape != expected_prefix or any(
            size != 1 for size in omitted_shape
        ):
            raise ValueError(
                f"Expected token shape {self.token_shape}, "
                f"got {actual_token_shape}."
            )
