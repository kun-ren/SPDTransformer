from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.models.StandardSingleHeadTransformer import TransformerEncoder


PoolingMode = Literal["mean", "first", "last", "center"]


@dataclass(frozen=True)
class EpochTagTransformerOutput:
    logits: torch.Tensor
    epoch_features: torch.Tensor
    encoded_tokens: torch.Tensor


class SPDEpochTagEmbedding(nn.Module):
    """
    Embeds a sequence of SPD matrices as per-matrix token groups.

    For every SPD matrix in a trial sequence, the embedding creates:
        [Epoch] + one row token per SPD matrix row

    Input:
        x: (batch, epochs, channels, channels)

    Output:
        tokens: (batch, epochs * (channels + 1), embed_dim)
    """

    def __init__(
        self,
        channels: int,
        max_epochs: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.max_epochs = max_epochs
        self.embed_dim = embed_dim

        self.row_projection = nn.Linear(channels, embed_dim)
        self.row_position_embedding = nn.Parameter(
            torch.empty(1, 1, channels, embed_dim)
        )
        self.time_position_embedding = nn.Parameter(
            torch.empty(1, max_epochs, 1, embed_dim)
        )
        self.epoch_tag = nn.Parameter(torch.empty(1, 1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.row_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.epoch_tag, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape (batch, epochs, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        batch_size, epochs, rows, cols = x.shape
        if epochs > self.max_epochs:
            raise ValueError(
                f"Input has {epochs} epochs, but max_epochs={self.max_epochs}."
            )
        if rows != self.channels or cols != self.channels:
            raise ValueError(
                f"Expected matrices with shape {self.channels}x{self.channels}, "
                f"got {rows}x{cols}."
            )

        time_pos = self.time_position_embedding[:, :epochs]

        row_tokens = self.row_projection(x)
        row_tokens = row_tokens + self.row_position_embedding + time_pos

        epoch_tags = self.epoch_tag + time_pos
        epoch_tags = epoch_tags.expand(batch_size, epochs, -1, -1)

        grouped_tokens = torch.cat([epoch_tags, row_tokens], dim=2)
        grouped_tokens = self.dropout(grouped_tokens)
        return grouped_tokens.reshape(
            batch_size,
            epochs * (self.channels + 1),
            self.embed_dim,
        )


class EpochTagTransformer(nn.Module):
    """
    Five-layer Transformer baseline for SPD trial sequences.

    Each SPD matrix is assumed to represent one 0.5s epoch. A learnable
    [Epoch] token is inserted before the row tokens of every SPD matrix. The
    final classifier pools only [Epoch] token outputs.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        max_epochs: int,
        embed_dim: int = 128,
        ffn_dim: int = 256,
        depth: int = 5,
        dropout: float = 0.1,
        pooling: PoolingMode = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "first", "last", "center"}:
            raise ValueError(f"Unsupported pooling mode: {pooling!r}")

        self.channels = channels
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.embed_dim = embed_dim
        self.depth = depth
        self.pooling = pooling

        self.embedding = SPDEpochTagEmbedding(
            channels=channels,
            max_epochs=max_epochs,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            depth=depth,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | EpochTagTransformerOutput:
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape (batch, epochs, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        batch_size, epochs, channels, cols = x.shape
        if channels != self.channels or cols != self.channels:
            raise ValueError(
                f"Expected matrices with shape {self.channels}x{self.channels}, "
                f"got {channels}x{cols}."
            )

        tokens = self.embedding(x)
        encoded_tokens = self.encoder(tokens)
        encoded_groups = encoded_tokens.reshape(
            batch_size,
            epochs,
            self.channels + 1,
            self.embed_dim,
        )

        epoch_features = encoded_groups[:, :, 0, :]
        pooled = self._pool_epoch_features(epoch_features, pool_mask)
        logits = self.classifier(pooled)

        if return_features:
            return EpochTagTransformerOutput(
                logits=logits,
                epoch_features=epoch_features,
                encoded_tokens=encoded_tokens,
            )
        return logits

    def _pool_epoch_features(
        self,
        epoch_features: torch.Tensor,
        pool_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if pool_mask is not None:
            if pool_mask.shape != epoch_features.shape[:2]:
                raise ValueError(
                    "pool_mask must have shape (batch, epochs), "
                    f"got {tuple(pool_mask.shape)}."
                )
            weights = pool_mask.to(epoch_features.dtype).unsqueeze(-1)
            weights_sum = weights.sum(dim=1).clamp_min(1.0)
            return (epoch_features * weights).sum(dim=1) / weights_sum

        if self.pooling == "mean":
            return epoch_features.mean(dim=1)
        if self.pooling == "first":
            return epoch_features[:, 0]
        if self.pooling == "last":
            return epoch_features[:, -1]
        if self.pooling == "center":
            return epoch_features[:, epoch_features.shape[1] // 2]

        raise RuntimeError(f"Unexpected pooling mode: {self.pooling!r}")
