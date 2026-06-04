from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


PoolingMode = Literal["mean", "first", "last", "center"]


@dataclass(frozen=True)
class StandardSingleHeadTransformerOutput:
    logits: torch.Tensor
    epoch_cls: torch.Tensor
    sequence_features: torch.Tensor


class SingleHeadSelfAttention(nn.Module):
    """Plain scaled dot-product self-attention with one attention head."""

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim**-0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, v)
        return self.output(context)


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm Transformer encoder block using single-head attention."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = SingleHeadSelfAttention(embed_dim, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_dropout(self.attention(self.attention_norm(x)))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        depth: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class EpochMatrixEmbedding(nn.Module):
    """
    Converts each epoch matrix into row tokens plus an epoch-local CLS token.

    Input shape:
        (batch, epochs, channels, channels)

    Output shape:
        (batch, epochs, channels + 1, embed_dim)
    """

    def __init__(
        self,
        channels: int,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim

        self.row_projection = nn.Linear(channels, embed_dim)
        self.row_position_embedding = nn.Parameter(
            torch.empty(1, 1, channels, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.row_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape (batch, epochs, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        batch_size, epochs, rows, cols = x.shape
        if rows != self.channels or cols != self.channels:
            raise ValueError(
                f"Expected square matrices with {self.channels} channels, "
                f"got {rows}x{cols}."
            )

        row_tokens = self.row_projection(x)
        row_tokens = row_tokens + self.row_position_embedding

        cls_tokens = self.cls_token.expand(batch_size, epochs, -1, -1)
        tokens = torch.cat([cls_tokens, row_tokens], dim=2)
        return self.dropout(tokens)


class StandardSingleHeadTransformer(nn.Module):
    """
    Baseline Transformer for inputs shaped (batch, epochs, channels, channels).

    Stage 1:
        Each epoch matrix is embedded as [CLS] + channel-row tokens, then encoded
        independently by an intra-epoch single-head Transformer.

    Stage 2:
        The CLS output from every epoch forms a rest+motion epoch sequence. A
        second single-head Transformer models the temporal relation between
        epochs before the final classifier.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        max_epochs: int,
        embed_dim: int = 128,
        ffn_dim: int = 256,
        intra_epoch_depth: int = 1,
        inter_epoch_depth: int = 1,
        dropout: float = 0.1,
        pooling: PoolingMode = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "first", "last", "center"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        self.channels = channels
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.embed_dim = embed_dim
        self.pooling = pooling

        self.embedding = EpochMatrixEmbedding(
            channels=channels,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.intra_epoch_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            depth=intra_epoch_depth,
            dropout=dropout,
        )

        self.epoch_position_embedding = nn.Parameter(
            torch.empty(1, max_epochs, embed_dim)
        )
        self.inter_epoch_dropout = nn.Dropout(dropout)
        self.inter_epoch_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            depth=inter_epoch_depth,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.epoch_position_embedding, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | StandardSingleHeadTransformerOutput:
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape (batch, epochs, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        batch_size, epochs, channels, cols = x.shape
        if epochs > self.max_epochs:
            raise ValueError(
                f"Input has {epochs} epochs, but max_epochs={self.max_epochs}."
            )
        if channels != self.channels or cols != self.channels:
            raise ValueError(
                f"Expected matrix size {self.channels}x{self.channels}, "
                f"got {channels}x{cols}."
            )

        epoch_tokens = self.embedding(x)
        epoch_tokens = epoch_tokens.reshape(
            batch_size * epochs,
            self.channels + 1,
            self.embed_dim,
        )

        epoch_tokens = self.intra_epoch_encoder(epoch_tokens)
        epoch_cls = epoch_tokens[:, 0].reshape(batch_size, epochs, self.embed_dim)

        sequence_features = epoch_cls + self.epoch_position_embedding[:, :epochs]
        sequence_features = self.inter_epoch_dropout(sequence_features)
        sequence_features = self.inter_epoch_encoder(sequence_features)

        pooled = self._pool_sequence(sequence_features, pool_mask=pool_mask)
        logits = self.classifier(pooled)

        if return_features:
            return StandardSingleHeadTransformerOutput(
                logits=logits,
                epoch_cls=epoch_cls,
                sequence_features=sequence_features,
            )
        return logits

    def _pool_sequence(
        self,
        sequence_features: torch.Tensor,
        pool_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pool_mask is not None:
            if pool_mask.shape != sequence_features.shape[:2]:
                raise ValueError(
                    "pool_mask must have shape (batch, epochs), "
                    f"got {tuple(pool_mask.shape)}."
                )
            weights = pool_mask.to(sequence_features.dtype).unsqueeze(-1)
            weights_sum = weights.sum(dim=1).clamp_min(1.0)
            return (sequence_features * weights).sum(dim=1) / weights_sum

        if self.pooling == "mean":
            return sequence_features.mean(dim=1)
        if self.pooling == "first":
            return sequence_features[:, 0]
        if self.pooling == "last":
            return sequence_features[:, -1]
        if self.pooling == "center":
            return sequence_features[:, sequence_features.shape[1] // 2]

        raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")
