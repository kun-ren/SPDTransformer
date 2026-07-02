from typing import Tuple, Any, Literal

import torch
from torch import nn

from src.models.SPDAttention import spd_log
from src.models.SPDClassifierBase import SPDClassifierBase
from src.models.SPDTransformer import SPDTransformer

SPDPoolingMode = Literal["mean", "band_mean", "attention"]
class SPDPoolingClassifier(SPDClassifierBase):
    """
    Classifier that pools all SPD tokens after the SPDTransformer encoder.

    Input:
        4D: (batch, time, channels, channels)
        5D: (batch, time, frequency_bands, channels, channels)

    Classification:
        encoder -> log map -> mean/attention pooling over all tokens
        -> upper triangular vector -> linear classifier
    """

    def __init__(
            self,
            spd_in_dim: int,
            attention_dim: int,
            num_classes: int,
            stage_transition: bool,
            time_sequence_length: int,
            frequency_sequence_length: int,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            pooling: SPDPoolingMode = "attention",
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
    ):
        super().__init__()
        if pooling not in {"mean", "band_mean", "attention"}:
            raise ValueError(
                "SPDPoolingClassifier pooling must be 'mean', "
                f"'band_mean', or 'attention', got {pooling!r}."
            )

        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.debug_tensor_stats = debug_tensor_stats
        self.encoder_spd_dim = attention_dim if stage_transition else spd_in_dim
        self.feature_dim = self.encoder_spd_dim * (self.encoder_spd_dim + 1) // 2
        self.classifier_feature_dim = (
            self.feature_dim * frequency_sequence_length
            if pooling == "band_mean"
            else self.feature_dim
        )

        self.encoder = SPDTransformer(
            spd_in_dim=spd_in_dim,
            attention_dim=attention_dim,
            depth=depth,
            stage_transition=stage_transition,
            time_sequence_length=time_sequence_length,
            frequency_sequence_length=frequency_sequence_length,
            tau=tau,
            ffn_hidden_spd_dim=ffn_hidden_spd_dim,
            metric=metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            debug_attention_shape=debug_attention_shape,
            debug_tensor_stats=debug_tensor_stats,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position_bias=use_position_bias,
            layer_norm_affine=layer_norm_affine,
            dropout=dropout,
        )

        if pooling == "attention":
            self.pool_score = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 1),
            )
        else:
            self.pool_score = None

        self.classifier = self.build_linear_classifier(
            feature_dim=self.classifier_feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[Any, Any]:

        x, aux = self.encoder(x)


        if self.pooling == "mean":
            pooled_log = self._mean_pool(x)
            features = self.upper_triangular_vectorize(pooled_log)
        elif self.pooling == "band_mean":
            features = self._band_mean_pool_features(x)
        else:
            pooled_log = self._attention_pool(x)
            features = self.upper_triangular_vectorize(pooled_log)

        logits = self.classifier(features)

        return logits, aux

    def _mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean value across all tokens that belong to one trial
        :return torch.Tensor: (batch, channels, channels)
        """
        log_x = spd_log(x)
        token_dims = tuple(range(1, log_x.ndim - 2))
        return log_x.mean(dim=token_dims)

    def _band_mean_pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Average over time but keep frequency-band features separate.

        For 5D input, this returns one tangent feature vector per frequency
        band and concatenates them. For 4D input, it falls back to temporal
        mean pooling.
        """
        log_x = spd_log(x)
        if log_x.ndim == 4:
            pooled_log = log_x.mean(dim=1)
            return self.upper_triangular_vectorize(pooled_log)

        if log_x.ndim != 5:
            raise ValueError(
                "Expected encoder output shape "
                "(batch, time, channels, channels) or "
                "(batch, time, frequency, channels, channels), "
                f"got {tuple(log_x.shape)}."
            )

        band_log = log_x.mean(dim=1)
        band_features = self.upper_triangular_vectorize(band_log)
        return band_features.reshape(band_features.shape[0], -1)

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return: pooled spd matrix, (batch, channels, channels)
        """
        batch_size = x.shape[0]
        spd_dim = x.shape[-1]
        log_x = spd_log(x)
        # log_tokens = (batch, tim x frequency_bands, channels, channels)
        log_tokens = log_x.reshape(batch_size, -1, spd_dim, spd_dim)
        token_features = self.upper_triangular_vectorize(log_tokens)


        scores = self.pool_score(token_features).squeeze(-1)

        weights = torch.softmax(scores, dim=-1)

        return torch.einsum("bt,btmn->bmn", weights, log_tokens)
