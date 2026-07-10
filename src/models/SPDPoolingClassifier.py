from typing import Tuple, Any, Literal

import torch
from torch import nn

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
            num_heads: int,
            spd_in_dim: int,
            attention_dim: [int],
            num_classes: int,
            stage_transition: bool,
            time_sequence_length: int,
            frequency_sequence_length: int,
            brain_region_sequence_length: int = 1,
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
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: str = "trace",
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
        self.transformer_out_dim = attention_dim[-1] if stage_transition else spd_in_dim
        self.feature_dim = self.transformer_out_dim * (self.transformer_out_dim + 1) // 2
        self.classifier_feature_dim = (
            self.feature_dim * frequency_sequence_length
            if pooling == "band_mean"
            else self.feature_dim
        )

        self.encoder = SPDTransformer(
            num_heads=num_heads,
            spd_in_dim=spd_in_dim,
            attention_dim=attention_dim,
            depth=depth,
            stage_transition=stage_transition,
            time_sequence_length=time_sequence_length,
            frequency_sequence_length=frequency_sequence_length,
            brain_region_sequence_length=brain_region_sequence_length,
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
            stage_projection_init=stage_projection_init,
            add_norm_type=add_norm_type,
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

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[Any, Any]:

        x_log, aux = self.encoder(
            x,
            return_log=True,
            return_aux=return_aux,
        )


        if self.pooling == "mean":
            pooled_log = self._mean_pool(x_log)
            features = self.upper_triangular_vectorize(pooled_log)
        elif self.pooling == "band_mean":
            features = self._band_mean_pool_features(x_log)
        else:
            pooled_log = self._attention_pool(x_log)
            features = self.upper_triangular_vectorize(pooled_log)

        logits = self.classifier(features)

        return logits, aux

    def _mean_pool(self, x_log: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-domain mean over all tokens that belong to one trial.
        :return torch.Tensor: (batch, channels, channels)
        """
        token_dims = tuple(range(1, x_log.ndim - 2))
        return x_log.mean(dim=token_dims)

    def _band_mean_pool_features(self, x_log: torch.Tensor) -> torch.Tensor:
        """
        Average over time but keep frequency-band features separate.

        For 5D/6D input, this returns one tangent feature vector per frequency
        band and concatenates them. Region tokens are averaged inside each
        frequency band. For 4D input, it falls back to temporal mean pooling.
        """
        if x_log.ndim == 4:
            pooled_log = x_log.mean(dim=1)
            return self.upper_triangular_vectorize(pooled_log)

        if x_log.ndim == 6:
            band_log = x_log.mean(dim=(1, 3))
            band_features = self.upper_triangular_vectorize(band_log)
            return band_features.reshape(band_features.shape[0], -1)

        if x_log.ndim != 5:
            raise ValueError(
                "Expected encoder output shape "
                "(batch, time, channels, channels) or "
                "(batch, time, frequency, channels, channels) or "
                "(batch, time, frequency, brain_region, channels, channels), "
                f"got {tuple(x_log.shape)}."
            )

        band_log = x_log.mean(dim=1)
        band_features = self.upper_triangular_vectorize(band_log)
        return band_features.reshape(band_features.shape[0], -1)

    def _attention_pool(self, x_log: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return: pooled spd matrix, (batch, channels, channels)
        """
        batch_size = x_log.shape[0]
        spd_dim = x_log.shape[-1]
        # log_tokens = (batch, tim x frequency_bands, channels, channels)
        log_tokens = x_log.reshape(batch_size, -1, spd_dim, spd_dim)
        token_features = self.upper_triangular_vectorize(log_tokens)


        scores = self.pool_score(token_features).squeeze(-1)

        weights = torch.softmax(scores, dim=-1)

        return torch.einsum("bt,btmn->bmn", weights, log_tokens)
