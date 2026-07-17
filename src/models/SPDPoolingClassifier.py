from typing import Any, Literal

import torch
from torch import nn

from src.models.SPDClassifierBase import SPDClassifierBase
from src.models.SPDTransformer import SPDTransformer

SPDPoolingMode = Literal["mean", "weighted"]


class SPDPoolingClassifier(SPDClassifierBase):
    """
    Classifier that pools all SPD tokens after the SPDTransformer encoder.

    Input:
        4D: (batch, time, channels, channels)
        5D: (batch, time, frequency_bands, channels, channels)
        6D: (batch, time, frequency_bands, brain_regions, channels, channels)

    Classification:
        encoder -> log map -> mean/weighted pooling over all tokens
        -> upper triangular vector -> linear classifier

    Weighted pooling learns one sample-independent scalar for each
    time/frequency/region position. Softmax-normalized weights are shared by
    all samples and all entries of the SPD matrix at that position.
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
            pooling: SPDPoolingMode = "weighted",
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
        pooling = self._normalize_pooling(pooling)

        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.debug_tensor_stats = debug_tensor_stats
        self.transformer_out_dim = attention_dim[-1] if stage_transition else spd_in_dim
        self.feature_dim = self.transformer_out_dim * (self.transformer_out_dim + 1) // 2
        self.classifier_feature_dim = self.feature_dim

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

        if pooling == "weighted":
            self.token_weight_logits = nn.Parameter(
                torch.zeros(
                    time_sequence_length,
                    frequency_sequence_length,
                    brain_region_sequence_length,
                )
            )
        else:
            self.register_parameter("token_weight_logits", None)

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
        else:
            pooled_log = self._weighted_pool(x_log)
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

    def _weighted_pool(self, x_log: torch.Tensor) -> torch.Tensor:
        if self.token_weight_logits is None:
            raise RuntimeError("token_weight_logits is only defined for weighted pooling.")

        if x_log.ndim not in {4, 5, 6}:
            raise ValueError(
                "Expected encoder output shape "
                "(batch, time, channels, channels), "
                "(batch, time, frequency, channels, channels), or "
                "(batch, time, frequency, brain_region, channels, channels), "
                f"got {tuple(x_log.shape)}."
            )

        token_shape = tuple(x_log.shape[1:-2])
        logits = self._token_logits_for_shape(token_shape)
        weights = torch.softmax(logits.reshape(-1), dim=0).reshape(token_shape)
        view_shape = (1, *token_shape, 1, 1)
        token_dims = tuple(range(1, x_log.ndim - 2))
        return (x_log * weights.view(view_shape)).sum(dim=token_dims)

    def _token_logits_for_shape(self, token_shape: tuple[int, ...]) -> torch.Tensor:
        assert self.token_weight_logits is not None
        if not 1 <= len(token_shape) <= 3:
            raise ValueError(
                "Weighted pooling supports time, time/frequency, or "
                f"time/frequency/region tokens, got token shape {token_shape}."
            )

        max_shape = self.token_weight_logits.shape
        if any(size > max_size for size, max_size in zip(token_shape, max_shape)):
            raise ValueError(
                "Weighted pooling was initialized for "
                f"{tuple(max_shape)} time/frequency/region tokens, "
                f"but encoder output has {token_shape}."
            )

        time_len = token_shape[0]
        if len(token_shape) == 1:
            return self.token_weight_logits[:time_len, 0, 0]
        frequency_len = token_shape[1]
        if len(token_shape) == 2:
            return self.token_weight_logits[:time_len, :frequency_len, 0]
        region_len = token_shape[2]
        return self.token_weight_logits[:time_len, :frequency_len, :region_len]

    @staticmethod
    def _normalize_pooling(pooling: str) -> SPDPoolingMode:
        normalized = str(pooling).strip().lower().replace("-", "_")
        if normalized == "mean":
            return "mean"
        if normalized in {"weighted", "weight", "learned_weighted"}:
            return "weighted"
        raise ValueError(
            "SPDPoolingClassifier pooling must be 'mean' or 'weighted', "
            f"got {pooling!r}."
        )
