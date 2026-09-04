from typing import Any, Literal

import torch
from torch import nn

from src.models.SPDClassifierBase import SPDClassifierBase
from src.models.SPDTransformer import SPDTransformer
from src.models.SimplexWeightRegularization import (
    PoolingWeightMode,
    factorized_logits,
    normalize_pooling_weight_mode,
    regularize_simplex_weights,
    validate_probability,
    validate_unit_interval,
)

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
            ffn_tangent_mixer_rank: int = 0,
            metric: str = "log-euclidean",
            depth: int = 1,
            pooling: SPDPoolingMode = "weighted",
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["full", "low-rank"] = "low-rank",
            learnable_metric_score: Literal["qgk", "distance"] = "qgk",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: str = "trace",
            position_bias_axes: str | tuple[str, ...] | list[str] | None = None,
            position_bias_max: float = 0.5,
            attention_score_target_rms: float = 1.0,
            attention_score_clip: float = 5.0,
            share_metric_across_layers: bool | str | list[bool] = False,
            independent_metric_per_axis: bool | str = True,
            head_dropout: float = 0.0,
            pooling_weight_mode: PoolingWeightMode = "full",
            pooling_dropout: float = 0.0,
            pooling_uniform_mix: float = 0.0,
            pooling_mean_anchor: bool = False,
    ):
        super().__init__()
        pooling = self._normalize_pooling(pooling)

        self.attention_dim = attention_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.pooling_weight_mode = normalize_pooling_weight_mode(
            pooling_weight_mode
        )
        self.pooling_dropout = validate_probability(
            pooling_dropout,
            "pooling_dropout",
        )
        self.pooling_uniform_mix = validate_unit_interval(
            pooling_uniform_mix,
            "pooling_uniform_mix",
        )
        if not isinstance(pooling_mean_anchor, bool):
            raise TypeError(
                "pooling_mean_anchor must be a bool, "
                f"got {type(pooling_mean_anchor).__name__}."
            )
        self.pooling_mean_anchor = pooling_mean_anchor
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
            ffn_tangent_mixer_rank=ffn_tangent_mixer_rank,
            metric=metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            debug_attention_shape=debug_attention_shape,
            debug_tensor_stats=debug_tensor_stats,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_score=learnable_metric_score,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position_bias=use_position_bias,
            position_bias_axes=position_bias_axes,
            position_bias_max=position_bias_max,
            attention_score_target_rms=attention_score_target_rms,
            attention_score_clip=attention_score_clip,
            layer_norm_affine=layer_norm_affine,
            dropout=dropout,
            stage_projection_init=stage_projection_init,
            add_norm_type=add_norm_type,
            share_metric_across_layers=share_metric_across_layers,
            independent_metric_per_axis=independent_metric_per_axis,
            head_dropout=head_dropout,
        )

        if pooling == "weighted":
            token_shape = (
                time_sequence_length,
                frequency_sequence_length,
                brain_region_sequence_length,
            )
            if self.pooling_weight_mode == "full":
                self.token_weight_logits = nn.Parameter(torch.zeros(token_shape))
                self.token_axis_weight_logits = nn.ParameterList()
            else:
                self.register_parameter("token_weight_logits", None)
                self.token_axis_weight_logits = nn.ParameterList(
                    [nn.Parameter(torch.zeros(size)) for size in token_shape]
                )
            if self.pooling_mean_anchor:
                initial_gate = torch.tensor(0.1)
                self.pooling_gate_logit = nn.Parameter(torch.logit(initial_gate))
            else:
                self.register_parameter("pooling_gate_logit", None)
        else:
            self.register_parameter("token_weight_logits", None)
            self.token_axis_weight_logits = nn.ParameterList()
            self.register_parameter("pooling_gate_logit", None)

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
        if (
            self.token_weight_logits is None
            and len(self.token_axis_weight_logits) == 0
        ):
            raise RuntimeError("Weighted-pooling parameters are not initialized.")

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
        base_weights = torch.softmax(logits.reshape(-1), dim=0)
        weights = regularize_simplex_weights(
            base_weights,
            sample_count=x_log.shape[0],
            dropout=self.pooling_dropout,
            uniform_mix=self.pooling_uniform_mix,
            training=self.training,
        ).reshape(-1, *token_shape)
        view_shape = (weights.shape[0], *token_shape, 1, 1)
        token_dims = tuple(range(1, x_log.ndim - 2))
        weighted_log = (x_log * weights.view(view_shape)).sum(dim=token_dims)
        if self.pooling_gate_logit is None:
            return weighted_log

        mean_log = self._mean_pool(x_log)
        gate = torch.sigmoid(self.pooling_gate_logit).to(weighted_log.dtype)
        return torch.lerp(mean_log, weighted_log, gate)

    def _token_logits_for_shape(self, token_shape: tuple[int, ...]) -> torch.Tensor:
        if not 1 <= len(token_shape) <= 3:
            raise ValueError(
                "Weighted pooling supports time, time/frequency, or "
                f"time/frequency/region tokens, got token shape {token_shape}."
            )

        if self.pooling_weight_mode == "factorized":
            return factorized_logits(self.token_axis_weight_logits, token_shape)

        assert self.token_weight_logits is not None
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
