from __future__ import annotations

import math
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F

from src.models.SPDTransformer import SPDTransformer


FusionClassifierType = Literal["cosine", "linear"]


class CosineClassifier(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            num_classes: int,
            initial_scale: float = 10.0,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_classes < 2:
            raise ValueError("Cosine classifier needs positive features and >=2 classes.")
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive.")

        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_scale)))
        )
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = F.normalize(features, dim=-1, eps=self.eps)
        normalized_weight = F.normalize(self.weight, dim=-1, eps=self.eps)
        scale = F.softplus(self.logit_scale_raw) + self.eps
        return scale * F.linear(normalized_features, normalized_weight)


class TemporalPowerEncoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_dim: int,
            feature_dim: int,
            kernel_size: int,
            dropout: float,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden_dim, feature_dim) < 1:
            raise ValueError("Power encoder dimensions must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("power_kernel_size must be a positive odd integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("power_dropout must be in [0, 1).")

        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, log_power: torch.Tensor) -> torch.Tensor:
        return self.network(log_power)


class SPDShapePowerClassifier(nn.Module):
    """Separate covariance shape and band power before feature fusion.

    Every SPD token is decomposed as C = pS, where p = trace(C) / d and
    trace(S) = d. The shape matrix S is encoded by SPDTransformer, while
    log(p) is processed as a time series with frequency-region channels.
    """

    def __init__(
            self,
            num_heads: int,
            spd_in_dim: int,
            attention_dim: list[int],
            num_classes: int,
            stage_transition: bool,
            time_sequence_length: int,
            frequency_sequence_length: int,
            brain_region_sequence_length: int = 1,
            tau: float = 1.0,
            ffn_hidden_spd_dim: int | None = None,
            metric: str = "log-euclidean",
            depth: int = 1,
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
            power_hidden_dim: int = 16,
            power_feature_dim: int = 16,
            power_kernel_size: int = 5,
            power_dropout: float = 0.2,
            power_center_log: bool = True,
            fusion_classifier: FusionClassifierType = "cosine",
            fusion_dropout: float = 0.2,
            cosine_initial_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= fusion_dropout < 1.0:
            raise ValueError("fusion_dropout must be in [0, 1).")

        normalized_classifier = str(fusion_classifier).strip().lower().replace(
            "-", "_"
        )
        if normalized_classifier not in {"cosine", "linear"}:
            raise ValueError("fusion_classifier must be 'cosine' or 'linear'.")

        self.spd_in_dim = int(spd_in_dim)
        self.time_sequence_length = int(time_sequence_length)
        self.frequency_sequence_length = int(frequency_sequence_length)
        self.brain_region_sequence_length = int(brain_region_sequence_length)
        self.power_center_log = bool(power_center_log)
        self.eps = float(eps)
        self.transformer_out_dim = (
            int(attention_dim[-1]) if stage_transition else self.spd_in_dim
        )
        self.shape_feature_dim = (
            self.transformer_out_dim * (self.transformer_out_dim + 1) // 2
        )

        self.shape_encoder = SPDTransformer(
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
            learnable_metric_score=learnable_metric_score,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position_bias=use_position_bias,
            layer_norm_affine=layer_norm_affine,
            dropout=dropout,
            stage_projection_init=stage_projection_init,
            add_norm_type=add_norm_type,
        )
        self.shape_norm = nn.LayerNorm(self.shape_feature_dim)

        power_channels = (
            self.frequency_sequence_length * self.brain_region_sequence_length
        )
        self.power_encoder = TemporalPowerEncoder(
            input_channels=power_channels,
            hidden_dim=int(power_hidden_dim),
            feature_dim=int(power_feature_dim),
            kernel_size=int(power_kernel_size),
            dropout=float(power_dropout),
        )

        self.fusion_feature_dim = self.shape_feature_dim + int(power_feature_dim)
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        if normalized_classifier == "cosine":
            self.classifier = CosineClassifier(
                feature_dim=self.fusion_feature_dim,
                num_classes=num_classes,
                initial_scale=cosine_initial_scale,
                eps=eps,
            )
        else:
            self.classifier = nn.Linear(self.fusion_feature_dim, num_classes)
        self.fusion_classifier = normalized_classifier

        row, col = torch.triu_indices(
            self.transformer_out_dim,
            self.transformer_out_dim,
        )
        self.register_buffer("shape_triu_row", row, persistent=False)
        self.register_buffer("shape_triu_col", col, persistent=False)
        shape_scale = torch.ones(row.numel())
        shape_scale[row != col] = math.sqrt(2.0)
        self.register_buffer("shape_triu_scale", shape_scale, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shape_spd, log_power = self.decompose_covariance(x)
        shape_log, aux = self.shape_encoder(
            shape_spd,
            return_log=True,
            return_aux=return_aux,
        )

        token_dims = tuple(range(1, shape_log.ndim - 2))
        pooled_shape = shape_log.mean(dim=token_dims)
        shape_features = pooled_shape[
            ..., self.shape_triu_row, self.shape_triu_col
        ]
        shape_features = shape_features * self.shape_triu_scale.to(
            dtype=shape_features.dtype
        )
        shape_features = self.shape_norm(shape_features)

        power_series = self._power_as_channel_time(log_power)
        power_features = self.power_encoder(power_series)
        fused_features = self.fusion_dropout(
            torch.cat((shape_features, power_features), dim=-1)
        )
        return self.classifier(fused_features), aux

    def decompose_covariance(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim not in {4, 5, 6}:
            raise ValueError(
                "Expected SPD input shaped (batch,time,[frequency],[region],d,d), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-2:] != (self.spd_in_dim, self.spd_in_dim):
            raise ValueError(
                f"Expected {self.spd_in_dim}x{self.spd_in_dim} SPD matrices, "
                f"got {tuple(x.shape[-2:])}."
            )

        symmetric_x = 0.5 * (x + x.transpose(-1, -2))
        mean_diagonal = torch.diagonal(
            symmetric_x, dim1=-2, dim2=-1
        ).mean(dim=-1).clamp_min(self.eps)
        shape_spd = symmetric_x / mean_diagonal[..., None, None]
        log_power = mean_diagonal.log()
        if self.power_center_log:
            sample_axes = tuple(range(1, log_power.ndim))
            log_power = log_power - log_power.mean(
                dim=sample_axes,
                keepdim=True,
            )
        return shape_spd, log_power

    def _power_as_channel_time(self, log_power: torch.Tensor) -> torch.Tensor:
        if log_power.ndim == 2:
            log_power = log_power[:, :, None, None]
        elif log_power.ndim == 3:
            log_power = log_power[:, :, :, None]

        expected = (
            self.frequency_sequence_length,
            self.brain_region_sequence_length,
        )
        if tuple(log_power.shape[2:4]) != expected:
            raise ValueError(
                "Power token dimensions do not match model configuration: "
                f"expected frequency/region={expected}, got "
                f"{tuple(log_power.shape[2:4])}."
            )
        if log_power.shape[1] > self.time_sequence_length:
            raise ValueError(
                "Power time dimension exceeds configured sequence length: "
                f"{log_power.shape[1]} > {self.time_sequence_length}."
            )

        batch, time, frequency, region = log_power.shape
        return log_power.permute(0, 2, 3, 1).reshape(
            batch,
            frequency * region,
            time,
        )
