from typing import Literal

import torch
from torch import nn

from src.models.SPDAttention import spd_log
from src.models.SPDClassifierBase import SPDClassifierBase
from src.models.SPDTransformer import SPDTransformer


class SPDTaskTagClassifier(SPDClassifierBase):
    """
    Classifier that inserts an SPD [TASK] token before the encoder.

    For 5D input, one task token is inserted on the time axis for every
    frequency band. After encoding, only task-token outputs are used for
    classification; non-task tokens are not pooled into the classifier.
    """

    def __init__(
            self,
            spd_in_dim: int,
            spd_out_dim: int,
            num_classes: int,
            time_sequence_length: int,
            frequency_sequence_length: int,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
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
        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.num_classes = num_classes
        self.debug_tensor_stats = debug_tensor_stats
        self.feature_dim = spd_out_dim * (spd_out_dim + 1) // 2

        task_log_init = torch.diag(torch.linspace(-1e-3, 1e-3, spd_in_dim))
        self.task_log_token = nn.Parameter(task_log_init)
        self.encoder = SPDTransformer(
            spd_in_dim=spd_in_dim,
            spd_out_dim=spd_out_dim,
            time_sequence_length=time_sequence_length + 1,
            frequency_sequence_length=frequency_sequence_length,
            tau=tau,
            depth=depth,
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
        self.classifier = self.build_linear_classifier(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._prepend_task_token(x)

        x = self.encoder(x)

        task_log = self._extract_task_log_feature(x)

        features = self.upper_triangular_vectorize(task_log)
        logits = self.classifier(features)
        return logits

    def _prepend_task_token(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return: (batch, time + 1, channels, channels) or (batch, time + 1, frequency_bands, channels, channels)
        """
        if x.ndim not in {4, 5}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency_bands, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        task_log = 0.5 * (self.task_log_token + self.task_log_token.transpose(-1, -2)) # sym
        task_log = task_log.to(device=x.device, dtype=x.dtype)

        # matrix_exp remains differentiable when the log token has repeated
        # eigenvalues, unlike an eigendecomposition-based implementation.
        task_token = torch.matrix_exp(task_log)  # Identity  matrix I
        eye = torch.eye(self.spd_in_dim, device=x.device, dtype=x.dtype)
        task_token = 0.5 * (task_token + task_token.transpose(-1, -2)) + self.eps * eye # sym

        if x.ndim == 4:
            batch_size = x.shape[0]
            task_token = task_token.expand(batch_size, 1, -1, -1)
        else:
            batch_size, _, n_bands = x.shape[:3]
            task_token = task_token.expand(batch_size, 1, n_bands, -1, -1)

        return torch.cat([task_token, x], dim=1)

    def _extract_task_log_feature(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, channels, channels)
        if x.ndim == 4:
            return spd_log(x[:, 0])
        # (batch, time, frequency_bands, channels, channels)
        task_tokens = x[:, 0]
        return spd_log(task_tokens).mean(dim=1)