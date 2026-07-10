from typing import Literal

import torch
from torch import nn

from src.models.SPDMDMClassifier import LogEuclideanMDMHead
from src.models.SPDTransformer import SPDTransformer


TaskTagPoolingMode = Literal["mean", "weighted"]


class SPDTaskTagClassifier(nn.Module):
    """
    Classifier that prepends a learnable SPD task token before the encoder.

    The encoder is queried with return_log=True. Classification then pools only
    the encoded task-token log matrices and feeds them to a log-Euclidean MDM
    head.
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
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            pooling: TaskTagPoolingMode = "mean",
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

        self.spd_in_dim = spd_in_dim
        self.transformer_out_dim = attention_dim[-1] if stage_transition else spd_in_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.debug_tensor_stats = debug_tensor_stats
        self.eps = eps

        task_log_init = torch.diag(torch.linspace(-1e-3, 1e-3, spd_in_dim))
        self.task_log_token = nn.Parameter(task_log_init)

        self.encoder = SPDTransformer(
            num_heads=num_heads,
            spd_in_dim=spd_in_dim,
            attention_dim=attention_dim,
            stage_transition=stage_transition,
            time_sequence_length=time_sequence_length + 1,
            frequency_sequence_length=frequency_sequence_length,
            brain_region_sequence_length=brain_region_sequence_length,
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
            stage_projection_init=stage_projection_init,
            add_norm_type=add_norm_type,
        )
        self.mdm_head = LogEuclideanMDMHead(
            spd_dim=self.transformer_out_dim,
            num_classes=num_classes,
            eps=eps,
        )

        if pooling == "weighted":
            self.task_frequency_weight_logits = nn.Parameter(
                torch.zeros(frequency_sequence_length, brain_region_sequence_length)
            )
        else:
            self.register_parameter("task_frequency_weight_logits", None)

    @staticmethod
    def _normalize_pooling(pooling: str) -> TaskTagPoolingMode:
        normalized = str(pooling).strip().lower().replace("-", "_")
        if normalized in {"mean", "task", "task_mean", "mdm_mean"}:
            return "mean"
        if normalized in {
            "weighted",
            "weight",
            "task_weighted",
            "mdm_weighted",
            "learned_weighted",
        }:
            return "weighted"
        raise ValueError(
            "SPDTaskTagClassifier pooling must be 'mean' or 'weighted', "
            f"got {pooling!r}."
        )

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self._prepend_task_token(x)
        x_log, aux = self.encoder(
            x,
            return_log=True,
            return_aux=return_aux,
        )
        task_log = self._pool_task_log_feature(x_log)
        logits = self.mdm_head(task_log)
        return logits, aux

    def _prepend_task_token(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in {4, 5, 6}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency, channels, channels) or "
                "(batch, time, frequency, brain_region, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        task_log = 0.5 * (
            self.task_log_token + self.task_log_token.transpose(-1, -2)
        )
        task_log = task_log.to(device=x.device, dtype=x.dtype)
        task_token = torch.matrix_exp(task_log)

        eye = torch.eye(self.spd_in_dim, device=x.device, dtype=x.dtype)
        task_token = 0.5 * (
            task_token + task_token.transpose(-1, -2)
        ) + self.eps * eye

        if x.ndim == 4:
            batch_size = x.shape[0]
            task_token = task_token.reshape(1, 1, self.spd_in_dim, self.spd_in_dim)
            task_token = task_token.expand(batch_size, 1, -1, -1)
        elif x.ndim == 5:
            batch_size, _, n_bands = x.shape[:3]
            task_token = task_token.reshape(
                1,
                1,
                1,
                self.spd_in_dim,
                self.spd_in_dim,
            )
            task_token = task_token.expand(batch_size, 1, n_bands, -1, -1)
        else:
            batch_size, _, n_bands, n_regions = x.shape[:4]
            task_token = task_token.reshape(
                1,
                1,
                1,
                1,
                self.spd_in_dim,
                self.spd_in_dim,
            )
            task_token = task_token.expand(
                batch_size,
                1,
                n_bands,
                n_regions,
                -1,
                -1,
            )

        return torch.cat([task_token, x], dim=1)

    def _pool_task_log_feature(self, x_log: torch.Tensor) -> torch.Tensor:
        if x_log.ndim == 4:
            return x_log[:, 0]

        if x_log.ndim == 5:
            task_tokens = x_log[:, 0]
            if self.pooling == "mean":
                return task_tokens.mean(dim=1)
            return self._weighted_pool_task_tokens(task_tokens)

        if x_log.ndim == 6:
            task_tokens = x_log[:, 0]
            if self.pooling == "mean":
                return task_tokens.mean(dim=(1, 2))
            return self._weighted_pool_task_tokens(task_tokens)

        raise ValueError(
            "Expected encoder log output shape "
            "(batch, time, channels, channels) or "
            "(batch, time, frequency, channels, channels) or "
            "(batch, time, frequency, brain_region, channels, channels), "
            f"got {tuple(x_log.shape)}."
        )

    def _weighted_pool_task_tokens(self, task_tokens: torch.Tensor) -> torch.Tensor:
        if self.task_frequency_weight_logits is None:
            raise RuntimeError(
                "task_frequency_weight_logits is only defined for weighted pooling."
            )

        if task_tokens.ndim not in {4, 5}:
            raise ValueError(
                "Expected task token shape "
                "(batch, frequency, channels, channels) or "
                "(batch, frequency, brain_region, channels, channels), "
                f"got {tuple(task_tokens.shape)}."
            )

        frequency_len = task_tokens.shape[1]
        region_len = task_tokens.shape[2] if task_tokens.ndim == 5 else 1
        if (
                frequency_len > self.task_frequency_weight_logits.shape[0]
                or region_len > self.task_frequency_weight_logits.shape[1]
        ):
            raise ValueError(
                "Weighted task pooling was initialized for "
                f"{tuple(self.task_frequency_weight_logits.shape)} "
                "frequency/region tokens, "
                f"but encoder output has {(frequency_len, region_len)}."
            )

        if task_tokens.ndim == 4:
            logits = self.task_frequency_weight_logits[:frequency_len, 0]
            weights = torch.softmax(logits, dim=0)
            return torch.einsum("f,bfmn->bmn", weights, task_tokens)

        logits = self.task_frequency_weight_logits[:frequency_len, :region_len]
        weights = torch.softmax(logits.reshape(-1), dim=0).reshape(
            frequency_len,
            region_len,
        )
        return torch.einsum("fr,bfrmn->bmn", weights, task_tokens)
