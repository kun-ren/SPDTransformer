from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.models.SPDTransformer import SPDTransformer


MDMPoolingMode = Literal["mean", "weighted"]


class LogEuclideanMDMHead(nn.Module):
    """Differentiable log-Euclidean MDM head with learnable class prototypes."""

    def __init__(
            self,
            spd_dim: int,
            num_classes: int,
            eps: float = 1e-6,
            prototype_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if spd_dim < 1:
            raise ValueError(f"spd_dim must be positive, got {spd_dim}.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}.")

        self.spd_dim = spd_dim
        self.num_classes = num_classes
        self.eps = eps

        prototypes = torch.empty(num_classes, spd_dim, spd_dim)
        nn.init.normal_(prototypes, mean=0.0, std=prototype_init_std)
        prototypes = 0.5 * (prototypes + prototypes.transpose(-1, -2))
        self.class_log_prototypes = nn.Parameter(prototypes)
        self.logit_scale_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, pooled_log: torch.Tensor) -> torch.Tensor:
        if pooled_log.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected log-SPD matrix shape (..., {self.spd_dim}, "
                f"{self.spd_dim}), got {tuple(pooled_log.shape)}."
            )

        pooled_log = 0.5 * (pooled_log + pooled_log.transpose(-1, -2))
        prototypes = 0.5 * (
            self.class_log_prototypes
            + self.class_log_prototypes.transpose(-1, -2)
        )
        diff = pooled_log.unsqueeze(-3) - prototypes
        distances = diff.square().sum(dim=(-2, -1))
        scale = F.softplus(self.logit_scale_raw) + self.eps
        return -scale * distances


class SPDMDMClassifier(nn.Module):
    """
    Log-Euclidean MDM classifier on top of the SPDTransformer encoder.

    The encoder returns log-domain SPD tokens. This head pools all
    time/frequency tokens into one symmetric log-domain matrix and classifies
    by distance to learnable class prototypes in the same log domain.
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
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            pooling: MDMPoolingMode = "mean",
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
        self.eps = eps

        self.encoder = SPDTransformer(
            num_heads=num_heads,
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
            stage_projection_init=stage_projection_init,
            add_norm_type=add_norm_type,
        )

        if pooling == "weighted":
            self.token_weight_logits = nn.Parameter(
                torch.zeros(time_sequence_length, frequency_sequence_length)
            )
        else:
            self.register_parameter("token_weight_logits", None)

        self.mdm_head = LogEuclideanMDMHead(
            spd_dim=self.transformer_out_dim,
            num_classes=num_classes,
            eps=eps,
        )

    @staticmethod
    def _normalize_pooling(pooling: str) -> MDMPoolingMode:
        normalized = str(pooling).strip().lower().replace("-", "_")
        if normalized in {"mean", "mdm_mean"}:
            return "mean"
        if normalized in {"weighted", "weight", "mdm_weighted", "learned_weighted"}:
            return "weighted"
        raise ValueError(
            "SPDMDMClassifier pooling must be 'mean' or 'weighted', "
            f"got {pooling!r}."
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

        logits = self._mdm_logits(pooled_log)
        return logits, aux

    def _mean_pool(self, x_log: torch.Tensor) -> torch.Tensor:
        token_dims = tuple(range(1, x_log.ndim - 2))
        return x_log.mean(dim=token_dims)

    def _weighted_pool(self, x_log: torch.Tensor) -> torch.Tensor:
        if self.token_weight_logits is None:
            raise RuntimeError("token_weight_logits is only defined for weighted pooling.")

        if x_log.ndim == 4:
            time_len = x_log.shape[1]
            self._check_weight_shape(time_len, 1)
            logits = self.token_weight_logits[:time_len, 0]
            weights = torch.softmax(logits, dim=0)
            return torch.einsum("t,btmn->bmn", weights, x_log)

        if x_log.ndim == 5:
            time_len, frequency_len = x_log.shape[1], x_log.shape[2]
            self._check_weight_shape(time_len, frequency_len)
            logits = self.token_weight_logits[:time_len, :frequency_len]
            weights = torch.softmax(logits.reshape(-1), dim=0).reshape(
                time_len,
                frequency_len,
            )

            return torch.einsum("tf,btfmn->bmn", weights, x_log)

        raise ValueError(
            "Expected encoder output shape "
            "(batch, time, channels, channels) or "
            "(batch, time, frequency, channels, channels), "
            f"got {tuple(x_log.shape)}."
            )


    def _check_weight_shape(self, time_len: int, frequency_len: int) -> None:
        assert self.token_weight_logits is not None
        if (
                time_len > self.token_weight_logits.shape[0]
                or frequency_len > self.token_weight_logits.shape[1]
        ):
            raise ValueError(
                "Weighted MDM pooling was initialized for "
                f"{tuple(self.token_weight_logits.shape)} time/frequency tokens, "
                f"but encoder output has {(time_len, frequency_len)}."
            )

    def _mdm_logits(self, pooled_log: torch.Tensor) -> torch.Tensor:
        return self.mdm_head(pooled_log)
