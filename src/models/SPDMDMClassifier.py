from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.models.SPDTransformer import SPDTransformer


MDMPoolingMode = Literal["mean", "weighted"]
BaselineMDMPoolingMode = Literal["original", "mean", "weighted"]


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

    def _prepare_inputs(
            self,
            pooled_log: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return pooled_log, prototypes

    def squared_distances(self, pooled_log: torch.Tensor) -> torch.Tensor:
        pooled_log, prototypes = self._prepare_inputs(pooled_log)
        diff = pooled_log.unsqueeze(-3) - prototypes
        return diff.square().sum(dim=(-2, -1))

    def forward(self, pooled_log: torch.Tensor) -> torch.Tensor:
        distances = self.squared_distances(pooled_log)
        scale = F.softplus(self.logit_scale_raw) + self.eps
        return -scale * distances

    def prototype_losses(
            self,
            pooled_log: torch.Tensor,
            targets: torch.Tensor,
            margin: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return intra-class distance and inter-class margin penalties."""
        if margin < 0:
            raise ValueError(f"prototype margin must be non-negative, got {margin}.")
        expected_target_shape = pooled_log.shape[:-2]
        if tuple(targets.shape) != tuple(expected_target_shape):
            raise ValueError(
                "Prototype targets must match pooled feature leading dimensions: "
                f"expected {tuple(expected_target_shape)}, got {tuple(targets.shape)}."
            )
        if targets.dtype != torch.long:
            targets = targets.long()
        if targets.numel() == 0:
            raise ValueError("Prototype losses require at least one target.")
        if int(targets.min()) < 0 or int(targets.max()) >= self.num_classes:
            raise ValueError(
                f"Prototype targets must be in [0, {self.num_classes - 1}]."
            )

        sample_distances = self.squared_distances(pooled_log)
        intra_loss = sample_distances.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).mean()

        _, prototypes = self._prepare_inputs(pooled_log)
        prototype_diff = (
            prototypes.unsqueeze(1) - prototypes.unsqueeze(0)
        )
        pairwise_squared_distance = prototype_diff.square().sum(dim=(-2, -1))
        pairwise_distance = torch.sqrt(
            pairwise_squared_distance.clamp_min(self.eps)
        )
        pair_mask = torch.triu(
            torch.ones(
                self.num_classes,
                self.num_classes,
                device=prototypes.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        inter_loss = F.relu(
            pairwise_distance.new_tensor(float(margin))
            - pairwise_distance[pair_mask]
        ).square().mean()
        return intra_loss, inter_loss


class LogEuclideanPrototypeClassifier(nn.Module):
    """Differentiable MDM classifier for fixed log-SPD input tokens."""

    def __init__(
            self,
            spd_dim: int,
            num_classes: int,
            token_shape: tuple[int, ...] = (),
            pooling: BaselineMDMPoolingMode = "mean",
            eps: float = 1e-6,
            prototype_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.spd_dim = int(spd_dim)
        self.num_classes = int(num_classes)
        self.token_shape = tuple(int(size) for size in token_shape)
        self.pooling = self._normalize_pooling(pooling)

        if self.spd_dim < 1:
            raise ValueError(f"spd_dim must be positive, got {self.spd_dim}.")
        if self.num_classes < 2:
            raise ValueError(
                f"num_classes must be >= 2, got {self.num_classes}."
            )
        if any(size < 1 for size in self.token_shape):
            raise ValueError(
                f"token_shape entries must be positive, got {self.token_shape}."
            )
        if self.pooling == "original" and self._token_count() != 1:
            raise ValueError(
                "pooling='original' requires exactly one log-SPD token, "
                f"got token_shape={self.token_shape}."
            )

        if self.pooling == "weighted" and self.token_shape:
            self.token_weight_logits = nn.Parameter(torch.zeros(self.token_shape))
        else:
            self.register_parameter("token_weight_logits", None)

        self.mdm_head = LogEuclideanMDMHead(
            spd_dim=self.spd_dim,
            num_classes=self.num_classes,
            eps=eps,
            prototype_init_std=prototype_init_std,
        )

    @staticmethod
    def _normalize_pooling(pooling: str) -> BaselineMDMPoolingMode:
        normalized = str(pooling).strip().lower().replace("-", "_")
        if normalized in {"original", "raw", "none", "identity", "trial", "single"}:
            return "original"
        if normalized in {"mean", "mdm_mean", "log_euclidean_mean"}:
            return "mean"
        if normalized in {"weighted", "weight", "mdm_weighted", "learned_weighted"}:
            return "weighted"
        raise ValueError(
            "LogEuclideanPrototypeClassifier pooling must be 'original', "
            f"'mean', or 'weighted', got {pooling!r}."
        )

    def _token_count(self) -> int:
        count = 1
        for size in self.token_shape:
            count *= size
        return count

    def token_weights(self) -> torch.Tensor | None:
        if self.token_weight_logits is None:
            return None
        return torch.softmax(self.token_weight_logits.reshape(-1), dim=0).reshape(
            self.token_shape
        )

    def pool_log_tokens(self, x_log: torch.Tensor) -> torch.Tensor:
        expected_shape = (*self.token_shape, self.spd_dim, self.spd_dim)
        if tuple(x_log.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected log-SPD input shape "
                f"(batch, {expected_shape}), got {tuple(x_log.shape)}."
            )

        x_log = 0.5 * (x_log + x_log.transpose(-1, -2))
        if not self.token_shape:
            return x_log
        if self.pooling == "original":
            return x_log.reshape(x_log.shape[0], self.spd_dim, self.spd_dim)

        token_dims = tuple(range(1, x_log.ndim - 2))
        if self.pooling == "mean":
            return x_log.mean(dim=token_dims)

        weights = self.token_weights()
        if weights is None:
            raise RuntimeError("Weighted pooling requires token_weight_logits.")
        view_shape = (1, *self.token_shape, 1, 1)
        return (x_log * weights.view(view_shape)).sum(dim=token_dims)

    def forward(self, x_log: torch.Tensor) -> torch.Tensor:
        return self.mdm_head(self.pool_log_tokens(x_log))


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
            brain_region_sequence_length: int = 1,
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
            learnable_metric_mode: Literal["full", "low-rank"] = "low-rank",
            learnable_metric_score: Literal["qgk", "distance"] = "qgk",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: str = "trace",
            share_metric_across_layers: bool = False,
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
            share_metric_across_layers=share_metric_across_layers,
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
        logits, aux, _pooled_log = self.forward_with_pooled(x, return_aux=return_aux)
        return logits, aux

    def forward_with_pooled(
            self,
            x: torch.Tensor,
            *,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
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
        return logits, aux, pooled_log

    def _mean_pool(self, x_log: torch.Tensor) -> torch.Tensor:
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
                "Weighted MDM pooling supports time, time/frequency, or "
                f"time/frequency/region tokens, got token shape {token_shape}."
            )

        max_shape = self.token_weight_logits.shape
        if any(size > max_size for size, max_size in zip(token_shape, max_shape)):
            raise ValueError(
                "Weighted MDM pooling was initialized for "
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

    def _mdm_logits(self, pooled_log: torch.Tensor) -> torch.Tensor:
        return self.mdm_head(pooled_log)
