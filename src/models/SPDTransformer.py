from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from src.models.BiMap import BiMap
from src.models.RiemannianLayerNorm import RiemannianLayerNorm
from src.models.SPDAttention import (
    SingleHeadAttention,
    _safe_eigh,
    maybe_check_tensor,
    spd_log,
)


class SPDAddNorm(nn.Module):
    """
    SPD Add & Norm block using a Log-Euclidean residual connection.

    The residual merge is a two-point Log-Euclidean barycenter:
        merged = exp((1 - alpha) log(residual) + alpha log(sublayer_output))
    followed by Riemannian layer normalization.
    """

    def __init__(
            self,
            spd_in_dim: int,
            sequence_length: int,
            tau: float = 1.0,
            eps: float = 1e-5,
            affine: bool = True,
    ):
        super().__init__()

        self.spd_in_dim = spd_in_dim
        self.residual_weight = nn.Parameter(
            torch.tensor(-2.0)
        )

        self.norm = RiemannianLayerNorm(
            spd_dim=spd_in_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            preserve_log_mean=False,
        )

    def forward(self, residual_log: torch.Tensor, sublayer_output_log: torch.Tensor) -> torch.Tensor:

        # Constrain the residual scale to (0, 1).
        eta = torch.sigmoid(self.residual_weight)

        S_res = (
                residual_log
                + eta * sublayer_output_log
        )

        # Protect against small floating-point asymmetry.
        S_res = 0.5 * (
                S_res + S_res.transpose(-1, -2)
        )

        output_log = self.norm(S_res)
        return output_log

class SPDActivation(nn.Module):
    """SPD-safe activation applied in the eigenvalue domain."""

    def __init__(
        self,
        activation: Literal["relu", "gelu"] = "gelu",
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        self.activation = activation
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 0.5 * (x + x.transpose(-1, -2))
        eigvals, eigvecs = _safe_eigh(x, eps=self.eps)

        if self.activation == "relu":
            eigvals = eigvals.clamp_min(self.eps)
        else:
            eigvals = F.gelu(eigvals).clamp_min(self.eps)

        y = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        return 0.5 * (y + y.transpose(-1, -2))

class SPDFeedForward(nn.Module):
    """
    Log-space feed-forward block for SPD Transformer.

    Input:
        x_log: (..., spd_dim, spd_dim)
               already computed matrix logarithm of SPD matrix.
               It is symmetric but not necessarily positive definite.

    Pipeline:
        x_log
        -> upper-triangular vectorization
        -> ordinary Linear FFN
        -> reconstruct symmetric log matrix
        -> matrix exponential
        -> SPD output

    Output:
        out: (..., spd_dim, spd_dim), SPD matrix
    """

    def __init__(
            self,
            spd_dim: int,
            hidden_spd_dim: int | None = None,
            eps: float = 1e-4,
            debug_tensor_stats: bool = False,
    ):
        super().__init__()

        self.spd_dim = spd_dim
        self.eps = eps
        self.debug_tensor_stats = debug_tensor_stats

        # Number of unique entries in a symmetric matrix
        self.feature_dim = spd_dim * (spd_dim + 1) // 2

        # Here hidden_spd_dim is treated as hidden feature dimension.
        # If None, use standard Transformer-style expansion.
        hidden_feature_dim = hidden_spd_dim or 4 * self.feature_dim

        row, col = torch.triu_indices(spd_dim, spd_dim)
        self.register_buffer("tri_row", row, persistent=False)
        self.register_buffer("tri_col", col, persistent=False)

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, self.feature_dim),
        )

    def forward(self, x_log: torch.Tensor) -> torch.Tensor:
        """
        x_log: (..., spd_dim, spd_dim), already in log/tangent space.
        return: (..., spd_dim, spd_dim), log space matrix.
        """

        if x_log.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected x_log shape (..., {self.spd_dim}, {self.spd_dim}), "
                f"got {tuple(x_log.shape)}."
            )

        # Make sure the tangent matrix is symmetric
        x_log = 0.5 * (x_log + x_log.transpose(-1, -2))

        # (..., C, C) -> (..., C * (C + 1) // 2)
        x_vec = self._upper_triangular_vectorize(x_log)


        # ordinary Euclidean FFN
        out_vec = self.ffn(x_vec)


        # (..., D) -> (..., C, C), symmetric log matrix
        out_log = self._upper_triangular_unvectorize(out_vec)

        return out_log

    def _upper_triangular_vectorize(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.tri_row, self.tri_col]

    def _upper_triangular_unvectorize(self, x_vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(
            *x_vec.shape[:-1],
            self.spd_dim,
            self.spd_dim,
            device=x_vec.device,
            dtype=x_vec.dtype,
        )

        out[..., self.tri_row, self.tri_col] = x_vec
        out[..., self.tri_col, self.tri_row] = x_vec

        return 0.5 * (out + out.transpose(-1, -2))


class SPDEncoder(nn.Module):
    def __init__(
            self,
            spd_in_dim,
            spd_out_dim,
            time_sequence_length,
            frequency_sequence_length,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric='log-euclidean',
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
    ):
        super().__init__()
        self.metric = metric
        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.debug_attention_shape = debug_attention_shape
        self.debug_tensor_stats = debug_tensor_stats

        self.time_attention = SingleHeadAttention(
            spd_in_dim,
            spd_out_dim,
            self.metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            metric_eps=metric_eps,
            use_position=use_position_bias,
            max_position=time_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )
        self.time_add_norm1 = SPDAddNorm(
            spd_out_dim,
            sequence_length=time_sequence_length,
            tau=tau,
            eps=metric_eps,
            affine=layer_norm_affine,
        )
        self.time_ffn = SPDFeedForward(
            spd_out_dim,
            ffn_hidden_spd_dim,
        )
        self.time_add_norm2 = SPDAddNorm(
            spd_out_dim,
            sequence_length=time_sequence_length,
            tau=tau,
            eps=metric_eps,
            affine=layer_norm_affine,
        )

        self.frequency_attention = SingleHeadAttention(
            spd_out_dim,
            spd_out_dim,
            self.metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            metric_eps=metric_eps,
            use_position=use_position_bias,
            max_position=frequency_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )
        self.frequency_add_norm1 = SPDAddNorm(
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=metric_eps,
            affine=layer_norm_affine,

        )
        self.frequency_ffn = SPDFeedForward(
            spd_out_dim,
            ffn_hidden_spd_dim,
        )
        self.frequency_add_norm2 = SPDAddNorm(
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=metric_eps,
            affine=layer_norm_affine,
        )

        self.attention = self.time_attention

    @staticmethod
    def _apply_attention_along_axis(
            attention: SingleHeadAttention,
            x: torch.Tensor,
            axis: int,
    ) -> torch.Tensor:
        if x.ndim < 4:
            raise ValueError(
                "Expected SPD input with shape (..., sequence, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        leading_ndim = x.ndim - 2
        if axis < 0:
            axis += leading_ndim
        if not 0 <= axis < leading_ndim:
            raise ValueError(
                f"axis must refer to one of the {leading_ndim} leading dimensions, "
                f"got axis={axis}."
            )

        seq_pos = leading_ndim - 1
        perm = list(range(x.ndim))
        if axis != seq_pos:
            moved_axis = perm.pop(axis)
            perm.insert(seq_pos, moved_axis)
            x = x.permute(perm)

        y_log = attention(x)

        if axis != seq_pos:
            inverse_perm = [0] * len(perm)
            for new_axis, old_axis in enumerate(perm):
                inverse_perm[old_axis] = new_axis
            y_log = y_log.permute(inverse_perm)

        return y_log

    def forward(self, x):
        if x.ndim not in {4, 5}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency_bands, channels, channels), "
                f"got {tuple(x.shape)}."
            )
        time_output_log = self._apply_attention_along_axis(
            self.time_attention,
            x,
            axis=1,
        )
        x_log = self.time_add_norm1(spd_log(x), time_output_log)

        x_log = self.time_add_norm2(x_log, self.time_ffn(x_log))


        # if (batch, time, frequency_bands, channels, channels)
        if x.ndim == 5:
            x_spd = torch.matrix_exp(
                0.5 * (x_log + x_log.transpose(-1, -2))
            )
            frequency_output_log = self._apply_attention_along_axis(
                self.frequency_attention,
                x_spd,
                axis=2,
            )

            x_log = self.frequency_add_norm1(x_log, frequency_output_log)

            x_log = self.frequency_add_norm2(x_log, self.frequency_ffn(x_log))

        x_log = 0.5 * (x_log + x_log.transpose(-1, -2))
        x_spd = torch.matrix_exp(x_log)
        return 0.5 * (x_spd + x_spd.transpose(-1, -2))


class SPDTransformer(nn.Module):
    """Stacked SPD Transformer encoder."""

    def __init__(
            self,
            spd_in_dim: int,
            spd_out_dim: int,
            time_sequence_length,
            frequency_sequence_length,
            tau=1.0,
            depth: int = 1,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.depth = depth
        self.debug_tensor_stats = debug_tensor_stats

        self.layers = nn.ModuleList([SPDEncoder(
                spd_in_dim=spd_in_dim,
                spd_out_dim=spd_out_dim,
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
                metric_eps=metric_eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
            ) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_index, layer in enumerate(self.layers):
            x = layer(x)
        return x


ClassifierType = Literal["pooling", "task"]
SPDPoolingMode = Literal["mean", "attention"]


class SPDClassifierBase(nn.Module):
    @staticmethod
    def upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
        spd_dim = x.shape[-1]
        row, col = torch.triu_indices(spd_dim, spd_dim, device=x.device)
        return x[..., row, col]

    @staticmethod
    def build_linear_classifier(
            feature_dim: int,
            num_classes: int,
            dropout: float,
    ) -> nn.Module:
        return nn.Sequential(
            #nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )


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
            spd_out_dim: int,
            num_classes: int,
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
            metric_eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
    ):
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(
                f"SPDPoolingClassifier pooling must be 'mean' or 'attention', got {pooling!r}."
            )

        self.spd_out_dim = spd_out_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.debug_tensor_stats = debug_tensor_stats
        self.feature_dim = spd_out_dim * (spd_out_dim + 1) // 2

        self.encoder = SPDTransformer(
            spd_in_dim=spd_in_dim,
            spd_out_dim=spd_out_dim,
            depth=depth,
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
            metric_eps=metric_eps,
            use_position_bias=use_position_bias,
            layer_norm_affine=layer_norm_affine,
        )

        if pooling == "attention":
            self.pool_score = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 1),
            )
        else:
            self.pool_score = None

        self.classifier = self.build_linear_classifier(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_check_tensor(self.debug_tensor_stats, "pooling_classifier/input", x)
        x = self.encoder(x)
        maybe_check_tensor(self.debug_tensor_stats, "pooling_classifier/encoded", x)

        if self.pooling == "mean":
            pooled_log = self._mean_pool(x)
        else:
            pooled_log = self._attention_pool(x)
        maybe_check_tensor(self.debug_tensor_stats, "pooling_classifier/pooled_log", pooled_log)

        features = self.upper_triangular_vectorize(pooled_log)
        maybe_check_tensor(self.debug_tensor_stats, "pooling_classifier/features", features)
        logits = self.classifier(features)
        maybe_check_tensor(self.debug_tensor_stats, "pooling_classifier/logits", logits)
        return logits

    def _mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean value across all tokens that belong to one trial
        :return torch.Tensor: (batch, channels, channels)
        """
        log_x = spd_log(x)
        maybe_check_tensor(self.debug_tensor_stats, "mean_pool/log_x", log_x)
        token_dims = tuple(range(1, log_x.ndim - 2))
        return log_x.mean(dim=token_dims)

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return: pooled spd matrix, (batch, channels, channels)
        """
        batch_size = x.shape[0]
        spd_dim = x.shape[-1]
        log_x = spd_log(x)
        maybe_check_tensor(self.debug_tensor_stats, "attention_pool/log_x", log_x)
        # log_tokens = (batch, tim x frequency_bands, channels, channels)
        log_tokens = log_x.reshape(batch_size, -1, spd_dim, spd_dim)
        maybe_check_tensor(self.debug_tensor_stats, "attention_pool/log_tokens", log_tokens)
        token_features = self.upper_triangular_vectorize(log_tokens)
        maybe_check_tensor(self.debug_tensor_stats, "attention_pool/token_features", token_features)

        scores = self.pool_score(token_features).squeeze(-1)
        maybe_check_tensor(self.debug_tensor_stats, "attention_pool/scores", scores)
        weights = torch.softmax(scores, dim=-1)
        maybe_check_tensor(self.debug_tensor_stats, "attention_pool/weights", weights)
        return torch.einsum("bt,btmn->bmn", weights, log_tokens)


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
            metric_eps: float = 1e-6,
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
            metric_eps=metric_eps,
            use_position_bias=use_position_bias,
            layer_norm_affine=layer_norm_affine,
        )
        self.classifier = self.build_linear_classifier(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/input", x)
        x = self._prepend_task_token(x)
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/with_task_token", x)
        x = self.encoder(x)
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/encoded", x)
        task_log = self._extract_task_log_feature(x)
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/task_log", task_log)
        features = self.upper_triangular_vectorize(task_log)
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/features", features)
        logits = self.classifier(features)
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/logits", logits)
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
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/task_log_parameter", task_log)

        # matrix_exp remains differentiable when the log token has repeated
        # eigenvalues, unlike an eigendecomposition-based implementation.
        task_token = torch.matrix_exp(task_log)  # Identity  matrix I
        eye = torch.eye(self.spd_in_dim, device=x.device, dtype=x.dtype)
        task_token = 0.5 * (task_token + task_token.transpose(-1, -2)) + 1e-5 * eye # sym
        maybe_check_tensor(self.debug_tensor_stats, "task_classifier/task_spd_token", task_token)

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


class SPDTransformerClassifier(nn.Module):
    """
    Selects the trial-level classifier style.

    classifier_type="pooling":
        no task tag; use mean or attention pooling over encoder tokens.

    classifier_type="task":
        insert SPD [TASK] token; classify from task-token output.
    """

    def __init__(
            self,
            spd_in_dim: int,
            spd_out_dim: int,
            num_classes: int,
            time_sequence_length,
            frequency_sequence_length,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            classifier_type: ClassifierType = "pooling",
            pooling: SPDPoolingMode = "attention",
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
    ):
        super().__init__()
        self.debug_tensor_stats = debug_tensor_stats
        if pooling == "task":
            classifier_type = "task"
            pooling = "attention"

        if classifier_type not in {"pooling", "task"}:
            raise ValueError(
                "classifier_type must be 'pooling' or 'task', "
                f"got {classifier_type!r}."
            )

        self.classifier_type = classifier_type
        if classifier_type == "pooling":
            self.model = SPDPoolingClassifier(
                spd_in_dim=spd_in_dim,
                spd_out_dim=spd_out_dim,
                num_classes=num_classes,
                time_sequence_length=time_sequence_length,
                frequency_sequence_length=frequency_sequence_length,
                tau=tau,
                ffn_hidden_spd_dim=ffn_hidden_spd_dim,
                metric=metric,
                depth=depth,
                pooling=pooling,
                dropout=dropout,
                attention_dropout=attention_dropout,
                debug_attention_dropout=debug_attention_dropout,
                debug_attention_shape=debug_attention_shape,
                debug_tensor_stats=debug_tensor_stats,
                learnable_metric_mode=learnable_metric_mode,
                learnable_metric_rank=learnable_metric_rank,
                metric_eps=metric_eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
            )
        else:
            print("initializing SPDTaskTagClassifier")
            self.model = SPDTaskTagClassifier(
                spd_in_dim=spd_in_dim,
                spd_out_dim=spd_out_dim,
                num_classes=num_classes,
                time_sequence_length=time_sequence_length,
                frequency_sequence_length=frequency_sequence_length,
                tau=tau,
                ffn_hidden_spd_dim=ffn_hidden_spd_dim,
                metric=metric,
                depth=depth,
                dropout=dropout,
                attention_dropout=attention_dropout,
                debug_attention_dropout=debug_attention_dropout,
                debug_attention_shape=debug_attention_shape,
                debug_tensor_stats=debug_tensor_stats,
                learnable_metric_mode=learnable_metric_mode,
                learnable_metric_rank=learnable_metric_rank,
                metric_eps=metric_eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
            )
            print("SPDTaskTagClassifier built")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_check_tensor(self.debug_tensor_stats, "classifier/input", x)
        logits = self.model(x)
        maybe_check_tensor(self.debug_tensor_stats, "classifier/output", logits)
        return logits
