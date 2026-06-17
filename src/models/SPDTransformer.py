from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from src.models.BiMap import BiMap
from src.models.SPDAttention import (
    SingleHeadAttention,
    _safe_eigh,
    spd_exp,
    spd_log,
)


class RiemannianLayerNorm(nn.Module):
    """
    Per-sample Riemannian Layer Norm for SPD matrices.

    Normalizes in the full Sym(n) tangent space (Log-Euclidean),
    computing statistics within each sample independently.

    Input/output: (..., spd_dim, spd_dim)
    """

    def __init__(
            self,
            spd_dim: int,
            eps: float = 1e-5,
            affine: bool = True,
    ):
        super().__init__()
        self.spd_dim = spd_dim
        self.eps = eps
        self.affine = affine

        if affine:
            # γ: scalar — keeps symmetry under multiplication
            self.weight = nn.Parameter(torch.ones(()))
            # β₀: raw parameter, symmetrized on forward pass
            self.bias = nn.Parameter(torch.zeros(spd_dim, spd_dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.spd_dim, self.spd_dim):
            raise ValueError(
                f"Expected (..., {self.spd_dim}, {self.spd_dim}), got {tuple(x.shape)}"
            )

        S = spd_log(x)  # (..., n, n)

        # ── Per-sample scalar mean ────────────────────────────────────────
        mu = S.diagonal(dim1=-2, dim2=-1).mean(dim=-1)  # (...,)
        mu = mu[..., None, None]  # (..., 1, 1)
        eye = torch.eye(self.spd_dim, device=x.device, dtype=x.dtype)
        Z = S - mu * eye
        # tr(Z) = 0  →  det(exp(Z)) = 1

        # ── Per-sample scalar variance (Frobenius-based) ──────────────────
        # σ² = mean squared entry of Z  =  ‖Z‖²_F / n²
        var = Z.pow(2).mean(dim=(-2, -1), keepdim=True)  # (..., 1, 1)

        Z_hat = Z / (var + self.eps).sqrt()  # (..., n, n)

        # ── Affine transform ──────────────────────────────────────────────
        if self.affine:
            B = 0.5 * (self.bias + self.bias.transpose(-1, -2))  # symmetrize β
            Z_hat = self.weight * Z_hat + B

        # ── Map back to SPD manifold ──────────────────────────────────────
        return spd_exp(Z_hat)  # (..., n, n)


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
            spd_out_dim: int,
            residual_weight: float = 0.5,
            eps: float = 1e-5,
            affine: bool = True,
    ):
        super().__init__()
        if not 0.0 <= residual_weight <= 1.0:
            raise ValueError(
                f"residual_weight must be in [0, 1], got {residual_weight}."
            )

        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.residual_weight = residual_weight
        self.residual_projection = (
            nn.Identity()
            if spd_in_dim == spd_out_dim
            else BiMap(in_dim=spd_in_dim, out_dim=spd_out_dim)
        )
        self.norm = RiemannianLayerNorm(spd_out_dim, eps=eps, affine=affine)

    def forward(self, residual: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(residual)

        expected_shape = (self.spd_out_dim, self.spd_out_dim)
        if residual.shape[-2:] != expected_shape:
            raise ValueError(
                f"Expected residual shape (..., {self.spd_out_dim}, {self.spd_out_dim}), "
                f"got {tuple(residual.shape)}."
            )
        if sublayer_output.shape[-2:] != expected_shape:
            raise ValueError(
                f"Expected sublayer_output shape (..., {self.spd_out_dim}, {self.spd_out_dim}), "
                f"got {tuple(sublayer_output.shape)}."
            )

        alpha = self.residual_weight
        merged_log = (1.0 - alpha) * spd_log(residual) + alpha * spd_log(sublayer_output)
        merged = spd_exp(merged_log)
        return self.norm(merged)


class _LegacySPDActivation(nn.Module):
    """
    SPD-preserving activation via eigenvalue-domain nonlinearity.

    The matrix is decomposed as  X = U Λ Uᵀ,  the chosen function is
    applied element-wise to the eigenvalues Λ, and the matrix is
    reconstructed.  The result is always SPD as long as all activated
    eigenvalues are strictly positive.

    Modes
    -----
    'relu' → ReEig  : λᵢ ← max(ε, λᵢ)
    'gelu' → GeEig  : λᵢ ← GELU(λᵢ),  then global shift so min(λᵢ) ≥ ε

    Parameters
    ----------
    activation : 'relu' | 'gelu'
    eps        : lower bound / shift margin for eigenvalue positivity
    """

    def __init__(
        self,
        activation: Literal["relu", "gelu"] = "gelu",
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        self.activation = activation
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : Tensor  shape (..., n, n)

        Returns
        -------
        Y : Tensor  shape (..., n, n)  SPD
        """
        eigvals, eigvecs = _safe_eigh(X, eps=self.eps)        # (..., n), (..., n, n)

        if self.activation == "relu":
            eigvals = eigvals.clamp(min=self.eps)

        else:  # gelu
            eigvals = F.gelu(eigvals)
            min_val = eigvals.amin(dim=-1, keepdim=True)
            shift = (self.eps - min_val).clamp(min=0.0)        # 合并 where，等价且更简洁
            eigvals = eigvals + shift

        # Reconstruct:  U · diag(λ) · Uᵀ
        y = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-2, -1)
        return 0.5 * (y + y.transpose(-2, -1))

    def extra_repr(self) -> str:
        return f"activation={self.activation!r}, eps={self.eps}"



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
            eigvals = F.gelu(eigvals)
            min_val = eigvals.amin(dim=-1, keepdim=True)
            eigvals = eigvals + (self.eps - min_val).clamp_min(0.0)

        y = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        return 0.5 * (y + y.transpose(-1, -2))


class SPDFeedForward(nn.Module):
    """
    SPDNet-style feed-forward block.

    Classic SPDNet uses repeated BiMap + ReEig blocks while features stay on
    the SPD manifold. LogEig is usually used only at the final classifier head
    to move SPD features into Euclidean space.

    Here the Transformer block still needs SPD output, so the FFN is:
        BiMap -> ReEig -> BiMap -> ReEig
    """

    def __init__(
            self,
            spd_dim: int,
            hidden_spd_dim: int | None = None,
            eps: float = 1e-4,
    ):
        super().__init__()
        hidden_spd_dim = hidden_spd_dim or spd_dim

        self.spd_dim = spd_dim
        self.hidden_spd_dim = hidden_spd_dim
        self.ffn = nn.Sequential(
            BiMap(in_dim=spd_dim, out_dim=hidden_spd_dim),
            SPDActivation(eps=eps),
            BiMap(in_dim=hidden_spd_dim, out_dim=spd_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SPDEncoder(nn.Module):
    def __init__(
            self,
            spd_in_dim,
            spd_out_dim,
            ffn_hidden_spd_dim=None,
            metric='log-euclidean',
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
    ):
        super().__init__()
        self.metric = metric
        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.debug_attention_shape = debug_attention_shape

        self.time_attention = SingleHeadAttention(
            spd_in_dim,
            spd_out_dim,
            self.metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            metric_eps=metric_eps,
        )
        self.time_add_norm1 = SPDAddNorm(spd_in_dim, spd_out_dim)
        self.time_ffn = SPDFeedForward(spd_out_dim, ffn_hidden_spd_dim)
        self.time_add_norm2 = SPDAddNorm(spd_out_dim, spd_out_dim)

        self.frequency_attention = SingleHeadAttention(
            spd_out_dim,
            spd_out_dim,
            self.metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            metric_eps=metric_eps,
        )
        self.frequency_add_norm1 = SPDAddNorm(spd_out_dim, spd_out_dim)
        self.frequency_ffn = SPDFeedForward(spd_out_dim, ffn_hidden_spd_dim)
        self.frequency_add_norm2 = SPDAddNorm(spd_out_dim, spd_out_dim)

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

        y = attention(x)

        if axis != seq_pos:
            inverse_perm = [0] * len(perm)
            for new_axis, old_axis in enumerate(perm):
                inverse_perm[old_axis] = new_axis
            y = y.permute(inverse_perm)

        return y

    def forward(self, x):
        if x.ndim not in {4, 5}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency_bands, channels, channels), "
                f"got {tuple(x.shape)}."
            )

        self._print_attention_shape("time/input", x)
        time_output = self._apply_attention_along_axis(
            self.time_attention,
            x,
            axis=1,
        )
        self._print_attention_shape("time/output", time_output)
        x = self.time_add_norm1(x, time_output)
        x = self.time_add_norm2(x, self.time_ffn(x))

        # if (batch, time, frequency_bands, channels, channels)
        if x.ndim == 5:
            self._print_attention_shape("frequency/input", x)
            frequency_output = self._apply_attention_along_axis(
                self.frequency_attention,
                x,
                axis=2,
            )
            self._print_attention_shape("frequency/output", frequency_output)
            x = self.frequency_add_norm1(x, frequency_output)
            x = self.frequency_add_norm2(x, self.frequency_ffn(x))

        return x

    def _print_attention_shape(self, name: str, x: torch.Tensor) -> None:
        if self.debug_attention_shape:
            print(f"[SPDAttentionShape] {name}: shape={tuple(x.shape)}")


class SPDTransformer(nn.Module):
    """Stacked SPD Transformer encoder."""

    def __init__(
            self,
            spd_in_dim: int,
            spd_out_dim: int,
            depth: int = 1,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.depth = depth

        self.layers = nn.ModuleList([SPDEncoder(
                spd_in_dim=spd_in_dim,
                spd_out_dim=spd_out_dim,
                ffn_hidden_spd_dim=ffn_hidden_spd_dim,
                metric=metric,
                attention_dropout=attention_dropout,
                debug_attention_dropout=debug_attention_dropout,
                debug_attention_shape=debug_attention_shape,
                learnable_metric_mode=learnable_metric_mode,
                learnable_metric_rank=learnable_metric_rank,
                metric_eps=metric_eps,
            ) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
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
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            pooling: SPDPoolingMode = "attention",
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
    ):
        super().__init__()
        if pooling not in {"mean", "attention"}:
            raise ValueError(
                f"SPDPoolingClassifier pooling must be 'mean' or 'attention', got {pooling!r}."
            )

        self.spd_out_dim = spd_out_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.feature_dim = spd_out_dim * (spd_out_dim + 1) // 2

        self.encoder = SPDTransformer(
            spd_in_dim=spd_in_dim,
            spd_out_dim=spd_out_dim,
            depth=depth,
            ffn_hidden_spd_dim=ffn_hidden_spd_dim,
            metric=metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            debug_attention_shape=debug_attention_shape,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            metric_eps=metric_eps,
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
        x = self.encoder(x)

        if self.pooling == "mean":
            pooled_log = self._mean_pool(x)
        else:
            pooled_log = self._attention_pool(x)

        features = self.upper_triangular_vectorize(pooled_log)
        return self.classifier(features)

    def _mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean value across all tokens that belong to one trial
        :return torch.Tensor: (batch, channels, channels)
        """
        log_x = spd_log(x)
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
        # log_tokens = (batch, tim x frequency_bands, channels, channels)
        log_tokens = log_x.reshape(batch_size, -1, spd_dim, spd_dim)
        token_features = self.upper_triangular_vectorize(log_tokens)

        scores = self.pool_score(token_features).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
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
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
    ):
        super().__init__()
        self.spd_in_dim = spd_in_dim
        self.spd_out_dim = spd_out_dim
        self.num_classes = num_classes
        self.feature_dim = spd_out_dim * (spd_out_dim + 1) // 2

        self.task_log_token = nn.Parameter(torch.zeros(spd_in_dim, spd_in_dim))
        self.encoder = SPDTransformer(
            spd_in_dim=spd_in_dim,
            spd_out_dim=spd_out_dim,
            depth=depth,
            ffn_hidden_spd_dim=ffn_hidden_spd_dim,
            metric=metric,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            debug_attention_shape=debug_attention_shape,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            metric_eps=metric_eps,
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
        return self.classifier(features)

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

        task_log = 0.5 * (self.task_log_token + self.task_log_token.transpose(-1, -2))
        task_token = spd_exp(task_log).to(device=x.device, dtype=x.dtype)

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
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            classifier_type: ClassifierType = "pooling",
            pooling: SPDPoolingMode = "attention",
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            metric_eps: float = 1e-6,
    ):
        super().__init__()
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
                ffn_hidden_spd_dim=ffn_hidden_spd_dim,
                metric=metric,
                depth=depth,
                pooling=pooling,
                dropout=dropout,
                attention_dropout=attention_dropout,
                debug_attention_dropout=debug_attention_dropout,
                debug_attention_shape=debug_attention_shape,
                learnable_metric_mode=learnable_metric_mode,
                learnable_metric_rank=learnable_metric_rank,
                metric_eps=metric_eps,
            )
        else:
            self.model = SPDTaskTagClassifier(
                spd_in_dim=spd_in_dim,
                spd_out_dim=spd_out_dim,
                num_classes=num_classes,
                ffn_hidden_spd_dim=ffn_hidden_spd_dim,
                metric=metric,
                depth=depth,
                dropout=dropout,
                attention_dropout=attention_dropout,
                debug_attention_dropout=debug_attention_dropout,
                debug_attention_shape=debug_attention_shape,
                learnable_metric_mode=learnable_metric_mode,
                learnable_metric_rank=learnable_metric_rank,
                metric_eps=metric_eps,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
