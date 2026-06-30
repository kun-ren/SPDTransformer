from typing import Literal, Any

import torch
from torch import nn
import torch.nn.functional as F
from src.models.BiMap import BiMap
from src.models.RiemannianLayerNorm import RiemannianLayerNorm
from src.models.SPDAttention import (
    SingleHeadAttention,
    _safe_eigh,
    spd_log,
)
from src.models.SPDPoolingClassifier import SPDPoolingClassifier, SPDPoolingMode
from src.models.SPDTaskTagClassifier import SPDTaskTagClassifier
from src.models.TraceAddNorm import TraceAddNorm


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

        #Constrain the residual scale to (0, 1).
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
            dropout: float = 0.0,
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
        hidden_feature_dim = hidden_spd_dim or 2 * self.feature_dim

        row, col = torch.triu_indices(spd_dim, spd_dim)
        self.register_buffer("tri_row", row, persistent=False)
        self.register_buffer("tri_col", col, persistent=False)

        self.ffn = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feature_dim, self.feature_dim),
            nn.Dropout(dropout),
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
            attention_dim,
            time_sequence_length,
            frequency_sequence_length,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            stage_transition=True,
            metric='log-euclidean',
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["low-rank", "kronecker"] = "low-rank",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-6,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.metric = metric
        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.debug_attention_shape = debug_attention_shape
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition

        self.stage_projection = None
        if self.stage_transition:
            self.stage_projection = BiMap(spd_in_dim, attention_dim)

        if self.stage_projection is not None:
            spd_out_dim = attention_dim
        else:
            spd_out_dim = spd_in_dim

        self.time_attention = SingleHeadAttention(
            spd_out_dim,
            attention_dim,
            self.metric,
            stage_transition=self.stage_transition,
            attention_dropout=attention_dropout,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position=use_position_bias,
            max_position=time_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )

        self.time_add_norm1 = TraceAddNorm(
            spd_out_dim,
            sequence_length=time_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
        )
        self.time_ffn = SPDFeedForward(
            spd_out_dim,
            ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.time_add_norm2 = TraceAddNorm(
            spd_out_dim,
            sequence_length=time_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
        )

        self.frequency_attention = SingleHeadAttention(
            spd_out_dim,
            attention_dim,
            self.metric,
            attention_dropout=attention_dropout,
            stage_transition=self.stage_transition,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position=use_position_bias,
            max_position=frequency_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )
        self.frequency_add_norm1 = TraceAddNorm(
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,

        )
        self.frequency_ffn = SPDFeedForward(
            spd_out_dim,
            ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.frequency_add_norm2 = TraceAddNorm(
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
        )

        self.attention = self.time_attention

    @staticmethod
    def _apply_attention_along_axis(
            attention: SingleHeadAttention,
            x: torch.Tensor,
            axis: int,
    ) -> tuple[Any, Any]:
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

        y_log, aux = attention(x)

        if axis != seq_pos:
            inverse_perm = [0] * len(perm)
            for new_axis, old_axis in enumerate(perm):
                inverse_perm[old_axis] = new_axis
            y_log = y_log.permute(inverse_perm)

        return y_log, aux

    def forward(self, x):
        if x.ndim not in {4, 5}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency_bands, channels, channels), "
                f"got {tuple(x.shape)}."
            )
        all_aux = {}
        # first above all
        if self.stage_transition:
            x = self.stage_projection(x)
            all_aux["P_x"] = x

        time_output_log, aux = self._apply_attention_along_axis(
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
            frequency_output_log, aux = self._apply_attention_along_axis(
                self.frequency_attention,
                x_spd,
                axis=2,
            )

            x_log = self.frequency_add_norm1(x_log, frequency_output_log)

            x_log = self.frequency_add_norm2(x_log, self.frequency_ffn(x_log))

        x_log = 0.5 * (x_log + x_log.transpose(-1, -2))
        x_spd = torch.matrix_exp(x_log)

        all_aux.update(aux)
        return 0.5 * (x_spd + x_spd.transpose(-1, -2)), all_aux


class SPDTransformer(nn.Module):
    """Stacked SPD Transformer encoder."""

    def __init__(
            self,
            spd_in_dim: int,
            attention_dim: int,
            time_sequence_length,
            stage_transition: True,
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
            eps: float = 1e-8,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.depth = depth
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition

        base_step, remainder = divmod(attention_dim - spd_in_dim, depth)

        result = [spd_in_dim, spd_in_dim + base_step + remainder]
        for _ in range(depth - 1):
            result.append(result[-1] + base_step)

        self.dims = result

        self.layers = nn.ModuleList([SPDEncoder(
                spd_in_dim=self.dims[index-1],
                attention_dim=dim,
                stage_transition=self.stage_transition,
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
            ) for index, dim in enumerate(self.dims)])

    def forward(self, x: torch.Tensor):
        all_aux = {}
        for layer_index, layer in enumerate(self.layers):
            x, aux = layer(x)
            for name, param in aux:
                all_aux[name + "_" + layer_index] = param

        return x, all_aux


ClassifierType = Literal["pooling", "task"]


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
            attention_dim: int,
            num_classes: int,
            stage_transition: True,
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
            eps: float = 1e-8,
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
                attention_dim=attention_dim,
                num_classes=num_classes,
                stage_transition=stage_transition,
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
                eps=eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
            )
        else:
            print("initializing SPDTaskTagClassifier")
            self.model = SPDTaskTagClassifier(
                spd_in_dim=spd_in_dim,
                attention_dim=attention_dim,
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
                eps=eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
            )
            print("SPDTaskTagClassifier built")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.model(x)
        return logits
