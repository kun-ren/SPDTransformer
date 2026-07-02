from typing import Literal, Any

import torch
from torch import nn
import torch.nn.functional as F
from src.models.GeooptBiMap import GeooptBiMap

from src.models.SPDAttention import (
    SingleHeadAttention,
    _safe_eigh,
    spd_log,
)
from src.models.SPDFeedForward import SPDFeedForward
from src.models.TraceAddNorm import TraceAddNorm
from src.models.MultiHeadEncoder import MultiHeadEncoder

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
        print(f"input_dim: {spd_in_dim}, attention_dim: {attention_dim}")
        self.metric = metric
        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.debug_attention_shape = debug_attention_shape
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition

        self.stage_projection = None
        if self.stage_transition:
            self.stage_projection = GeooptBiMap(
                spd_in_dim,
                attention_dim,
                eps=eps,
            )

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
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=-3,
        )
        self.time_ffn = SPDFeedForward(
            spd_out_dim,
            hidden_spd_dim=ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.time_add_norm2 = TraceAddNorm(
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=-3,
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
            position_axis=-3,

        )
        self.frequency_ffn = SPDFeedForward(
            spd_out_dim,
            hidden_spd_dim=ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.frequency_add_norm2 = TraceAddNorm(
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=-3,
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
            num_heads: int,
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

        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        
        if num_heads == 1:
            self.layers = nn.ModuleList([SPDEncoder(
                spd_in_dim=dim if stage_transition else self.dims[0],
                attention_dim=self.dims[index+1],
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
            ) for index, dim in enumerate(self.dims[:-1])])
        elif num_heads > 1:
            self.layers = nn.ModuleList([MultiHeadEncoder(
                num_heads=num_heads,
                spd_in_dim=dim if stage_transition else self.dims[0],
                attention_dim=self.dims[index+1],
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
            ) for index, dim in enumerate(self.dims[:-1])])
        self.layers = nn.ModuleList([SPDEncoder(
                spd_in_dim=dim if stage_transition else self.dims[0],
                attention_dim=self.dims[index+1],
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
            ) for index, dim in enumerate(self.dims[:-1])])

    def forward(self, x: torch.Tensor):
        all_aux = {}
        for layer_index, layer in enumerate(self.layers):
            x, aux = layer(x)
            for name, param in aux.items():
                all_aux[name + "_" + str(layer_index)] = param

        return x, all_aux


