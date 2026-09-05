from typing import Literal

import torch
from torch import nn
from src.models.AxisMetricSharing import share_axis_metrics
from src.models.GeooptBiMap import GeooptBiMap
from src.models.MultiHeadEncoder import (
    AddNormType,
    SPDMultiHeadEncoder,
    _make_add_norm,
    _symmetrize,
)

from src.models.SPDAttention import (
    SingleHeadAttention,
    spd_log,
)
from src.models.SPDFeedForward import SPDFeedForward


class SPDEncoder(nn.Module):
    def __init__(
            self,
            spd_in_dim,
            attention_dim,
            time_sequence_length,
            frequency_sequence_length,
            brain_region_sequence_length=1,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            stage_transition=True,
            metric='log-euclidean',
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
            dropout: float = 0.0,
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: AddNormType = "trace",
    ):
        super().__init__()
        print("init single head encoder")
        print(f"input_dim: {spd_in_dim}, attention_dim: {attention_dim}")
        self.metric = metric
        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.debug_attention_shape = debug_attention_shape
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition
        self.eps = eps

        self.stage_projection = None
        if self.stage_transition and attention_dim != spd_in_dim:
            self.stage_projection = GeooptBiMap(
                spd_in_dim,
                attention_dim,
                eps=eps,
                init=stage_projection_init,
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
            learnable_metric_score=learnable_metric_score,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position=use_position_bias,
            max_position=time_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )

        self.time_add_norm1 = _make_add_norm(
            add_norm_type,
            spd_out_dim,
            sequence_length=time_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=1,
        )
        self.time_ffn = SPDFeedForward(
            spd_out_dim,
            hidden_spd_dim=ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.time_add_norm2 = _make_add_norm(
            add_norm_type,
            spd_out_dim,
            sequence_length=time_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=1,
        )

        self.frequency_attention = SingleHeadAttention(
            spd_out_dim,
            attention_dim,
            self.metric,
            attention_dropout=attention_dropout,
            stage_transition=self.stage_transition,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_score=learnable_metric_score,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position=use_position_bias,
            max_position=frequency_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )
        self.frequency_add_norm1 = _make_add_norm(
            add_norm_type,
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=2,

        )
        self.frequency_ffn = SPDFeedForward(
            spd_out_dim,
            hidden_spd_dim=ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.frequency_add_norm2 = _make_add_norm(
            add_norm_type,
            spd_out_dim,
            sequence_length=frequency_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=2,
        )

        self.region_attention = SingleHeadAttention(
            spd_out_dim,
            attention_dim,
            self.metric,
            attention_dropout=attention_dropout,
            stage_transition=self.stage_transition,
            debug_attention_dropout=debug_attention_dropout,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_score=learnable_metric_score,
            learnable_metric_rank=learnable_metric_rank,
            eps=eps,
            use_position=use_position_bias,
            max_position=brain_region_sequence_length,
            debug_tensor_stats=debug_tensor_stats,
        )
        self.region_add_norm1 = _make_add_norm(
            add_norm_type,
            spd_out_dim,
            sequence_length=brain_region_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=3,
        )
        self.region_ffn = SPDFeedForward(
            spd_out_dim,
            hidden_spd_dim=ffn_hidden_spd_dim,
            dropout=dropout,
            eps=eps
        )
        self.region_add_norm2 = _make_add_norm(
            add_norm_type,
            spd_out_dim,
            sequence_length=brain_region_sequence_length,
            tau=tau,
            eps=eps,
            affine=layer_norm_affine,
            position_axis=3,
        )

        self.attention = self.time_attention


    @staticmethod
    def _apply_attention_along_axis(
            attention: SingleHeadAttention,
            x: torch.Tensor,
            axis: int,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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

        y_log, aux = attention(x, return_aux=return_aux)

        if axis != seq_pos:
            inverse_perm = [0] * len(perm)
            for new_axis, old_axis in enumerate(perm):
                inverse_perm[old_axis] = new_axis
            y_log = y_log.permute(inverse_perm).contiguous()

        return y_log, aux

    def forward(
            self,
            x,
            return_log: bool = False,
            return_aux: bool = True,
    ):
        if x.ndim not in {4, 5, 6}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency, channels, channels) or "
                "(batch, time, frequency, brain_region, channels, channels), "
                f"got {tuple(x.shape)}."
            )
        all_aux: dict[str, torch.Tensor] = {}
        if self.stage_projection is not None:
            attention_input = self.stage_projection(x)
            if return_aux:
                all_aux["P_x"] = attention_input
        else:
            attention_input = x

        residual_log = spd_log(attention_input)
        time_output_log, aux = self._apply_attention_along_axis(
            self.time_attention,
            attention_input,
            axis=1,
            return_aux=return_aux,
        )
        if return_aux:
            all_aux.update(
                {f"axis_1_head_0_{name}": param for name, param in aux.items()}
            )
        x_log = self.time_add_norm1(residual_log, time_output_log)
        x_log = self.time_add_norm2(x_log, self.time_ffn(x_log))

        if attention_input.ndim >= 5 and attention_input.shape[2] > 1:
            x_spd = torch.matrix_exp(_symmetrize(x_log).contiguous())
            frequency_output_log, aux = self._apply_attention_along_axis(
                self.frequency_attention,
                x_spd,
                axis=2,
                return_aux=return_aux,
            )
            if return_aux:
                all_aux.update(
                    {f"axis_2_head_0_{name}": param for name, param in aux.items()}
                )

            x_log = self.frequency_add_norm1(x_log, frequency_output_log)
            x_log = self.frequency_add_norm2(x_log, self.frequency_ffn(x_log))

        if attention_input.ndim == 6 and attention_input.shape[3] > 1:
            x_spd = torch.matrix_exp(_symmetrize(x_log))
            region_output_log, aux = self._apply_attention_along_axis(
                self.region_attention,
                x_spd,
                axis=3,
                return_aux=return_aux,
            )
            if return_aux:
                all_aux.update(
                    {f"axis_3_head_0_{name}": param for name, param in aux.items()}
                )

            x_log = self.region_add_norm1(x_log, region_output_log)
            x_log = self.region_add_norm2(x_log, self.region_ffn(x_log))

        x_log = _symmetrize(x_log)
        if return_log:
            return x_log, all_aux

        x_spd = torch.matrix_exp(x_log.contiguous())

        return _symmetrize(x_spd), all_aux


class SPDTransformer(nn.Module):
    """Stacked SPD Transformer encoder."""

    def __init__(
            self,
            num_heads: int,
            spd_in_dim: int,
            attention_dim: [int],
            time_sequence_length,
            stage_transition: True,
            frequency_sequence_length,
            brain_region_sequence_length=1,
            tau=1.0,
            depth: int = 1,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            attention_dropout: float = 0.0,
            debug_attention_dropout: bool = False,
            debug_attention_shape: bool = False,
            debug_tensor_stats: bool = False,
            learnable_metric_mode: Literal["full", "low-rank"] = "low-rank",
            learnable_metric_score: Literal["qgk", "distance"] = "qgk",
            learnable_metric_rank: int | None = None,
            eps: float = 1e-8,
            use_position_bias: bool = True,
            layer_norm_affine: bool = True,
            dropout: float = 0.0,
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: AddNormType = "trace",
            share_metric_across_layers: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.spd_in_dim = spd_in_dim
        self.attention_dim = [spd_in_dim, *[int(dim) for dim in attention_dim]]
        self.depth = depth
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition

        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        
        if num_heads == 1:
            self.layers = nn.ModuleList([SPDEncoder(
                spd_in_dim=self.attention_dim[index] if self.stage_transition else spd_in_dim,
                attention_dim=self.attention_dim[index+1],
                stage_transition=self.stage_transition,
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
            ) for index, dim in enumerate(self.attention_dim[:-1])])
        elif num_heads > 1:
            self.layers = nn.ModuleList([SPDMultiHeadEncoder(
                num_heads=num_heads,
                spd_in_dim=self.attention_dim[index] if self.stage_transition else spd_in_dim,
                attention_dim=self.attention_dim[index + 1],
                stage_transition=self.stage_transition,
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
            ) for index, dim in enumerate(self.attention_dim[:-1])])

        share_axis_metrics(self.layers, across_layers=share_metric_across_layers)

    def forward(
            self,
            x: torch.Tensor,
            return_log: bool = False,
            return_aux: bool = True,
    ):
        all_aux = {}
        for layer_index, layer in enumerate(self.layers):
            layer_return_log = return_log and layer_index == len(self.layers) - 1
            x, aux = layer(
                x,
                return_log=layer_return_log,
                return_aux=return_aux,
            )
            if return_aux:
                for name, param in aux.items():
                    all_aux[name + "_" + str(layer_index)] = param

        return x, all_aux


