from typing import Literal

import torch
from torch import nn

from src.models.GeooptBiMap import GeooptBiMap
from src.models.SPDAttention import (
    AttentionMetricParameters,
    SingleHeadAttention,
    normalize_position_bias_axes,
    spd_exp,
    spd_log,
)
from src.models.SPDFeedForward import SPDFeedForward
from src.models.LogResidualAdd import LogResidualAdd
from src.models.ScaleShapeSequenceAddNorm import ScaleShapeSequenceAddNorm
from src.models.SequenceAddNorm import SequenceAddNorm
from src.models.SharedTraceAddNorm import SharedTraceAddNorm
from src.models.TraceAddNorm import TraceAddNorm


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


AddNormType = Literal[
    "trace",
    "trace_add_norm",
    "shared_trace",
    "shared_trace_add_norm",
    "scale_shape_sequence",
    "scale_shape_sequence_add_norm",
    "log_residual",
    "log_residual_add",
    "sequence_add_norm",
    "none",
]


def _make_add_norm(
        add_norm_type: str,
        spd_dim: int,
        sequence_length: int,
        tau: float,
        eps: float,
        affine: bool,
        position_axis: int,
) -> nn.Module:
    normalized = str(add_norm_type).strip().lower().replace("-", "_")
    if normalized in {"trace", "trace_add_norm"}:
        return TraceAddNorm(
            spd_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            position_axis=position_axis,
        )
    if normalized in {"shared_trace", "shared_trace_add_norm"}:
        return SharedTraceAddNorm(
            spd_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            position_axis=position_axis,
        )
    if normalized in {
            "scale_shape_sequence",
            "scale_shape_sequence_add_norm",
    }:
        return ScaleShapeSequenceAddNorm(
            spd_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            position_axis=position_axis,
        )
    if normalized in {"log_residual", "log_residual_add", "none"}:
        return LogResidualAdd(
            spd_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            position_axis=position_axis,
        )
    if normalized in {"sequence_add_norm"}:
        return SequenceAddNorm(
            spd_dim,
            sequence_length=sequence_length,
            tau=tau,
            eps=eps,
            affine=affine,
            position_axis=position_axis,
        )
    raise ValueError(
        "add_norm_type must be 'trace', 'shared_trace', "
        "'scale_shape_sequence', 'sequence_add_norm', or 'log_residual', "
        f"got {add_norm_type!r}."
    )


class SPDMultiHeadEncoder(nn.Module):
    def __init__(
            self,
            spd_in_dim,
            attention_dim,
            time_sequence_length,
            frequency_sequence_length,
            brain_region_sequence_length=1,
            num_heads=4,
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
            position_bias_axes: str | tuple[str, ...] | list[str] | None = None,
            position_bias_max: float = 0.5,
            attention_score_target_rms: float = 1.0,
            attention_score_clip: float = 5.0,
            layer_norm_affine: bool = True,
            dropout: float = 0.0,
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: AddNormType = "trace",
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        print("init multi head encoder")

        print(f"input_dim: {spd_in_dim}, attention_dim: {attention_dim}, heads={num_heads}")
        self.metric = metric
        self.num_heads = num_heads
        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.debug_attention_shape = debug_attention_shape
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition
        self.eps = eps
        enabled_position_bias_axes = normalize_position_bias_axes(
            use_position_bias,
            position_bias_axes,
        )

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

        self.shared_metric = AttentionMetricParameters(
            attention_dim=attention_dim,
            metric=self.metric,
            learnable_metric_mode=learnable_metric_mode,
            learnable_metric_score=learnable_metric_score,
            learnable_metric_rank=learnable_metric_rank,
        )

        self.time_attention = nn.ModuleList([
            SingleHeadAttention(
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
                use_position="time" in enabled_position_bias_axes,
                max_position=time_sequence_length,
                position_bias_max=position_bias_max,
                attention_score_target_rms=attention_score_target_rms,
                attention_score_clip=attention_score_clip,
                debug_tensor_stats=debug_tensor_stats,
                own_metric_parameters=False,
            )
            for _ in range(num_heads)
        ])
        self.time_head_logits = nn.Parameter(torch.zeros(num_heads))

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

        self.frequency_attention = nn.ModuleList([
            SingleHeadAttention(
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
                use_position="frequency" in enabled_position_bias_axes,
                max_position=frequency_sequence_length,
                position_bias_max=position_bias_max,
                attention_score_target_rms=attention_score_target_rms,
                attention_score_clip=attention_score_clip,
                debug_tensor_stats=debug_tensor_stats,
                own_metric_parameters=False,
            )
            for _ in range(num_heads)
        ])
        self.frequency_head_logits = nn.Parameter(torch.zeros(num_heads))

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

        self.region_attention = nn.ModuleList([
            SingleHeadAttention(
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
                use_position="region" in enabled_position_bias_axes,
                max_position=brain_region_sequence_length,
                position_bias_max=position_bias_max,
                attention_score_target_rms=attention_score_target_rms,
                attention_score_clip=attention_score_clip,
                debug_tensor_stats=debug_tensor_stats,
                own_metric_parameters=False,
            )
            for _ in range(num_heads)
        ])
        self.region_head_logits = nn.Parameter(torch.zeros(num_heads))

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
    def _combine_head_logs(
            head_logs: list[torch.Tensor],
            head_logits: torch.Tensor,
    ) -> torch.Tensor:
        if len(head_logs) != head_logits.numel():
            raise ValueError(
                f"Expected {head_logits.numel()} head outputs, got {len(head_logs)}."
            )

        stacked = torch.stack(head_logs, dim=0)
        weights = torch.softmax(
            head_logits.to(device=stacked.device, dtype=stacked.dtype),
            dim=0,
        )
        view_shape = [weights.shape[0], *([1] * (stacked.ndim - 1))]
        weighted_log = (stacked * weights.view(view_shape)).sum(dim=0)
        return weighted_log

    @classmethod
    def _apply_attention_along_axis(
            cls,
            attention_heads: nn.ModuleList,
            head_logits: torch.Tensor,
            x: torch.Tensor | list[torch.Tensor],
            axis: int,
            metric_parameters: AttentionMetricParameters,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(x, torch.Tensor):
            head_inputs = [x] * len(attention_heads)
        else:
            head_inputs = list(x)
            if len(head_inputs) != len(attention_heads):
                raise ValueError(
                    f"Expected {len(attention_heads)} head inputs, "
                    f"got {len(head_inputs)}."
                )

        x0 = head_inputs[0]
        if x0.ndim < 4:
            raise ValueError(
                "Expected SPD input with shape (..., sequence, channels, channels), "
                f"got {tuple(x0.shape)}."
            )

        leading_ndim = x0.ndim - 2
        if axis < 0:
            axis += leading_ndim
        if not 0 <= axis < leading_ndim:
            raise ValueError(
                f"axis must refer to one of the {leading_ndim} leading dimensions, "
                f"got axis={axis}."
            )

        seq_pos = leading_ndim - 1
        perm = list(range(x0.ndim))
        if axis != seq_pos:
            moved_axis = perm.pop(axis)
            perm.insert(seq_pos, moved_axis)
            head_inputs = [x_head.permute(perm) for x_head in head_inputs]

        all_aux: dict[str, torch.Tensor] = {}
        head_logs = []
        for head_index, attention_head in enumerate(attention_heads):
            y_log, aux = attention_head(
                head_inputs[head_index],
                return_aux=return_aux,
                metric_parameters=metric_parameters,
            )
            head_logs.append(y_log)
            if return_aux:
                for key, value in aux.items():
                    all_aux[f"axis_{axis}_head_{head_index}_{key}"] = value

        y_log = cls._combine_head_logs(head_logs, head_logits)

        if axis != seq_pos:
            inverse_perm = [0] * len(perm)
            for new_axis, old_axis in enumerate(perm):
                inverse_perm[old_axis] = new_axis
            y_log = y_log.permute(inverse_perm).contiguous()

        return y_log, all_aux

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
        all_aux = {}
        if self.stage_projection is not None:
            attention_input = self.stage_projection(x)
            if return_aux:
                all_aux["P_x"] = attention_input
        else:
            attention_input = x

        residual_log = spd_log(attention_input)

        time_output_log, aux = self._apply_attention_along_axis(
            self.time_attention,
            self.time_head_logits,
            attention_input,
            axis=1,
            metric_parameters=self.shared_metric,
            return_aux=return_aux,
        )
        if return_aux:
            all_aux.update(aux)
        x_log = self.time_add_norm1(residual_log, time_output_log)

        x_log = self.time_add_norm2(x_log, self.time_ffn(x_log))

        if attention_input.ndim >= 5 and attention_input.shape[2] > 1:
            x_spd = spd_exp(x_log, eps=self.eps)

            frequency_output_log, aux = self._apply_attention_along_axis(
                self.frequency_attention,
                self.frequency_head_logits,
                x_spd,
                axis=2,
                metric_parameters=self.shared_metric,
                return_aux=return_aux,
            )
            if return_aux:
                all_aux.update(aux)

            x_log = self.frequency_add_norm1(x_log, frequency_output_log)
            x_log = self.frequency_add_norm2(x_log, self.frequency_ffn(x_log))

        if attention_input.ndim == 6 and attention_input.shape[3] > 1:
            x_spd = spd_exp(x_log, eps=self.eps)

            region_output_log, aux = self._apply_attention_along_axis(
                self.region_attention,
                self.region_head_logits,
                x_spd,
                axis=3,
                metric_parameters=self.shared_metric,
                return_aux=return_aux,
            )
            if return_aux:
                all_aux.update(aux)

            x_log = self.region_add_norm1(x_log, region_output_log)
            x_log = self.region_add_norm2(x_log, self.region_ffn(x_log))

        x_log = _symmetrize(x_log)
        if return_log:
            return x_log, all_aux

        x_spd = spd_exp(x_log, eps=self.eps)

        return x_spd, all_aux
