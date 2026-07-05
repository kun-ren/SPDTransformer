from typing import Literal

import torch
from torch import nn

from src.models.GeooptBiMap import GeooptBiMap
from src.models.SPDAttention import SingleHeadAttention, spd_log
from src.models.SPDFeedForward import SPDFeedForward
from src.models.TraceAddNorm import TraceAddNorm


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


class SPDMultiHeadEncoder(nn.Module):
    def __init__(
            self,
            spd_in_dim,
            attention_dim,
            time_sequence_length,
            frequency_sequence_length,
            num_heads=4,
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
            stage_projection_init: Literal["identity", "random"] = "identity",
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")

        print(f"input_dim: {spd_in_dim}, attention_dim: {attention_dim}, heads={num_heads}")
        self.metric = metric
        self.num_heads = num_heads
        self.spd_in_dim = spd_in_dim
        self.attention_dim = attention_dim
        self.debug_attention_shape = debug_attention_shape
        self.debug_tensor_stats = debug_tensor_stats
        self.stage_transition = stage_transition
        self.eps = eps

        self.stage_projection = None
        self.head_stage_projections = None
        if self.stage_transition and attention_dim != spd_in_dim:
            if stage_projection_init == "identity":
                self.head_stage_projections = nn.ModuleList([
                    GeooptBiMap(
                        spd_in_dim,
                        attention_dim,
                        eps=eps,
                        init="identity",
                        identity_indices=torch.randperm(spd_in_dim)[:attention_dim],
                    )
                    for _ in range(num_heads)
                ])
            else:
                self.stage_projection = GeooptBiMap(
                    spd_in_dim,
                    attention_dim,
                    eps=eps,
                    init=stage_projection_init,
                )

        if self.stage_projection is not None or self.head_stage_projections is not None:
            spd_out_dim = attention_dim
        else:
            spd_out_dim = spd_in_dim

        self.time_attention = nn.ModuleList([
            SingleHeadAttention(
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
            for _ in range(num_heads)
        ])
        self.time_head_logits = nn.Parameter(torch.zeros(num_heads))

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

        self.frequency_attention = nn.ModuleList([
            SingleHeadAttention(
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
            for _ in range(num_heads)
        ])
        self.frequency_head_logits = nn.Parameter(torch.zeros(num_heads))

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
        return _symmetrize(weighted_log)

    @classmethod
    def _apply_attention_along_axis(
            cls,
            attention_heads: nn.ModuleList,
            head_logits: torch.Tensor,
            x: torch.Tensor | list[torch.Tensor],
            axis: int,
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
            y_log = y_log.permute(inverse_perm)

        return y_log, all_aux

    def forward(
            self,
            x,
            return_log: bool = False,
            return_aux: bool = True,
    ):
        if x.ndim not in {4, 5}:
            raise ValueError(
                "Expected input shape (batch, time, channels, channels) or "
                "(batch, time, frequency_bands, channels, channels), "
                f"got {tuple(x.shape)}."
            )
        all_aux = {}
        if self.head_stage_projections is not None:
            projected_inputs = [
                stage_projection(x)
                for stage_projection in self.head_stage_projections
            ]
            if return_aux:
                for head_index, projected_input in enumerate(projected_inputs):
                    all_aux[f"P_x_head_{head_index}"] = projected_input

            residual_log = self._combine_head_logs(
                [spd_log(projected_input) for projected_input in projected_inputs],
                self.time_head_logits,
            )
            attention_input = projected_inputs
        elif self.stage_projection is not None:
            x = self.stage_projection(x)
            if return_aux:
                all_aux["P_x"] = x
            residual_log = spd_log(x)
            attention_input = x
        else:
            eye = torch.eye(
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            x = _symmetrize(x) + self.eps * eye
            residual_log = spd_log(x)
            attention_input = x

        time_output_log, aux = self._apply_attention_along_axis(
            self.time_attention,
            self.time_head_logits,
            attention_input,
            axis=1,
            return_aux=return_aux,
        )
        if return_aux:
            all_aux.update(aux)
        x_log = self.time_add_norm1(residual_log, time_output_log)

        x_log = self.time_add_norm2(x_log, self.time_ffn(x_log))

        if x.ndim == 5 and x.shape[-3] > 1:
            x_spd = torch.matrix_exp(
                0.5 * (x_log + x_log.transpose(-1, -2))
            )
            frequency_output_log, aux = self._apply_attention_along_axis(
                self.frequency_attention,
                self.frequency_head_logits,
                x_spd,
                axis=2,
                return_aux=return_aux,
            )
            if return_aux:
                all_aux.update(aux)

            x_log = self.frequency_add_norm1(x_log, frequency_output_log)

            x_log = self.frequency_add_norm2(x_log, self.frequency_ffn(x_log))

        x_log = _symmetrize(x_log)
        if return_log:
            return x_log, all_aux

        x_spd = torch.matrix_exp(x_log)

        return _symmetrize(x_spd), all_aux
