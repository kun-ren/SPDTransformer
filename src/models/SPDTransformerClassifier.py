from typing import Literal

import torch
from torch import nn

from src.models.SPDMDMClassifier import SPDMDMClassifier
from src.models.SPDPoolingClassifier import SPDPoolingClassifier
from src.models.SPDTaskTagClassifier import SPDTaskTagClassifier


ClassifierType = Literal["pooling", "task", "mdm"]


class SPDTransformerClassifier(nn.Module):
    """
    Selects the trial-level classifier style.

    classifier_type="pooling":
        no task tag; use mean or learned positional weighted pooling over
        encoder tokens, then classify tangent-space features with a linear head.

    classifier_type="task":
        insert SPD [TASK] token; classify encoded task-token output with MDM.

    classifier_type="mdm":
        pool encoder tokens into one log-domain SPD matrix and classify by
        distance to learnable MDM prototypes.
    """

    def __init__(
            self,
            num_heads: int,
            spd_in_dim: int,
            attention_dim: [int],
            num_classes: int,
            stage_transition: True,
            time_sequence_length,
            frequency_sequence_length,
            brain_region_sequence_length=1,
            tau=1.0,
            ffn_hidden_spd_dim=None,
            metric: str = "log-euclidean",
            depth: int = 1,
            classifier_type: ClassifierType = "pooling",
            pooling: str = "weighted",
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
            stage_projection_init: Literal["identity", "random"] = "identity",
            add_norm_type: str = "trace",
    ):
        super().__init__()
        self.debug_tensor_stats = debug_tensor_stats
        if pooling == "task":
            classifier_type = "task"
            pooling = "mean"
        if classifier_type not in {"pooling", "task", "mdm"}:
            raise ValueError(
                "classifier_type must be 'pooling', 'task', or 'mdm', "
                f"got {classifier_type!r}."
            )

        self.classifier_type = classifier_type
        if classifier_type == "pooling":
            self.model = SPDPoolingClassifier(
                num_heads=num_heads,
                spd_in_dim=spd_in_dim,
                attention_dim=attention_dim,
                num_classes=num_classes,
                stage_transition=stage_transition,
                time_sequence_length=time_sequence_length,
                frequency_sequence_length=frequency_sequence_length,
                brain_region_sequence_length=brain_region_sequence_length,
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
                stage_projection_init=stage_projection_init,
                add_norm_type=add_norm_type,
            )
        elif classifier_type == "mdm":
            self.model = SPDMDMClassifier(
                num_heads=num_heads,
                spd_in_dim=spd_in_dim,
                attention_dim=attention_dim,
                num_classes=num_classes,
                stage_transition=stage_transition,
                time_sequence_length=time_sequence_length,
                frequency_sequence_length=frequency_sequence_length,
                brain_region_sequence_length=brain_region_sequence_length,
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
                stage_projection_init=stage_projection_init,
                add_norm_type=add_norm_type,
            )
        else:
            self.model = SPDTaskTagClassifier(
                num_heads=num_heads,
                spd_in_dim=spd_in_dim,
                attention_dim=attention_dim,
                num_classes=num_classes,
                stage_transition=stage_transition,
                time_sequence_length=time_sequence_length,
                frequency_sequence_length=frequency_sequence_length,
                brain_region_sequence_length=brain_region_sequence_length,
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
                stage_projection_init=stage_projection_init,
                add_norm_type=add_norm_type,
            )

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict]:

        return self.model(x, return_aux=return_aux)
