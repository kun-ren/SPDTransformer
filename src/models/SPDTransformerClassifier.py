from typing import Literal

import torch
from torch import nn

from src.models.SPDMDMClassifier import SPDMDMClassifier
from src.models.SPDPoolingClassifier import SPDPoolingClassifier
from src.models.SPDTaskTagClassifier import SPDTaskTagClassifier
from src.models.TangentTransformerMDMClassifier import (
    TangentTransformerMDMClassifier,
)


ClassifierType = Literal["pooling", "task", "mdm"]
EncoderType = Literal["spd", "tangent"]


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
            encoder_type: EncoderType = "spd",
            tangent_d_model: int | None = None,
            tangent_nhead: int | None = None,
            tangent_num_layers: int | None = None,
            tangent_dim_feedforward: int | None = None,
            tangent_activation: Literal["relu", "gelu"] = "gelu",
            tangent_norm_first: bool = False,
            tangent_use_position_embedding: bool | None = None,
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

        encoder_type = str(encoder_type).strip().lower().replace("-", "_")
        encoder_aliases = {
            "spd": "spd",
            "spd_transformer": "spd",
            "riemannian": "spd",
            "tangent": "tangent",
            "tangent_transformer": "tangent",
            "euclidean": "tangent",
        }
        if encoder_type not in encoder_aliases:
            raise ValueError(
                "encoder_type must be 'spd' or 'tangent', "
                f"got {encoder_type!r}."
            )
        encoder_type = encoder_aliases[encoder_type]
        if encoder_type == "tangent" and classifier_type != "mdm":
            raise ValueError(
                "The tangent Transformer ablation currently requires "
                "classifier_type='mdm' so that the classifier remains "
                "identical to the SPDTransformer weighted-MDM setup."
            )

        self.classifier_type = classifier_type
        self.encoder_type = encoder_type
        if encoder_type == "tangent":
            transformer_out_dim = (
                int(attention_dim[-1]) if stage_transition else int(spd_in_dim)
            )
            tangent_input_dim = spd_in_dim * (spd_in_dim + 1) // 2
            if tangent_d_model is None:
                tangent_d_model = tangent_input_dim
            if tangent_nhead is None:
                tangent_nhead = num_heads
            if tangent_num_layers is None:
                tangent_num_layers = depth
            if tangent_dim_feedforward is None:
                tangent_dim_feedforward = ffn_hidden_spd_dim
            if tangent_use_position_embedding is None:
                tangent_use_position_embedding = use_position_bias

            self.model = TangentTransformerMDMClassifier(
                spd_dim=spd_in_dim,
                output_spd_dim=transformer_out_dim,
                num_classes=num_classes,
                token_shape=(
                    int(time_sequence_length),
                    int(frequency_sequence_length),
                    int(brain_region_sequence_length),
                ),
                d_model=int(tangent_d_model),
                nhead=int(tangent_nhead),
                num_layers=int(tangent_num_layers),
                dim_feedforward=(
                    None
                    if tangent_dim_feedforward is None
                    else int(tangent_dim_feedforward)
                ),
                dropout=dropout,
                activation=tangent_activation,
                norm_first=tangent_norm_first,
                pooling=pooling,
                use_position_embedding=bool(tangent_use_position_embedding),
                eps=eps,
            )
        elif classifier_type == "pooling":
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
        if self.encoder_type == "tangent":
            logits = self.model(x, return_aux=False)
            return logits, {}
        return self.model(x, return_aux=return_aux)
