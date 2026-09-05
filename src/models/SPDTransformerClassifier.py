from typing import Literal

import torch
from torch import nn

from src.models.DomainAdversarial import DomainAdversarialHead
from src.models.SPDMDMClassifier import SPDMDMClassifier
from src.models.SPDPoolingClassifier import SPDPoolingClassifier



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
            learnable_metric_mode: Literal["full", "low-rank"] = "low-rank",
            learnable_metric_score: Literal["qgk", "distance"] = "qgk",
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
            share_metric_across_layers: bool = False,
            domain_adversarial: bool = False,
            num_domains: int | None = None,
            domain_hidden_dim: int = 32,
            domain_dropout: float = 0.3,
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

            self.model = None
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
                learnable_metric_score=learnable_metric_score,
                learnable_metric_rank=learnable_metric_rank,
                eps=eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
                stage_projection_init=stage_projection_init,
                add_norm_type=add_norm_type,
                share_metric_across_layers=share_metric_across_layers,
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
                learnable_metric_score=learnable_metric_score,
                learnable_metric_rank=learnable_metric_rank,
                eps=eps,
                use_position_bias=use_position_bias,
                layer_norm_affine=layer_norm_affine,
                stage_projection_init=stage_projection_init,
                add_norm_type=add_norm_type,
                share_metric_across_layers=share_metric_across_layers,
            )
        else:
            self.model = None

        if not isinstance(domain_adversarial, bool):
            raise TypeError(
                "domain_adversarial must be a bool, "
                f"got {type(domain_adversarial).__name__}."
            )
        self.domain_adversarial = domain_adversarial
        if self.domain_adversarial:
            if self.encoder_type != "spd" or self.classifier_type not in {
                    "pooling",
                    "mdm",
            }:
                raise ValueError(
                    "Domain adversarial training requires encoder_type='spd' "
                    "and classifier_type='pooling' or 'mdm'."
                )
            if num_domains is None:
                raise ValueError(
                    "num_domains is required when domain_adversarial=true."
                )
            self.domain_head = DomainAdversarialHead(
                spd_dim=self.model.transformer_out_dim,
                num_domains=int(num_domains),
                hidden_dim=int(domain_hidden_dim),
                dropout=float(domain_dropout),
            )
        else:
            self.domain_head = None

    def forward(
            self,
            x: torch.Tensor,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        if self.encoder_type == "tangent":
            logits = self.model(x, return_aux=False)
            return logits, {}
        return self.model(x, return_aux=return_aux)

    def forward_with_prototype_losses(
            self,
            x: torch.Tensor,
            targets: torch.Tensor,
            *,
            prototype_margin: float,
            return_aux: bool = True,
    ) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        logits, aux, intra_loss, inter_loss, _domain_logits = (
            self.forward_with_training_outputs(
                x,
                targets,
                prototype_margin=prototype_margin,
                compute_prototype_losses=True,
                compute_domain_logits=False,
                domain_reversal_coefficient=0.0,
                return_aux=return_aux,
            )
        )
        return logits, aux, intra_loss, inter_loss

    def forward_with_training_outputs(
            self,
            x: torch.Tensor,
            targets: torch.Tensor,
            *,
            prototype_margin: float,
            compute_prototype_losses: bool,
            compute_domain_logits: bool,
            domain_reversal_coefficient: float,
            return_aux: bool = True,
    ) -> tuple[
        torch.Tensor,
        dict,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if self.encoder_type != "spd" or self.model is None:
            raise ValueError(
                "Training auxiliary outputs require encoder_type='spd'."
            )
        logits, aux, pooled_log = self.model.forward_with_pooled(
            x,
            return_aux=return_aux,
        )
        intra_loss = logits.new_zeros(())
        inter_loss = logits.new_zeros(())
        if compute_prototype_losses:
            if self.classifier_type != "mdm":
                raise ValueError(
                    "Prototype losses require classifier_type='mdm'."
                )
            intra_loss, inter_loss = self.model.mdm_head.prototype_losses(
                pooled_log,
                targets,
                margin=prototype_margin,
            )

        domain_logits = None
        if compute_domain_logits:
            if self.domain_head is None:
                raise ValueError(
                    "Domain labels were provided but domain_adversarial is disabled."
                )
            domain_logits = self.domain_head(
                pooled_log,
                reversal_coefficient=domain_reversal_coefficient,
            )
        return logits, aux, intra_loss, inter_loss, domain_logits

    def set_domain_head_trainable(self, trainable: bool) -> None:
        if self.domain_head is None:
            return
        for parameter in self.domain_head.parameters():
            parameter.requires_grad_(trainable)
