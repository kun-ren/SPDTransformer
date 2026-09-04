from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


LossComponents = dict[str, torch.Tensor]


def validate_prototype_loss_settings(
        intra_weight: float,
        inter_weight: float,
        margin: float,
) -> tuple[float, float, float]:
    intra_weight = float(intra_weight)
    inter_weight = float(inter_weight)
    margin = float(margin)
    if intra_weight < 0:
        raise ValueError(
            f"prototype_intra_weight must be non-negative, got {intra_weight}."
        )
    if inter_weight < 0:
        raise ValueError(
            f"prototype_inter_weight must be non-negative, got {inter_weight}."
        )
    if margin < 0:
        raise ValueError(f"prototype_margin must be non-negative, got {margin}.")
    return intra_weight, inter_weight, margin


def prototype_loss_settings(config: dict) -> tuple[float, float, float]:
    return validate_prototype_loss_settings(
        config.get("prototype_intra_weight", 0.0),
        config.get("prototype_inter_weight", 0.0),
        config.get("prototype_margin", 1.0),
    )


def compute_training_objective(
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        *,
        condition_regularization_weight: float = 0.0,
        condition_regularization_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prototype_intra_weight: float = 0.0,
        prototype_inter_weight: float = 0.0,
        prototype_margin: float = 1.0,
) -> tuple[torch.Tensor, dict, LossComponents]:
    """Run one forward pass and compose all configured supervised losses."""
    condition_weight = float(condition_regularization_weight)
    if condition_weight < 0:
        raise ValueError(
            "condition_regularization_weight must be non-negative, got "
            f"{condition_weight}."
        )
    intra_weight, inter_weight, margin = validate_prototype_loss_settings(
        prototype_intra_weight,
        prototype_inter_weight,
        prototype_margin,
    )
    use_condition_loss = condition_weight > 0
    use_prototype_loss = intra_weight > 0 or inter_weight > 0

    if use_prototype_loss:
        prototype_forward = getattr(model, "forward_with_prototype_losses", None)
        if not callable(prototype_forward):
            raise ValueError(
                "Non-zero prototype loss weights require a model with an MDM "
                "prototype head."
            )
        logits, aux, intra_loss, inter_loss = prototype_forward(
            inputs,
            targets,
            prototype_margin=margin,
            return_aux=use_condition_loss,
        )
    else:
        logits, aux = model(inputs, return_aux=use_condition_loss)
        intra_loss = logits.new_zeros(())
        inter_loss = logits.new_zeros(())

    cross_entropy = criterion(logits, targets)
    condition_loss = logits.new_zeros(())
    if use_condition_loss:
        if condition_regularization_fn is None:
            raise ValueError(
                "condition_regularization_fn is required when condition "
                "regularization is enabled."
            )
        if aux:
            condition_loss = torch.stack(
                [condition_regularization_fn(matrix) for matrix in aux.values()]
            ).mean()

    # Add only enabled terms so lambda1=lambda2=0 follows the original CE path.
    total_loss = cross_entropy
    if condition_weight > 0:
        total_loss = total_loss + condition_weight * condition_loss
    if intra_weight > 0:
        total_loss = total_loss + intra_weight * intra_loss
    if inter_weight > 0:
        total_loss = total_loss + inter_weight * inter_loss

    components = {
        "loss": total_loss,
        "cross_entropy": cross_entropy,
        "condition_loss": condition_loss,
        "prototype_intra_loss": intra_loss,
        "prototype_inter_loss": inter_loss,
    }
    return logits, aux, components
