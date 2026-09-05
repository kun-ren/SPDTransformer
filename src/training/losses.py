from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


LossComponents = dict[str, torch.Tensor]


def _parse_bool_setting(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse {name} boolean value {value!r}.")


def domain_adversarial_settings(
        config: dict,
) -> tuple[float, int, str, bool]:
    max_weight = float(config.get("domain_adversarial_max_weight", 0.03))
    warmup_epochs = int(config.get("domain_adversarial_warmup_epochs", 0))
    schedule = str(
        config.get("domain_adversarial_schedule", "dann")
    ).strip().lower().replace("-", "_")
    normalize_loss = _parse_bool_setting(
        config.get("domain_loss_normalize", True),
        "domain_loss_normalize",
    )
    if not math.isfinite(max_weight) or max_weight < 0.0:
        raise ValueError(
            "domain_adversarial_max_weight must be finite and non-negative, "
            f"got {max_weight}."
        )
    if warmup_epochs < 0:
        raise ValueError(
            "domain_adversarial_warmup_epochs must be non-negative, "
            f"got {warmup_epochs}."
        )
    aliases = {
        "dann": "dann",
        "sigmoid": "dann",
        "linear": "linear",
        "constant": "constant",
    }
    if schedule not in aliases:
        raise ValueError(
            "domain_adversarial_schedule must be 'dann', 'linear', or "
            f"'constant', got {schedule!r}."
        )
    return max_weight, warmup_epochs, aliases[schedule], normalize_loss


def domain_adversarial_coefficient(
        *,
        epoch: int,
        total_epochs: int,
        max_weight: float,
        warmup_epochs: int,
        schedule: str,
) -> float:
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}.")
    if not 1 <= epoch <= total_epochs:
        raise ValueError(
            f"epoch must be in [1, {total_epochs}], got {epoch}."
        )
    if epoch <= warmup_epochs or max_weight == 0.0:
        return 0.0

    active_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / active_epochs, 0.0), 1.0)
    if schedule == "constant":
        factor = 1.0
    elif schedule == "linear":
        factor = progress
    elif schedule == "dann":
        factor = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    else:
        raise ValueError(f"Unsupported domain adversarial schedule {schedule!r}.")
    return float(max_weight) * factor


def validate_prototype_loss_settings(
        intra_weight: float,
        inter_weight: float,
        margin: float,
) -> tuple[float, float, float]:
    intra_weight = float(intra_weight)
    inter_weight = float(inter_weight)
    margin = float(margin)
    if not math.isfinite(intra_weight) or intra_weight < 0:
        raise ValueError(
            f"prototype_intra_weight must be non-negative, got {intra_weight}."
        )
    if not math.isfinite(inter_weight) or inter_weight < 0:
        raise ValueError(
            f"prototype_inter_weight must be non-negative, got {inter_weight}."
        )
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"prototype_margin must be non-negative, got {margin}.")
    return intra_weight, inter_weight, margin


def prototype_loss_settings(config: dict) -> tuple[float, float, float]:
    return validate_prototype_loss_settings(
        config.get("prototype_intra_weight", 0.0),
        config.get("prototype_inter_weight", 0.0),
        config.get("prototype_margin", 1.0),
    )


def prototype_loss_options(config: dict) -> dict[str, float]:
    return dict(zip(
        ("prototype_intra_weight", "prototype_inter_weight", "prototype_margin"),
        prototype_loss_settings(config),
    ))


def auxiliary_loss_history(metrics: dict, prefix: str = "train") -> dict:
    return {
        f"{prefix}_{name}": float(metrics[name])
        for name in (
            "cross_entropy", "prototype_intra_loss", "prototype_inter_loss",
            "domain_loss", "domain_accuracy", "domain_adversarial_coefficient",
        )
        if name in metrics
    }


def format_auxiliary_losses(metrics: dict) -> str:
    return (
        f"ce={metrics['cross_entropy']:.4f} "
        f"intra={metrics['prototype_intra_loss']:.4f} "
        f"inter={metrics['prototype_inter_loss']:.4f} "
        f"domain={metrics.get('domain_loss', 0.0):.4f} "
        f"domain_acc={metrics.get('domain_accuracy', 0.0):.4f} "
        f"grl={metrics.get('domain_adversarial_coefficient', 0.0):.4f}"
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
        domain_targets: torch.Tensor | None = None,
        domain_adversarial_coefficient: float = 0.0,
        domain_loss_normalize: bool = True,
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
    use_domain_loss = domain_targets is not None

    domain_normalizer = 1.0
    reversal_coefficient = float(domain_adversarial_coefficient)
    if use_domain_loss:
        if not math.isfinite(reversal_coefficient) or reversal_coefficient < 0.0:
            raise ValueError(
                "domain_adversarial_coefficient must be finite and "
                f"non-negative, got {reversal_coefficient}."
            )
        domain_head = getattr(model, "domain_head", None)
        num_domains = getattr(domain_head, "num_domains", None)
        if num_domains is None:
            raise ValueError(
                "Domain labels require a model built with "
                "domain_adversarial=true."
            )
        if domain_loss_normalize:
            domain_normalizer = max(math.log(float(num_domains)), 1.0)

    if use_domain_loss:
        training_forward = getattr(model, "forward_with_training_outputs", None)
        if not callable(training_forward):
            raise ValueError(
                "Domain adversarial training requires a model with "
                "forward_with_training_outputs()."
            )
        logits, aux, intra_loss, inter_loss, domain_logits = training_forward(
            inputs,
            targets,
            prototype_margin=margin,
            compute_prototype_losses=use_prototype_loss,
            compute_domain_logits=True,
            # Keep max_weight as the effective encoder coefficient after
            # optional CE normalization.
            domain_reversal_coefficient=(
                reversal_coefficient * domain_normalizer
            ),
            return_aux=use_condition_loss,
        )

    elif use_prototype_loss:
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

    domain_loss = logits.new_zeros(())
    domain_accuracy = logits.new_zeros(())
    if use_domain_loss:
        assert domain_targets is not None
        assert domain_logits is not None
        if domain_targets.shape != targets.shape:
            raise ValueError(
                "domain targets must match class target shape: "
                f"{tuple(domain_targets.shape)} != {tuple(targets.shape)}."
            )
        domain_targets = domain_targets.long()
        if domain_targets.numel() == 0 or int(domain_targets.min()) < 0:
            raise ValueError("Domain targets must contain non-negative IDs.")
        if int(domain_targets.max()) >= domain_logits.shape[-1]:
            raise ValueError(
                "Domain target exceeds classifier output range: "
                f"max target={int(domain_targets.max())}, "
                f"num domains={domain_logits.shape[-1]}."
            )
        domain_loss = F.cross_entropy(domain_logits, domain_targets)
        domain_loss = domain_loss / domain_normalizer
        domain_accuracy = (
            domain_logits.argmax(dim=-1) == domain_targets
        ).float().mean()

    # Add only enabled terms so all zero/disabled settings keep the CE path.
    total_loss = cross_entropy
    if condition_weight > 0:
        total_loss = total_loss + condition_weight * condition_loss
    if intra_weight > 0:
        total_loss = total_loss + intra_weight * intra_loss
    if inter_weight > 0:
        total_loss = total_loss + inter_weight * inter_loss
    if use_domain_loss:
        total_loss = total_loss + domain_loss

    components = {
        "loss": total_loss,
        "cross_entropy": cross_entropy,
        "condition_loss": condition_loss,
        "prototype_intra_loss": intra_loss,
        "prototype_inter_loss": inter_loss,
        "domain_loss": domain_loss,
        "domain_accuracy": domain_accuracy,
        "domain_adversarial_coefficient": logits.new_tensor(
            reversal_coefficient
        ),
    }
    return logits, aux, components
