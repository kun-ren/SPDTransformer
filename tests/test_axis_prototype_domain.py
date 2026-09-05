from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from src.models.DomainAdversarial import gradient_reverse
from src.models.SPDMDMClassifier import LogEuclideanMDMHead
from src.models.SPDTransformerClassifier import SPDTransformerClassifier
from src.training.domain_adversarial import encode_subject_domains, domain_epoch_options
from src.training.losses import compute_training_objective


def model_config(**overrides):
    return {
        "num_heads": 2, "spd_in_dim": 3, "attention_dim": [3, 3],
        "depth": 2, "num_classes": 2, "stage_transition": True,
        "time_sequence_length": 2, "frequency_sequence_length": 2,
        "brain_region_sequence_length": 2, "metric": "learnable-metric",
        "learnable_metric_rank": 2, "classifier_type": "mdm",
        "pooling": "weighted", "dropout": 0.0, "attention_dropout": 0.0,
        "use_position_bias": False, "layer_norm_affine": False,
        "add_norm_type": "sequence_add_norm", **overrides,
    }


def spd_inputs(batch=4):
    generator = torch.Generator().manual_seed(12)
    factor = 0.1 * torch.randn(batch, 2, 2, 2, 3, 3, generator=generator)
    return factor @ factor.transpose(-1, -2) + torch.eye(3)


def axis_heads(layer, axis):
    attention = getattr(layer, f"{axis}_attention")
    return list(attention) if isinstance(attention, nn.ModuleList) else [attention]


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("mode", ["low-rank", "full"])
def test_axis_metrics_and_qkv_have_requested_parameter_identity(heads, shared, mode):
    model = SPDTransformerClassifier(**model_config(
        num_heads=heads, share_metric_across_layers=shared,
        learnable_metric_mode=mode,
    )).double()
    metric_name = "metric_low_rank" if mode == "low-rank" else "metric_matrix"
    layers = model.model.encoder.layers
    projections = []
    for layer in layers:
        axis_parameters = []
        for axis in ("time", "frequency", "region"):
            attention_heads = axis_heads(layer, axis)
            metric = getattr(attention_heads[0], metric_name)
            assert all(getattr(head, metric_name) is metric for head in attention_heads)
            axis_parameters.append(metric)
            for head in attention_heads:
                projections.extend([head.query.weight, head.key.weight, head.value.weight])
        assert len({id(p) for p in axis_parameters}) == 3
    assert len({id(p) for p in projections}) == len(projections)
    for axis in ("time", "frequency", "region"):
        first = getattr(axis_heads(layers[0], axis)[0], metric_name)
        second = getattr(axis_heads(layers[1], axis)[0], metric_name)
        assert (first is second) == shared

    # Sharing must survive checkpoint round-trips without duplicating optimizer params.
    restored = SPDTransformerClassifier(**model_config(
        num_heads=heads, share_metric_across_layers=shared,
        learnable_metric_mode=mode,
    )).double()
    restored.load_state_dict(model.state_dict())
    parameters = list(restored.parameters())
    assert len(parameters) == len({id(p) for p in parameters})
    torch.testing.assert_close(
        model(spd_inputs().double(), return_aux=False)[0],
        restored(spd_inputs().double(), return_aux=False)[0],
    )


def test_different_layer_dimensions_have_separate_metric_groups():
    model = SPDTransformerClassifier(**model_config(
        depth=3, attention_dim=[3, 2, 2], share_metric_across_layers=True,
    ))
    layers = model.model.encoder.layers
    metrics = [axis_heads(layer, "time")[0].metric_low_rank for layer in layers]
    assert metrics[0].shape == (6, 2)
    assert metrics[1].shape == (3, 2)
    assert metrics[1] is metrics[2]
    assert metrics[0] is not metrics[1]
    assert model(spd_inputs(), return_aux=False)[0].shape == (4, 2)


def test_prototype_penalties_are_class_conditional_and_use_margin():
    head = LogEuclideanMDMHead(2, 2)
    with torch.no_grad():
        head.class_log_prototypes.zero_()
        head.class_log_prototypes[1, 0, 0] = 2.0
    inputs = head.class_log_prototypes.detach().clone()
    intra, inter = head.prototype_losses(inputs, torch.tensor([0, 1]), margin=3.0)
    assert intra.item() == 0.0
    assert inter.item() == pytest.approx(1.0)
    wrong_intra, _ = head.prototype_losses(inputs, torch.tensor([1, 0]), margin=3.0)
    assert wrong_intra.item() == pytest.approx(4.0)


def test_zero_auxiliary_weights_keep_exact_ce_logits_and_gradients():
    model = SPDTransformerClassifier(**model_config(num_heads=1))
    reference = deepcopy(model)
    inputs, targets = spd_inputs(), torch.tensor([0, 1, 0, 1])
    expected_logits, _ = reference(inputs, return_aux=False)
    expected = nn.functional.cross_entropy(expected_logits, targets)
    logits, _, losses = compute_training_objective(
        model, inputs, targets, nn.CrossEntropyLoss(),
        prototype_intra_weight=0.0, prototype_inter_weight=0.0,
    )
    assert torch.equal(logits, expected_logits)
    assert torch.equal(losses["loss"], expected)
    losses["loss"].backward()
    expected.backward()
    for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
        if parameter.grad is None:
            assert ref_parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)


def test_grl_reverses_encoder_gradient_and_warmup_stops_it():
    for coefficient in (0.0, 0.03, 1.0):
        features = torch.tensor([2.0, -1.0], requires_grad=True)
        head_weight = nn.Parameter(torch.tensor([1.0, 3.0]))
        (gradient_reverse(features, coefficient) * head_weight).sum().backward()
        torch.testing.assert_close(features.grad, -coefficient * head_weight.detach())
        torch.testing.assert_close(head_weight.grad, features.detach())


@pytest.mark.parametrize("classifier", ["mdm", "pooling"])
def test_joint_objective_uses_one_encoder_pass_and_trains_both_paths(classifier):
    model = SPDTransformerClassifier(**model_config(
        classifier_type=classifier, domain_adversarial=True, num_domains=2,
        domain_dropout=0.0, share_metric_across_layers=True,
    ))
    calls = []
    handle = model.model.encoder.register_forward_hook(lambda *_: calls.append(1))
    _, _, losses = compute_training_objective(
        model, spd_inputs(), torch.tensor([0, 1, 0, 1]), nn.CrossEntropyLoss(),
        prototype_intra_weight=0.001 if classifier == "mdm" else 0.0,
        prototype_inter_weight=0.01 if classifier == "mdm" else 0.0,
        domain_targets=torch.tensor([0, 0, 1, 1]),
        domain_adversarial_coefficient=0.03,
    )
    handle.remove()
    assert calls == [1]
    losses["loss"].backward()
    assert losses["domain_loss"] > 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    assert model.domain_head.classifier[-1].weight.grad.abs().sum() > 0
    metric = axis_heads(model.model.encoder.layers[0], "time")[0].metric_low_rank
    assert metric.grad is not None and metric.grad.abs().sum() > 0
    model.set_domain_head_trainable(False)
    assert domain_epoch_options(model, {}, 1, 10) == {}


def test_unseen_subjects_do_not_receive_a_source_domain_id():
    labels, mapping = encode_subject_domains(
        np.array(["S001", "S002", "S003", "S002"]), ["S002", "S003"],
    )
    assert mapping == {"S002": 0, "S003": 1}
    assert labels.tolist() == [-1, 0, 1, 0]
    with pytest.raises(ValueError, match="at least two"):
        encode_subject_domains(np.array(["S001"]), ["S001"])
