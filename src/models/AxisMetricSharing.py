from torch import nn


def share_axis_metrics(layers: nn.ModuleList, across_layers: bool = False) -> None:
    """Tie metric parameters by axis and compatible geometry, never Q/K/V."""
    if not isinstance(across_layers, bool):
        raise TypeError("share_metric_across_layers must be a bool.")
    shared = {}
    for layer in layers:
        for axis in ("time", "frequency", "region"):
            attention = getattr(layer, f"{axis}_attention")
            heads = list(attention) if isinstance(attention, nn.ModuleList) else [attention]
            reference = heads[0]
            signature = (
                axis,
                reference.metric,
                reference.learnable_metric_mode,
                reference.learnable_metric_score,
                reference.attention_dim,
                reference.learnable_metric_rank,
            )
            if across_layers:
                reference = shared.setdefault(signature, reference)
            for head in heads:
                for name in ("metric_matrix", "metric_low_rank", "affine_log_scale_raw"):
                    parameter = getattr(reference, name, None)
                    target = getattr(head, name, None)
                    if parameter is not None:
                        if target is None or target.shape != parameter.shape:
                            raise ValueError(f"Incompatible {axis} metric parameter {name}.")
                        setattr(head, name, parameter)
