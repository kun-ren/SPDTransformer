"""Fail at the first non-finite SPD operation without replacing its values."""

import torch


def tensor_summary(x: torch.Tensor) -> str:
    with torch.no_grad():
        finite = torch.isfinite(x)
        values = x[finite]
        bounds = (
            f"min={values.min().item():.6e} max={values.max().item():.6e}"
            if values.numel() else "min=NA max=NA"
        )
        return (
            f"shape={tuple(x.shape)} dtype={x.dtype} "
            f"finite={int(finite.sum())}/{x.numel()} "
            f"nan={int(torch.isnan(x).sum())} "
            f"inf={int(torch.isinf(x).sum())} {bounds}"
        )


def require_finite(x: torch.Tensor, context: str) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite tensor at {context}: {tensor_summary(x)}")


def checked_matrix_exp(x: torch.Tensor, context: str) -> torch.Tensor:
    require_finite(x, f"{context}.input_log")
    output = torch.matrix_exp(x)
    if not torch.isfinite(output).all():
        with torch.no_grad():
            symmetric = x.detach().double()
            symmetric = 0.5 * symmetric + 0.5 * symmetric.transpose(-1, -2)
            try:
                spectrum = torch.linalg.eigvalsh(symmetric)
                spectral_info = (
                    f"log_eigenvalue_min={spectrum.min().item():.6e} "
                    f"log_eigenvalue_max={spectrum.max().item():.6e} "
                    f"max_log_spectral_spread="
                    f"{(spectrum[..., -1] - spectrum[..., 0]).max().item():.6e}"
                )
            except RuntimeError:
                spectral_info = "log_spectrum=unavailable"
        raise RuntimeError(
            f"Non-finite matrix_exp at {context}: {tensor_summary(output)}; "
            f"input_log: {tensor_summary(x)}; {spectral_info}. "
            "The failure precedes attention scoring; score clipping cannot fix it."
        )
    return output
