import torch

from torch import nn


class SPDClassifierBase(nn.Module):
    @staticmethod
    def upper_triangular_vectorize(x: torch.Tensor) -> torch.Tensor:
        spd_dim = x.shape[-1]
        row, col = torch.triu_indices(spd_dim, spd_dim, device=x.device)
        return x[..., row, col]

    @staticmethod
    def build_linear_classifier(
            feature_dim: int,
            num_classes: int,
            dropout: float,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )