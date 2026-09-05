from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.losses import (
    domain_adversarial_coefficient,
    domain_adversarial_settings,
)


def domain_epoch_options(model, config: dict, epoch: int, epochs: int) -> dict:
    head = getattr(model, "domain_head", None)
    if head is None or not any(p.requires_grad for p in head.parameters()):
        return {}
    weight, warmup, schedule, normalize = domain_adversarial_settings(config)
    return {
        "domain_adversarial_coefficient": domain_adversarial_coefficient(
            epoch=epoch, total_epochs=epochs, max_weight=weight,
            warmup_epochs=warmup, schedule=schedule,
        ),
        "domain_loss_normalize": normalize,
    }


class SubjectDomainDataset(Dataset):
    """Add source-domain IDs without copying the full EEG tensor dataset."""

    def __init__(self, dataset: Dataset, domain_labels: np.ndarray) -> None:
        if np.shape(domain_labels) != (len(dataset),):
            raise ValueError("Domain labels must contain one ID per trial.")
        self.dataset = dataset
        self.domain_labels = torch.as_tensor(domain_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        inputs, targets, _domain = unpack_supervised_batch(self.dataset[index])
        return inputs, targets, self.domain_labels[index]


def encode_subject_domains(
        subject_labels: np.ndarray,
        source_subjects: Sequence[str],
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode source subjects while leaving held-out subjects as ``-1``."""

    labels = np.asarray(subject_labels, dtype=np.str_)
    subjects = sorted({str(subject) for subject in source_subjects})
    if len(subjects) < 2:
        raise ValueError(
            "Domain adversarial training requires at least two source subjects, "
            f"got {subjects}."
        )

    available = set(labels.astype(str).tolist())
    missing = sorted(set(subjects) - available)
    if missing:
        raise ValueError(f"Source domain subjects are absent from data: {missing}.")

    mapping = {subject: index for index, subject in enumerate(subjects)}
    encoded = np.full(labels.shape, -1, dtype=np.int64)
    for subject, index in mapping.items():
        encoded[labels == subject] = index
    return encoded, mapping


def unpack_supervised_batch(
        batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, (tuple, list)):
        raise TypeError(
            "A supervised batch must be a tuple/list containing (x, y) or "
            f"(x, y, domain), got {type(batch).__name__}."
        )
    if len(batch) == 2:
        inputs, targets = batch
        return inputs, targets, None
    if len(batch) == 3:
        inputs, targets, domains = batch
        return inputs, targets, domains
    raise ValueError(
        "A supervised batch must contain (x, y) or (x, y, domain), "
        f"got {len(batch)} items."
    )
