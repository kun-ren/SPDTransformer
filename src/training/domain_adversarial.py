from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


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
