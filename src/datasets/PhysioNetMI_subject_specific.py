"""Subject-specific views of the PhysioNet motor-imagery dataset.

The regular PhysioNet preprocessor returns one array containing trials from all
subjects.  That representation is useful for cross-subject experiments, but a
subject-dependent BCI benchmark must train an independent model for every
subject.  This module turns the shared preprocessed tensors into validated,
per-subject datasets without changing trial order or mixing subjects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SubjectSpecificDataset:
    """All retained trials for one subject."""

    subject_id: str
    x: np.ndarray
    y: np.ndarray
    class_names: tuple[str, ...]
    class_counts: tuple[int, ...]

    @property
    def n_trials(self) -> int:
        return int(len(self.y))


def group_trials_by_subject(
    x: np.ndarray,
    y: np.ndarray,
    subject_labels: np.ndarray,
    class_names: Iterable[str],
    *,
    expected_num_classes: int = 4,
    min_trials_per_class: int = 5,
    on_incomplete_subject: str = "raise",
) -> tuple[list[SubjectSpecificDataset], list[dict[str, Any]]]:
    """Return one validated dataset per subject.

    Parameters
    ----------
    on_incomplete_subject:
        ``"raise"`` fails immediately. ``"skip"`` omits an incomplete subject
        and returns an audit record describing why it was skipped.
    """

    x = np.asarray(x)
    y = np.asarray(y, dtype=np.int64)
    subject_labels = np.asarray(subject_labels, dtype=np.str_)
    names = tuple(str(name) for name in class_names)
    policy = str(on_incomplete_subject).strip().lower()
    if policy not in {"raise", "skip"}:
        raise ValueError("on_incomplete_subject must be 'raise' or 'skip'.")
    if len(x) != len(y) or len(subject_labels) != len(y):
        raise ValueError(
            "x, y, and subject_labels must contain the same number of trials: "
            f"got {len(x)}, {len(y)}, and {len(subject_labels)}."
        )
    if len(names) != int(expected_num_classes):
        raise ValueError(
            f"Expected {expected_num_classes} classes, got {len(names)}: {names}."
        )
    if min_trials_per_class < 1:
        raise ValueError("min_trials_per_class must be at least 1.")
    if not np.isfinite(x).all():
        raise ValueError("Subject-specific input contains NaN or Inf values.")
    expected_labels = np.arange(len(names), dtype=np.int64)
    unexpected_labels = sorted(set(y.tolist()) - set(expected_labels.tolist()))
    if unexpected_labels:
        raise ValueError(f"Unexpected encoded class labels: {unexpected_labels}.")

    datasets: list[SubjectSpecificDataset] = []
    skipped: list[dict[str, Any]] = []
    for subject_id in sorted(set(subject_labels.tolist())):
        indices = np.flatnonzero(subject_labels == subject_id)
        subject_y = y[indices]
        counts = np.bincount(subject_y, minlength=len(names)).astype(int)
        problems = []
        missing = [names[index] for index, count in enumerate(counts) if count == 0]
        if missing:
            problems.append(f"missing classes: {', '.join(missing)}")
        too_small = [
            f"{names[index]}={count}"
            for index, count in enumerate(counts)
            if 0 < count < min_trials_per_class
        ]
        if too_small:
            problems.append(
                "fewer than "
                f"{min_trials_per_class} trials: {', '.join(too_small)}"
            )
        if problems:
            audit = {
                "subject_id": subject_id,
                "n_trials": int(len(indices)),
                "class_counts": {
                    name: int(count) for name, count in zip(names, counts)
                },
                "reason": "; ".join(problems),
            }
            if policy == "raise":
                raise ValueError(
                    f"Incomplete subject {subject_id}: {audit['reason']}."
                )
            skipped.append(audit)
            continue

        datasets.append(
            SubjectSpecificDataset(
                subject_id=subject_id,
                x=x[indices],
                y=subject_y,
                class_names=names,
                class_counts=tuple(int(count) for count in counts),
            )
        )

    if not datasets:
        raise RuntimeError("No complete subject-specific datasets remain.")
    return datasets, skipped


def load_subject_specific_datasets(
    data_config: dict[str, Any],
    cache_dir: str | Path,
) -> tuple[list[SubjectSpecificDataset], list[dict[str, Any]]]:
    """Preprocess PhysioNet once, cache it, then construct per-subject views."""

    # Local import avoids making this lightweight grouping module depend on the
    # complete PyTorch training stack during unit tests.
    from src.training.train import load_or_preprocess_dataset

    x, y, subject_labels, class_names = load_or_preprocess_dataset(
        data_config,
        Path(cache_dir),
    )
    return group_trials_by_subject(
        x,
        y,
        subject_labels,
        class_names,
        expected_num_classes=int(data_config.get("expected_num_classes", 4)),
        min_trials_per_class=int(data_config.get("min_trials_per_class", 5)),
        on_incomplete_subject=str(
            data_config.get("on_incomplete_subject", "raise")
        ),
    )
