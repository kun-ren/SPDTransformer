from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _labels_hash(y: np.ndarray) -> str:
    labels = np.asarray(y, dtype=np.int64)
    return hashlib.sha1(labels.tobytes()).hexdigest()


def _subjects_hash(subjects: np.ndarray | None) -> str | None:
    if subjects is None:
        return None
    subject_labels = np.asarray(subjects, dtype=np.str_)
    return hashlib.sha1("\n".join(subject_labels.tolist()).encode("utf-8")).hexdigest()


def create_split_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=y[train_val_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def create_subject_split_indices(
    y: np.ndarray,
    subjects: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int64)
    subjects = np.asarray(subjects, dtype=np.str_)
    if len(subjects) != len(y):
        raise ValueError(
            f"subjects length ({len(subjects)}) must match labels length ({len(y)})."
        )

    indices = np.arange(len(y))
    train_val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_val_idx, test_idx = next(
        train_val_splitter.split(indices, y, groups=subjects)
    )

    relative_val_size = val_size / (1.0 - test_size)
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=seed,
    )
    train_rel_idx, val_rel_idx = next(
        val_splitter.split(
            train_val_idx,
            y[train_val_idx],
            groups=subjects[train_val_idx],
        )
    )
    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def load_or_create_split_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    split_file: str | Path | None,
    subjects: np.ndarray | None = None,
    allow_subject_overlap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_strategy = "epoch" if allow_subject_overlap else "subject"
    if not allow_subject_overlap and subjects is None:
        raise ValueError(
            "subjects must be provided when allow_subject_overlap is False."
        )

    if split_file is None:
        if allow_subject_overlap:
            return create_split_indices(y, test_size, val_size, seed)
        return create_subject_split_indices(y, subjects, test_size, val_size, seed)

    path = Path(split_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        expected_hash = _labels_hash(y)
        expected_subjects_hash = _subjects_hash(subjects)
        payload_strategy = payload.get("split_strategy", "epoch")
        if payload.get("n_samples") != len(y):
            raise ValueError(
                f"Split file {path} was built for {payload.get('n_samples')} samples, "
                f"but current data has {len(y)} samples."
            )
        if payload_strategy != split_strategy:
            raise ValueError(
                f"Split file {path} uses split_strategy={payload_strategy!r}, "
                f"but current config requests {split_strategy!r}. "
                "Use a different split_file or delete the old split file."
            )
        if payload.get("labels_hash") != expected_hash:
            raise ValueError(
                f"Split file {path} does not match the current label order. "
                "Regenerate it with the same dataset and preprocessing order."
            )
        if split_strategy == "subject" and payload.get("subjects_hash") != expected_subjects_hash:
            raise ValueError(
                f"Split file {path} does not match the current subject order. "
                "Regenerate it with the same dataset and preprocessing order."
            )

        return (
            np.asarray(payload["train_idx"], dtype=np.int64),
            np.asarray(payload["val_idx"], dtype=np.int64),
            np.asarray(payload["test_idx"], dtype=np.int64),
        )

    if allow_subject_overlap:
        train_idx, val_idx, test_idx = create_split_indices(
            y=y,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )
    else:
        train_idx, val_idx, test_idx = create_subject_split_indices(
            y=y,
            subjects=subjects,
            test_size=test_size,
            val_size=val_size,
            seed=seed,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": int(len(y)),
        "labels_hash": _labels_hash(y),
        "subjects_hash": _subjects_hash(subjects),
        "split_strategy": split_strategy,
        "allow_subject_overlap": bool(allow_subject_overlap),
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "train_idx": train_idx.astype(int).tolist(),
        "val_idx": val_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return train_idx, val_idx, test_idx
