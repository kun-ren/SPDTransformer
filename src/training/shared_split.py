from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def _labels_hash(y: np.ndarray) -> str:
    labels = np.asarray(y, dtype=np.int64)
    return hashlib.sha1(labels.tobytes()).hexdigest()


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


def load_or_create_split_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
    split_file: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_file is None:
        return create_split_indices(y, test_size, val_size, seed)

    path = Path(split_file)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        expected_hash = _labels_hash(y)
        if payload.get("n_samples") != len(y):
            raise ValueError(
                f"Split file {path} was built for {payload.get('n_samples')} samples, "
                f"but current data has {len(y)} samples."
            )
        if payload.get("labels_hash") != expected_hash:
            raise ValueError(
                f"Split file {path} does not match the current label order. "
                "Regenerate it with the same dataset and preprocessing order."
            )

        return (
            np.asarray(payload["train_idx"], dtype=np.int64),
            np.asarray(payload["val_idx"], dtype=np.int64),
            np.asarray(payload["test_idx"], dtype=np.int64),
        )

    train_idx, val_idx, test_idx = create_split_indices(
        y=y,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": int(len(y)),
        "labels_hash": _labels_hash(y),
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
