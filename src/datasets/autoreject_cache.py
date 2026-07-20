from __future__ import annotations

import hashlib
import json
import os
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


AUTOREJECT_MASK_CACHE_VERSION = 1


def installed_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _labels_hash(labels: Sequence[Any]) -> str:
    return _json_hash([str(label) for label in labels])


def source_file_fingerprints(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    fingerprints = []
    for path_value in paths:
        path = Path(path_value).resolve()
        stat = path.stat()
        fingerprints.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return fingerprints


def build_autoreject_cache_metadata(
    *,
    subject_dir: str | Path,
    source_files: Sequence[str | Path],
    labels: Sequence[Any],
    epoch_shape: Sequence[int],
    channel_names: Sequence[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "cache_version": AUTOREJECT_MASK_CACHE_VERSION,
        "subject_dir": str(Path(subject_dir).resolve()),
        "source_files": source_file_fingerprints(source_files),
        "labels_hash": _labels_hash(labels),
        "n_epochs": int(len(labels)),
        "epoch_shape": [int(size) for size in epoch_shape],
        "channel_names": [str(name) for name in channel_names],
        "settings": settings,
    }
    return json.loads(json.dumps(metadata, sort_keys=True, default=str))


def autoreject_cache_path(
    cache_dir: str | Path,
    subject_name: str,
    metadata: dict[str, Any],
) -> Path:
    safe_subject = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(subject_name)
    )
    return Path(cache_dir) / f"{safe_subject}_{_json_hash(metadata)}.npz"


def load_or_compute_autoreject_mask(
    *,
    cache_dir: str | Path | None,
    subject_name: str,
    metadata: dict[str, Any],
    compute_mask: Callable[[], np.ndarray],
    force_rebuild: bool = False,
) -> tuple[np.ndarray, bool, Path | None]:
    """Return a validated mask and persist it atomically when caching is enabled."""
    expected_epochs = int(metadata["n_epochs"])
    cache_path = None
    if cache_dir not in {None, "", False}:
        cache_path = autoreject_cache_path(cache_dir, subject_name, metadata)

    if cache_path is not None and cache_path.exists() and not force_rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                cached_metadata = json.loads(str(payload["metadata_json"].item()))
                keep_mask = np.asarray(payload["keep_mask"], dtype=bool)
            if cached_metadata == metadata and keep_mask.shape == (expected_epochs,):
                return keep_mask, True, cache_path
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    keep_mask = np.asarray(compute_mask(), dtype=bool)
    if keep_mask.shape != (expected_epochs,):
        raise ValueError(
            "AutoReject keep mask shape mismatch: expected "
            f"({expected_epochs},), got {keep_mask.shape}."
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with tmp_path.open("wb") as handle:
                np.savez(
                    handle,
                    keep_mask=keep_mask.astype(np.uint8, copy=False),
                    metadata_json=np.asarray(
                        json.dumps(metadata, sort_keys=True, default=str),
                        dtype=np.str_,
                    ),
                )
            tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return keep_mask, False, cache_path
