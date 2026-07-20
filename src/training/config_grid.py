from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any


TIME_PAIR_KEYS = {"epoch_slice", "segment_slice"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_pair(value: Any, *, allow_none_second: bool = False) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    first, second = value
    return _is_number(first) and (
        _is_number(second) or (allow_none_second and second is None)
    )


def normalize_filter_bank(filter_bank: Any) -> list[list[float]]:
    if not isinstance(filter_bank, (list, tuple)) or not filter_bank:
        raise ValueError("data.filter_bank must be a non-empty list.")

    normalized = []
    for band in filter_bank:
        if not _is_pair(band):
            raise ValueError(
                "Each filter bank item must be [low_freq, high_freq], "
                f"got {band!r}."
            )
        low_freq, high_freq = float(band[0]), float(band[1])
        if not 0.0 <= low_freq < high_freq:
            raise ValueError(
                "Each filter band must satisfy 0 <= low_freq < high_freq, "
                f"got {band!r}."
            )
        normalized.append([low_freq, high_freq])
    return normalized


def is_filter_bank_scheme(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(_is_pair(item) for item in value)
    )


def normalize_epoch_slice(value: Any) -> list[float]:
    if not _is_pair(value):
        raise ValueError(
            "data.epoch_slice must be [epoch_tmin, epoch_tmax], "
            f"got {value!r}."
        )
    epoch_tmin, epoch_tmax = float(value[0]), float(value[1])
    if epoch_tmax <= epoch_tmin:
        raise ValueError(
            "data.epoch_slice requires epoch_tmax > epoch_tmin, "
            f"got {value!r}."
        )
    return [epoch_tmin, epoch_tmax]


def normalize_segment_slice(value: Any) -> list[float | None]:
    if not _is_pair(value, allow_none_second=True):
        raise ValueError(
            "data.segment_slice must be [segment_duration, stride_duration], "
            "where stride_duration may be null; "
            f"got {value!r}."
        )
    segment_duration = float(value[0])
    stride_duration = None if value[1] is None else float(value[1])
    if segment_duration <= 0.0:
        raise ValueError(
            "data.segment_slice segment_duration must be positive, "
            f"got {value!r}."
        )
    if stride_duration is not None and stride_duration <= 0.0:
        raise ValueError(
            "data.segment_slice stride_duration must be positive or null, "
            f"got {value!r}."
        )
    return [segment_duration, stride_duration]


def grid_values(key: str, value: Any) -> list[Any]:
    if key == "filter_bank":
        if is_filter_bank_scheme(value):
            return [normalize_filter_bank(value)]
        if (
            isinstance(value, (list, tuple))
            and bool(value)
            and all(is_filter_bank_scheme(candidate) for candidate in value)
        ):
            return [normalize_filter_bank(candidate) for candidate in value]
        raise ValueError(
            "data.filter_bank must be one filter-bank scheme such as "
            "[[8, 13], [13, 30]], or a list of schemes such as "
            "[[[8, 30]], [[8, 13], [13, 30]]]."
        )

    if key in TIME_PAIR_KEYS:
        normalizer = (
            normalize_epoch_slice if key == "epoch_slice" else normalize_segment_slice
        )
        pair_allows_none = key == "segment_slice"
        if _is_pair(value, allow_none_second=pair_allows_none):
            return [normalizer(value)]
        if (
            isinstance(value, (list, tuple))
            and bool(value)
            and all(
                _is_pair(candidate, allow_none_second=pair_allows_none)
                for candidate in value
            )
        ):
            return [normalizer(candidate) for candidate in value]
        return [normalizer(value)]

    if isinstance(value, list):
        return value
    return [value]


def expand_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    if not section:
        return [{}]

    keys = list(section)
    value_lists = [grid_values(key, section[key]) for key in keys]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def normalize_data_time_config(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize old scalar time fields into the two paired data fields."""
    normalized = deepcopy(data_cfg)
    dataset = str(normalized.get("dataset", "physionet_mi")).lower().replace("-", "_")
    default_epoch_slice = [0.0, 4.0] if dataset in {
        "bnci2014_001",
        "bnci2014001",
        "bci_iv_2a",
        "bci_competition_iv_2a",
        "bciciv_2a",
        "bcic_iv_2a",
    } else [-2.0, 4.0]

    old_epoch_keys = {"epoch_tmin", "epoch_tmax"} & normalized.keys()
    if "epoch_slice" in normalized and old_epoch_keys:
        raise ValueError(
            "Use either data.epoch_slice or data.epoch_tmin/data.epoch_tmax, "
            "not both."
        )
    if "epoch_slice" in normalized:
        epoch_slice = normalize_epoch_slice(normalized["epoch_slice"])
    else:
        epoch_slice = normalize_epoch_slice(
            [
                normalized.get("epoch_tmin", default_epoch_slice[0]),
                normalized.get("epoch_tmax", default_epoch_slice[1]),
            ]
        )

    old_segment_keys = {"segment_duration", "stride_duration"} & normalized.keys()
    if "segment_slice" in normalized and old_segment_keys:
        raise ValueError(
            "Use either data.segment_slice or "
            "data.segment_duration/data.stride_duration, not both."
        )
    if "segment_slice" in normalized:
        segment_slice = normalize_segment_slice(normalized["segment_slice"])
    else:
        segment_slice = normalize_segment_slice(
            [
                normalized.get("segment_duration", 1.0),
                normalized.get("stride_duration", 0.5),
            ]
        )

    for key in ("epoch_tmin", "epoch_tmax", "segment_duration", "stride_duration"):
        normalized.pop(key, None)
    normalized["epoch_slice"] = epoch_slice
    normalized["segment_slice"] = segment_slice
    return normalized


def expand_data_grid(section: dict[str, Any]) -> list[dict[str, Any]]:
    return [normalize_data_time_config(item) for item in expand_grid(section)]
