"""Visualize raw and filter-bank EEG signals grouped by motor task.

Examples:
    python script/visualize_task_signals.py --subjects 1-10
    python script/visualize_task_signals.py --subjects 1 --channels C3,Cz,C4
    python script/visualize_task_signals.py --band-view waveform --show-trials 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid failures when the user's MNE config directory is read-only.
MNE_HOME = Path(tempfile.gettempdir()) / "spdtransformer_mne"
MNE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(MNE_HOME))

DEFAULT_ROOT_DIR = "data/MNE-eegbci-data/files/eegmmidb/1.0.0"
DEFAULT_FILTER_BANK = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
)
DEFAULT_CHANNELS = ("C3", "Cz", "C4")
TASK_ORDER = ("left_hand", "right_hand", "hands", "feet")
RUN_IDS = {4, 6, 8, 10, 12, 14}


def parse_filter_bank(raw: str) -> list[tuple[float, float]]:
    bands = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            low_raw, high_raw = part.split("-", 1)
            low, high = float(low_raw), float(high_raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid band {part!r}; use LOW-HIGH, for example 8-12."
            ) from exc
        if low <= 0 or high <= low:
            raise argparse.ArgumentTypeError(
                f"Invalid band {part!r}; require 0 < LOW < HIGH."
            )
        bands.append((low, high))
    if not bands:
        raise argparse.ArgumentTypeError("Filter bank cannot be empty.")
    return bands


def parse_subjects(raw: str) -> list[int]:
    subjects = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start > end:
                raise argparse.ArgumentTypeError(f"Invalid subject range: {part}")
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    subjects = sorted(set(subjects))
    if not subjects or any(subject < 1 for subject in subjects):
        raise argparse.ArgumentTypeError("Subject ids must be positive integers.")
    return subjects


def normalize_channel_name(name: str) -> str:
    return name.strip().rstrip(".").upper()


def find_edf_metadata(root_dir: Path) -> tuple[list[str], float]:
    import mne

    for edf_file in sorted(root_dir.glob("S*/*.edf")):
        match = re.search(r"R(\d+)", edf_file.name)
        if match is None or int(match.group(1)) not in RUN_IDS:
            continue
        raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
        return raw.ch_names, float(raw.info["sfreq"])
    raise FileNotFoundError(f"No motor-imagery EDF file found under {root_dir}.")


def resolve_channels(
    channel_names: list[str],
    requested_channels: list[str],
) -> tuple[list[int], list[str]]:
    normalized_to_index = {
        normalize_channel_name(channel_name): index
        for index, channel_name in enumerate(channel_names)
    }
    missing = [
        channel for channel in requested_channels
        if normalize_channel_name(channel) not in normalized_to_index
    ]
    if missing:
        available = ", ".join(
            normalize_channel_name(channel_name) for channel_name in channel_names
        )
        raise ValueError(
            f"Unknown channels: {', '.join(missing)}. Available channels: {available}"
        )

    indices = [
        normalized_to_index[normalize_channel_name(channel)]
        for channel in requested_channels
    ]
    display_names = [
        normalize_channel_name(channel_names[index]).title()
        for index in indices
    ]
    return indices, display_names


def sample_class_indices(
    labels: np.ndarray,
    class_names: list[str],
    max_trials_per_class: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices_by_class = {}
    for class_name in class_names:
        indices = np.flatnonzero(labels == class_name)
        if max_trials_per_class is not None and len(indices) > max_trials_per_class:
            indices = np.sort(
                rng.choice(indices, size=max_trials_per_class, replace=False)
            )
        indices_by_class[class_name] = indices
    return indices_by_class


def add_signal_summary(
    axis,
    times: np.ndarray,
    signals: np.ndarray,
    color: str,
    show_trials: int,
) -> None:
    scale = 1e6
    plotted = signals * scale
    for trial in plotted[:show_trials]:
        axis.plot(times, trial, color=color, linewidth=0.5, alpha=0.12)

    mean = plotted.mean(axis=0)
    if len(plotted) > 1:
        sem = plotted.std(axis=0, ddof=1) / np.sqrt(len(plotted))
    else:
        sem = np.zeros_like(mean)
    axis.plot(times, mean, color=color, linewidth=1.2)
    axis.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.22)


def style_axis(axis, tmin: float, tmax: float) -> None:
    axis.axvspan(tmin, min(0.0, tmax), color="#d9d9d9", alpha=0.35, zorder=0)
    if tmax > 0:
        axis.axvspan(max(0.0, tmin), tmax, color="#ffe8b3", alpha=0.18, zorder=0)
    axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.grid(alpha=0.18, linewidth=0.5)
    axis.set_xlim(tmin, tmax)


def plot_task_signals(
    X: np.ndarray,
    labels: np.ndarray,
    channel_indices: list[int],
    channel_names: list[str],
    sfreq: float,
    tmin: float,
    tmax: float,
    filter_bank: list[tuple[float, float]],
    band_view: str,
    max_trials_per_class: int | None,
    show_trials: int,
    seed: int,
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert

    from src.datasets.PhysioNetMI_preprocess import bandpass_filter

    class_names = [
        class_name for class_name in TASK_ORDER if np.any(labels == class_name)
    ]
    class_names.extend(
        sorted(set(labels.tolist()) - set(class_names))
    )
    indices_by_class = sample_class_indices(
        labels, class_names, max_trials_per_class, seed
    )
    sampled_indices = np.sort(
        np.concatenate(list(indices_by_class.values()))
    )
    local_index = {
        dataset_index: index for index, dataset_index in enumerate(sampled_indices)
    }
    local_indices_by_class = {
        class_name: np.array(
            [local_index[index] for index in indices], dtype=np.int64
        )
        for class_name, indices in indices_by_class.items()
    }

    selected = X[
        sampled_indices[:, np.newaxis],
        np.asarray(channel_indices)[np.newaxis, :],
        :,
    ]
    expected_samples = int(round((tmax - tmin) * sfreq))
    if selected.shape[-1] != expected_samples:
        raise ValueError(
            f"Epoch has {selected.shape[-1]} samples, but tmin/tmax/sfreq imply "
            f"{expected_samples} samples."
        )
    times = tmin + np.arange(selected.shape[-1]) / sfreq

    n_rows = 1 + len(filter_bank)
    n_cols = len(class_names)
    figures = []
    axes_by_channel = []
    colors = plt.get_cmap("tab10").colors

    for channel_name in channel_names:
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 2.25 * n_rows),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        figures.append(figure)
        axes_by_channel.append(axes)
        figure.suptitle(
            f"{channel_name}: task-grouped raw and filter-bank signals",
            fontsize=15,
        )

    for channel_position, axes in enumerate(axes_by_channel):
        for class_position, class_name in enumerate(class_names):
            axis = axes[0, class_position]
            indices = local_indices_by_class[class_name]
            add_signal_summary(
                axis,
                times,
                selected[indices, channel_position, :],
                colors[class_position % len(colors)],
                show_trials,
            )
            axis.set_title(
                f"{class_name.replace('_', ' ')} (n={len(indices)})",
                fontsize=11,
            )
            style_axis(axis, tmin, tmax)
            if class_position == 0:
                axis.set_ylabel("Raw\namplitude (uV)")

    for row, (low_freq, high_freq) in enumerate(filter_bank, start=1):
        filtered = bandpass_filter(
            selected,
            sfreq=sfreq,
            low_freq=low_freq,
            high_freq=high_freq,
        )
        if band_view == "envelope":
            filtered = np.abs(hilbert(filtered, axis=-1))

        for channel_position, axes in enumerate(axes_by_channel):
            for class_position, class_name in enumerate(class_names):
                axis = axes[row, class_position]
                indices = local_indices_by_class[class_name]
                add_signal_summary(
                    axis,
                    times,
                    filtered[indices, channel_position, :],
                    colors[class_position % len(colors)],
                    show_trials,
                )
                style_axis(axis, tmin, tmax)
                if class_position == 0:
                    quantity = "envelope" if band_view == "envelope" else "amplitude"
                    axis.set_ylabel(
                        f"{low_freq:g}-{high_freq:g} Hz\n{quantity} (uV)"
                    )
                if row == n_rows - 1:
                    axis.set_xlabel("Time from task onset (s)")
        del filtered

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for figure, channel_name in zip(figures, channel_names):
        safe_channel = re.sub(r"[^A-Za-z0-9_-]+", "_", channel_name)
        output_path = output_dir / f"task_signals_{safe_channel}.png"
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot complete rest+task epochs and filter-bank signals, grouped by "
            "PhysioNet motor-imagery task."
        )
    )
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--tmin", type=float, default=-2.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=None,
        help="Optional subject ids/ranges, for example 1-10,15. Default: all.",
    )
    parser.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        help="Comma-separated EEG channels. Default: C3,Cz,C4.",
    )
    parser.add_argument(
        "--filter-bank",
        type=parse_filter_bank,
        default=list(DEFAULT_FILTER_BANK),
        help="Comma-separated bands, for example 8-12,12-16,16-20.",
    )
    parser.add_argument(
        "--band-view",
        choices=("envelope", "waveform"),
        default="envelope",
        help=(
            "envelope shows class-level band energy without phase cancellation; "
            "waveform shows the signed band-pass signal."
        ),
    )
    parser.add_argument(
        "--max-trials-per-class",
        type=int,
        default=400,
        help="Randomly sample at most this many trials per task; use 0 for all.",
    )
    parser.add_argument(
        "--show-trials",
        type=int,
        default=0,
        help="Overlay this many individual trials behind each class mean.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="experiments/visualizations/task_signals",
    )
    parser.add_argument("--imaged", type=bool, default=True)
    parser.add_argument("--executed", type=bool, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    if args.tmax <= args.tmin:
        raise ValueError("tmax must be greater than tmin.")
    if args.max_trials_per_class < 0:
        raise ValueError("max-trials-per-class must be non-negative.")
    if args.show_trials < 0:
        raise ValueError("show-trials must be non-negative.")

    root_dir = Path(args.root_dir)
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir

    channel_names, sfreq = find_edf_metadata(root_dir)
    requested_channels = [
        channel.strip() for channel in args.channels.split(",") if channel.strip()
    ]
    channel_indices, display_channel_names = resolve_channels(
        channel_names, requested_channels
    )

    nyquist = sfreq / 2.0
    invalid_bands = [
        band for band in args.filter_bank if band[1] >= nyquist
    ]
    if invalid_bands:
        raise ValueError(
            f"Band upper bounds must be below Nyquist ({nyquist:g} Hz): "
            f"{invalid_bands}"
        )

    from src.datasets.PhysioNetMI_preprocess import build_dataset

    dataset = build_dataset(
        root_dir,
        tmin=args.tmin,
        tmax=args.tmax,
        subjects=args.subjects,
        imaged=args.imaged,
        executed=args.executed,
    )
    print(f"Dataset X shape: {dataset['X'].shape}")
    print(f"Tasks: {sorted(set(dataset['y'].tolist()))}")
    print(f"Channels: {display_channel_names}; sampling frequency: {sfreq:g} Hz")

    max_trials = args.max_trials_per_class or None
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_paths = plot_task_signals(
        X=dataset["X"],
        labels=dataset["y"],
        channel_indices=channel_indices,
        channel_names=display_channel_names,
        sfreq=sfreq,
        tmin=args.tmin,
        tmax=args.tmax,
        filter_bank=args.filter_bank,
        band_view=args.band_view,
        max_trials_per_class=max_trials,
        show_trials=args.show_trials,
        seed=args.seed,
        output_dir=output_dir,
    )
    for output_path in output_paths:
        print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
