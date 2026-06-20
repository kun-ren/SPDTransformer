"""Plot task-related scalp topomaps for motor imagery EEG.

This script generates one clean publication-style topomap per motor imagery class.
It plots task-related ERD/ERS relative to a fixed pre-task rest baseline:

    ERD/ERS (%) = (P_task - P_rest) / P_rest * 100

For each class:
    - all electrode positions are shown
    - the top-k most task-related electrodes are highlighted and labeled
    - no extra in-figure annotation is added except the electrode labels and colorbar

Additional explanations are written separately to:
    - summary.json
    - top_electrodes.csv
    - captions.txt

Each topomap is computed from a fixed pre-task rest window and a task window:
    - rest: -2 to 0 s
    - task: 0 to 3.5 s

The default frequency band is 8-30 Hz.

Example:
    python script/plot_scalp_topology_rest_task.py
    python script/plot_scalp_topology_rest_task.py --subjects 1-20
    python script/plot_scalp_topology_rest_task.py --classes left_hand,right_hand
    python script/plot_scalp_topology_rest_task.py --low-freq 8 --high-freq 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid failures when MNE writes config under a restricted home folder
MNE_HOME = Path(tempfile.gettempdir()) / "spdtransformer_mne"
MNE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(MNE_HOME))

DEFAULT_ROOT_DIR = PROJECT_ROOT / "data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "figures" / "task_related_topomap"

TASK_ORDER = ("left_hand", "right_hand", "hands", "feet")
REST_WINDOW = (-2.0, 0.0)


@dataclass
class TrialPower:
    rest: np.ndarray  # shape: (n_channels,)
    task: np.ndarray  # shape: (n_channels,)


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


def parse_classes(raw: str) -> list[str]:
    classes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(classes) - set(TASK_ORDER))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown classes: {', '.join(unknown)}. "
            f"Valid classes: {', '.join(TASK_ORDER)}"
        )
    return classes


def task_types_for_classes(classes: list[str]) -> tuple[str, ...]:
    task_types = []
    class_set = set(classes)
    if class_set & {"left_hand", "right_hand"}:
        task_types.append("unilateral_fist")
    if class_set & {"hands", "feet"}:
        task_types.append("both")
    return tuple(task_types)


def sensorimotor_roi_indices(ch_names: list[str]) -> np.ndarray:
    """Return indices of standard sensorimotor electrodes."""
    roi_names = {
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    }

    indices = [
        idx for idx, name in enumerate(ch_names)
        if name in roi_names
    ]

    if not indices:
        raise RuntimeError(
            "No sensorimotor ROI electrodes were found. "
            "Please check channel names after EEGBCI standardization."
        )

    return np.asarray(indices, dtype=int)


def choose_top_electrodes(
    values: np.ndarray,
    ch_names: list[str],
    top_k: int,
    mode: str = "most_negative",
    restrict_to_sensorimotor: bool = True,
) -> np.ndarray:
    """Choose top task-related electrodes.

    values:
        Usually ERD/ERS (%) for each channel.

    mode:
        "most_negative":
            Select electrodes with strongest ERD.

        "absolute":
            Select electrodes with strongest absolute task-related modulation.
    """
    if top_k <= 0:
        return np.array([], dtype=int)

    if restrict_to_sensorimotor:
        candidate_idx = sensorimotor_roi_indices(ch_names)
    else:
        candidate_idx = np.arange(len(ch_names), dtype=int)

    candidate_values = values[candidate_idx]

    if mode == "most_negative":
        local_order = np.argsort(candidate_values)
    elif mode == "absolute":
        local_order = np.argsort(-np.abs(candidate_values))
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    selected_local = local_order[: min(top_k, len(local_order))]
    selected_global = candidate_idx[selected_local]

    return np.asarray(selected_global, dtype=int)

def parse_channels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    channels = [item.strip() for item in raw.split(",") if item.strip()]
    return channels or None


def first_available_edf_file(root_dir: Path, subjects: list[int] | None) -> Path:
    subject_names = None
    if subjects is not None:
        subject_names = {f"S{subject:03d}" for subject in subjects}

    for subject_dir in sorted(root_dir.glob("S*")):
        if subject_names is not None and subject_dir.name not in subject_names:
            continue
        for edf_file in sorted(subject_dir.glob("*.edf")):
            return edf_file
    raise FileNotFoundError(f"No EDF file found under {root_dir}")


def build_topomap_info(
    root_dir: Path,
    subjects: list[int] | None,
    channels: list[str] | None,
):
    import mne

    from src.datasets.PhysioNetMI_preprocess import pick_raw_channels

    edf_file = first_available_edf_file(root_dir, subjects)
    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)

    # Standardize EEGBCI channel names
    try:
        mne.datasets.eegbci.standardize(raw)
    except Exception as exc:
        print(f"[WARN] Could not standardize EEGBCI channel names for {edf_file}: {exc}")

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    raw = pick_raw_channels(raw, channels=channels)

    valid_indices = []
    for idx, channel in enumerate(raw.info["chs"]):
        loc = channel["loc"][:3]
        if np.isfinite(loc).all() and not np.allclose(loc, 0.0):
            valid_indices.append(idx)

    if not valid_indices:
        raise RuntimeError("No EEG channels with valid montage positions were found.")

    if len(valid_indices) != len(raw.ch_names):
        missing = sorted(
            set(raw.ch_names) - {raw.ch_names[index] for index in valid_indices}
        )
        raise RuntimeError(
            "Some loaded EEG channels do not have valid montage positions: "
            f"{', '.join(missing)}. Use --channels to restrict the topomap channels."
        )

    return raw.info.copy()


def collect_class_power(
    root_dir: Path,
    subjects: list[int] | None,
    classes: list[str],
    low_freq: float,
    high_freq: float,
    task_window: tuple[float, float],
    channels: list[str] | None,
    reject_threshold_uv: float | None,
    eps: float,
) -> dict[str, list[TrialPower]]:
    from src.datasets.PhysioNetMI_preprocess import build_dataset

    tmin = min(REST_WINDOW[0], task_window[0])
    tmax = max(REST_WINDOW[1], task_window[1])

    dataset = build_dataset(
        root_dir=root_dir,
        tmin=tmin,
        tmax=tmax,
        subjects=subjects,
        imaged=True,
        executed=False,
        task_types=task_types_for_classes(classes),
        low_freq=low_freq,
        high_freq=high_freq,
        channels=channels,
        reject_threshold_uv=reject_threshold_uv,
    )

    X = dataset["X"]
    labels = dataset["y"]
    sfreq = 160.0
    class_power: dict[str, list[TrialPower]] = defaultdict(list)

    rest_start = int(round((REST_WINDOW[0] - tmin) * sfreq))
    rest_end = int(round((REST_WINDOW[1] - tmin) * sfreq))
    task_start = int(round((task_window[0] - tmin) * sfreq))
    task_end = int(round((task_window[1] - tmin) * sfreq))

    if rest_start < 0 or rest_end > X.shape[-1] or rest_end <= rest_start:
        raise ValueError(f"Invalid rest window {REST_WINDOW} for epoch [{tmin}, {tmax}].")
    if task_start < 0 or task_end > X.shape[-1] or task_end <= task_start:
        raise ValueError(f"Invalid task window {task_window} for epoch [{tmin}, {tmax}].")

    for epoch, label in zip(X, labels):
        if label not in classes:
            continue
        rest = np.mean(epoch[:, rest_start:rest_end] ** 2, axis=1) + eps
        task = np.mean(epoch[:, task_start:task_end] ** 2, axis=1) + eps
        class_power[str(label)].append(TrialPower(rest=rest, task=task))

    return class_power


def compute_class_statistics(trials: list[TrialPower]) -> dict[str, np.ndarray]:
    """Compute class-wise mean power and task-related ERD/ERS."""
    rest_power = np.stack([trial.rest for trial in trials], axis=0)
    task_power = np.stack([trial.task for trial in trials], axis=0)
    mean_rest = rest_power.mean(axis=0)
    mean_task = task_power.mean(axis=0)
    rest_db = 10.0 * np.log10(mean_rest)
    task_db = 10.0 * np.log10(mean_task)
    diff_db = task_db - rest_db
    erd_percent = (mean_task - mean_rest) / mean_rest * 100.0

    return {
        "mean_rest": mean_rest,
        "mean_task": mean_task,
        "rest_db": rest_db,
        "task_db": task_db,
        "diff_db": diff_db,
        "erd_percent": erd_percent,
    }


def selective_names(ch_names: list[str], highlight_idx: np.ndarray) -> list[str]:
    highlight_set = set(int(i) for i in highlight_idx)
    return [name if idx in highlight_set else "" for idx, name in enumerate(ch_names)]


def plot_task_related_topomap(
    values: np.ndarray,
    info,
    highlight_idx: np.ndarray,
    output_file: Path,
    colorbar_label: str,
    vlim: tuple[float, float],
    cmap: str = "RdBu_r",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    ch_names = list(info["ch_names"])

    fig, axis = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)

    im, _ = mne.viz.plot_topomap(
        values,
        info,
        ch_type="eeg",
        sensors=True,
        names=None,
        contours=0,
        axes=axis,
        show=False,
        vlim=vlim,
        cmap=cmap,
    )

    # Get 2D electrode positions in the same coordinate system as plot_topomap.
    try:
        pos = mne.channels.layout._find_topomap_coords(
            info,
            picks=np.arange(len(ch_names)),
        )
    except Exception:
        layout = mne.channels.find_layout(info, ch_type="eeg")
        pos = layout.pos[:, :2]

    # Highlight selected electrodes without drawing any connecting lines.
    for idx in highlight_idx:
        idx = int(idx)
        x, y = pos[idx]

        axis.scatter(
            x,
            y,
            s=95,
            facecolors="white",
            edgecolors="black",
            linewidths=1.3,
            zorder=10,
        )

        axis.text(
            x + 0.012,
            y + 0.012,
            ch_names[idx],
            fontsize=8,
            ha="left",
            va="center",
            color="black",
            zorder=11,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                boxstyle="round,pad=0.10",
            ),
        )

    # No in-figure title for publication style.
    axis.set_title("")

    cbar = fig.colorbar(im, ax=axis, shrink=0.82)
    cbar.set_label(colorbar_label)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_top_electrodes_csv(
    output_file: Path,
    class_results: dict[str, dict],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class",
                "rank",
                "electrode",
                "erd_percent",
                "task_minus_rest_db",
            ]
        )

        for class_name, result in class_results.items():
            top_idx = result["top_idx"]
            ch_names = result["ch_names"]
            erd_percent = result["erd_percent"]
            diff_db = result["diff_db"]

            for rank, idx in enumerate(top_idx, start=1):
                writer.writerow(
                    [
                        class_name,
                        rank,
                        ch_names[int(idx)],
                        f"{erd_percent[int(idx)]:.6f}",
                        f"{diff_db[int(idx)]:.6f}",
                    ]
                )


def write_captions_txt(
    output_file: Path,
    class_results: dict[str, dict],
    low_freq: float,
    high_freq: float,
    task_window: tuple[float, float],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("Figure captions / notes")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"All figures show task-related scalp topography in the {low_freq:g}-{high_freq:g} Hz band."
    )
    lines.append(
        f"Rest window: [{REST_WINDOW[0]:g}, {REST_WINDOW[1]:g}] s relative to task onset."
    )
    lines.append(
        f"Task window: [{task_window[0]:g}, {task_window[1]:g}] s relative to task onset."
    )
    lines.append("Metric: ERD/ERS (%) = (P_task - P_rest) / P_rest * 100.")
    lines.append("Negative values indicate event-related desynchronization (ERD).")
    lines.append("Positive values indicate event-related synchronization (ERS).")
    lines.append("All electrode locations are shown as black dots.")
    lines.append("Highlighted and labeled electrodes are the top task-related electrodes.")
    lines.append("")

    for class_name, result in class_results.items():
        pretty_name = class_name.replace("_", " ").title()
        top_electrodes = [result["ch_names"][int(i)] for i in result["top_idx"]]
        n_trials = result["n_trials"]

        lines.append(f"{pretty_name}")
        lines.append("-" * len(pretty_name))
        lines.append(f"Number of valid trials: {n_trials}")
        lines.append(
            f"Most task-related electrodes (strongest ERD): {', '.join(top_electrodes)}"
        )
        lines.append(
            f"Suggested caption: Task-related scalp topography for {pretty_name} motor imagery. "
            f"Colors indicate ERD/ERS in the {low_freq:g}-{high_freq:g} Hz band relative to rest. "
            f"Labeled electrodes denote the most task-related sites."
        )
        lines.append("")

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot publication-style task-related topomaps for motor imagery EEG."
    )

    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=None,
        help="Subject ids, e.g. 1,2,3 or 1-20. Default: all subjects.",
    )
    parser.add_argument(
        "--classes",
        type=parse_classes,
        default=list(TASK_ORDER),
        help="Comma-separated class names. Default: left_hand,right_hand,hands,feet.",
    )

    parser.add_argument("--low-freq", type=float, default=8.0)
    parser.add_argument("--high-freq", type=float, default=30.0)
    parser.add_argument(
        "--channels",
        type=parse_channels,
        default=None,
        help=(
            "Optional comma-separated EEG channels. Default: all EEG channels "
            "provided by the shared PhysioNetMI preprocessing interface."
        ),
    )
    parser.add_argument(
        "--reject-threshold-uv",
        type=float,
        default=None,
        help=(
            "Optional epoch-level peak-to-peak rejection threshold in microvolts. "
            "This is passed through to build_dataset."
        ),
    )

    parser.add_argument(
        "--task-window",
        type=float,
        nargs=2,
        default=(0.0, 3.5),
        metavar=("START", "END"),
        help="Task window in seconds relative to task onset. Default: 0 3.5",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of most task-related electrodes to highlight. Default: 5",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("most_negative", "absolute"),
        default="most_negative",
        help=(
            "How to choose task-related electrodes. "
            "'most_negative' picks strongest ERD electrodes, "
            "'absolute' picks strongest absolute modulation."
        ),
    )

    parser.add_argument("--eps", type=float, default=1e-20)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.root_dir.is_absolute():
        args.root_dir = PROJECT_ROOT / args.root_dir
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir

    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    task_window = (float(args.task_window[0]), float(args.task_window[1]))

    class_power = collect_class_power(
        root_dir=args.root_dir,
        subjects=args.subjects,
        classes=args.classes,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        task_window=task_window,
        channels=args.channels,
        reject_threshold_uv=args.reject_threshold_uv,
        eps=args.eps,
    )
    info = build_topomap_info(
        root_dir=args.root_dir,
        subjects=args.subjects,
        channels=args.channels,
    )

    class_results: dict[str, dict] = {}
    all_effect_values = []

    for class_name in args.classes:
        trials = class_power.get(class_name, [])
        if not trials:
            print(f"[WARN] No valid trials for class {class_name!r}; skipping.")
            continue

        stats = compute_class_statistics(trials)
        if len(stats["erd_percent"]) != len(info["ch_names"]):
            raise RuntimeError(
                "Topomap metadata channel count does not match loaded epochs: "
                f"{len(info['ch_names'])} info channels vs "
                f"{len(stats['erd_percent'])} epoch channels."
            )
        top_idx = choose_top_electrodes(
            values=stats["erd_percent"],
            ch_names=info["ch_names"],
            top_k=args.top_k,
            mode=args.selection_mode,
            restrict_to_sensorimotor=True,
        )

        class_results[class_name] = {
            "n_trials": len(trials),
            "ch_names": list(info["ch_names"]),
            "mean_rest": stats["mean_rest"],
            "mean_task": stats["mean_task"],
            "rest_db": stats["rest_db"],
            "task_db": stats["task_db"],
            "diff_db": stats["diff_db"],
            "erd_percent": stats["erd_percent"],
            "top_idx": top_idx,
        }

        all_effect_values.append(stats["erd_percent"])

    if not class_results:
        raise RuntimeError("No figures were generated because no valid trials were found.")

    # Use a global symmetric color scale across classes for fair comparison.
    all_values = np.concatenate(all_effect_values)
    vmax = float(np.percentile(np.abs(all_values), 98))
    if np.isclose(vmax, 0.0):
        vmax = 1.0
    vlim = (-vmax, vmax)

    summary = {
        "root_dir": str(args.root_dir),
        "subjects": args.subjects if args.subjects is not None else "all",
        "classes": args.classes,
        "frequency_band_hz": [args.low_freq, args.high_freq],
        "rest_window_seconds": list(REST_WINDOW),
        "task_window_seconds": list(task_window),
        "channels": args.channels if args.channels is not None else "all",
        "reject_threshold_uv": args.reject_threshold_uv,
        "selection_mode": args.selection_mode,
        "top_k": args.top_k,
        "colorbar": "ERD/ERS (%)",
        "vlim_percent": list(vlim),
        "figures": [],
        "top_electrodes": {},
    }

    for class_name, result in class_results.items():
        output_file = output_dir / f"{class_name}_erd_topomap.png"

        plot_task_related_topomap(
            values=result["erd_percent"],
            info=info,
            highlight_idx=result["top_idx"],
            output_file=output_file,
            colorbar_label="ERD/ERS (%)",
            vlim=vlim,
            cmap="RdBu_r",
        )

        summary["figures"].append(str(output_file))
        summary["top_electrodes"][class_name] = [
            {
                "electrode": result["ch_names"][int(idx)],
                "erd_percent": float(result["erd_percent"][int(idx)]),
                "task_minus_rest_db": float(result["diff_db"][int(idx)]),
            }
            for idx in result["top_idx"]
        ]

        print(f"Saved {output_file}")

    # Save summary JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Save CSV of highlighted electrodes
    write_top_electrodes_csv(output_dir / "top_electrodes.csv", class_results)

    # Save captions / explanation text
    write_captions_txt(
        output_file=output_dir / "captions.txt",
        class_results=class_results,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        task_window=task_window,
    )

    print(f"Summary saved to {output_dir / 'summary.json'}")
    print(f"CSV saved to {output_dir / 'top_electrodes.csv'}")
    print(f"Captions saved to {output_dir / 'captions.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
