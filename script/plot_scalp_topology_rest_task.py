"""Plot task-related scalp topomaps for motor imagery EEG.

This script generates one clean publication-style topomap per motor imagery class.
Instead of plotting absolute rest/task power separately, it plots task-related
ERD/ERS relative to rest:

    ERD/ERS (%) = (P_task - P_rest) / P_rest * 100

This better reveals which electrodes are most related to each task.

For each class:
    - all electrode positions are shown
    - the top-k most task-related electrodes are highlighted and labeled
    - no extra in-figure annotation is added except the electrode labels and colorbar

Additional explanations are written separately to:
    - summary.json
    - top_electrodes.csv
    - captions.txt

Each ERD/ERS map is computed from two windows around task onset:
    - rest: -3.5 to 0 s
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
import re
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
RUN_IDS = {4, 6, 8, 10, 12, 14}


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


def map_event_to_label(run_id: int, event_name: str) -> str | None:
    # EEGBCI MI runs
    # Runs 4,8,12 : T1 -> left hand, T2 -> right hand
    # Runs 6,10,14: T1 -> both hands, T2 -> both feet
    if run_id in {4, 8, 12}:
        if event_name == "T1":
            return "left_hand"
        if event_name == "T2":
            return "right_hand"
    if run_id in {6, 10, 14}:
        if event_name == "T1":
            return "hands"
        if event_name == "T2":
            return "feet"
    return None


def subject_dir_name(subject_id: int) -> str:
    return f"S{subject_id:03d}"


def discover_edf_files(root_dir: Path, subjects: list[int] | None) -> list[Path]:
    if subjects is None:
        subject_dirs = sorted(root_dir.glob("S*"))
    else:
        subject_dirs = [root_dir / subject_dir_name(subject) for subject in subjects]

    edf_files = []
    for subject_dir in subject_dirs:
        if not subject_dir.exists():
            print(f"[WARN] Missing subject directory: {subject_dir}")
            continue
        for edf_file in sorted(subject_dir.glob("*.edf")):
            match = re.search(r"R(\d+)", edf_file.name)
            if match is None:
                continue
            if int(match.group(1)) in RUN_IDS:
                edf_files.append(edf_file)

    if not edf_files:
        raise FileNotFoundError(f"No motor-imagery EDF files found under {root_dir}")
    return edf_files


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
            Select strongest ERD electrodes.
            This is usually preferred for motor imagery.

        "absolute":
            Select electrodes with strongest absolute modulation,
            regardless of ERD or ERS.
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

def standardize_and_filter_raw(edf_file: Path, low_freq: float, high_freq: float):
    import mne

    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    # Standardize EEGBCI channel names
    try:
        mne.datasets.eegbci.standardize(raw)
    except Exception as exc:
        print(f"[WARN] Could not standardize EEGBCI channel names for {edf_file}: {exc}")

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)

    # Keep EEG channels only
    raw.pick_types(eeg=True, exclude=[])

    # Use MNE FIR filter for cleaner EEG filtering
    raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        phase="zero",
        fir_design="firwin",
        pad="reflect_limited",
        verbose=False,
    )
    return raw


def eeg_picks_with_positions(raw) -> np.ndarray:
    import mne

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    valid_picks = []

    for pick in eeg_picks:
        loc = raw.info["chs"][pick]["loc"][:3]
        if np.isfinite(loc).all() and not np.allclose(loc, 0.0):
            valid_picks.append(pick)

    if not valid_picks:
        raise RuntimeError("No EEG channels with valid montage positions were found.")

    return np.asarray(valid_picks, dtype=int)


def window_power(
    data: np.ndarray,
    sfreq: float,
    onset: float,
    start_offset: float,
    end_offset: float,
    eps: float,
) -> np.ndarray | None:
    """Compute channel-wise band power in one time window.

    Parameters
    ----------
    data : np.ndarray
        Shape: (n_channels, n_times)
    """
    start = int(round((onset + start_offset) * sfreq))
    end = int(round((onset + end_offset) * sfreq))

    if start < 0 or end > data.shape[1] or end <= start:
        return None

    segment = data[:, start:end]
    power = np.mean(segment * segment, axis=1) + eps
    return power


def collect_class_power(
    root_dir: Path,
    subjects: list[int] | None,
    classes: list[str],
    low_freq: float,
    high_freq: float,
    rest_window: tuple[float, float],
    task_window: tuple[float, float],
    eps: float,
) -> tuple[dict[str, list[TrialPower]], object]:
    import mne

    class_power: dict[str, list[TrialPower]] = defaultdict(list)
    reference_info = None
    reference_channel_names = None

    for edf_file in discover_edf_files(root_dir, subjects):
        run_match = re.search(r"R(\d+)", edf_file.name)
        if run_match is None:
            continue
        run_id = int(run_match.group(1))

        raw = standardize_and_filter_raw(edf_file, low_freq, high_freq)
        picks = eeg_picks_with_positions(raw)
        picked_info = mne.pick_info(raw.info.copy(), picks, copy=True)

        channel_names = tuple(picked_info["ch_names"])
        if reference_info is None:
            reference_info = picked_info
            reference_channel_names = channel_names
        else:
            if channel_names != reference_channel_names:
                raise RuntimeError(
                    "The valid EEG channel set changed across files. "
                    "Please check channel names and montage assignment."
                )

        data = raw.get_data(picks=picks)
        sfreq = float(raw.info["sfreq"])

        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            label = map_event_to_label(run_id, desc)
            if label is None or label not in classes:
                continue

            rest = window_power(
                data=data,
                sfreq=sfreq,
                onset=float(onset),
                start_offset=rest_window[0],
                end_offset=rest_window[1],
                eps=eps,
            )
            task = window_power(
                data=data,
                sfreq=sfreq,
                onset=float(onset),
                start_offset=task_window[0],
                end_offset=task_window[1],
                eps=eps,
            )

            if rest is None or task is None:
                continue

            class_power[label].append(TrialPower(rest=rest, task=task))

    if reference_info is None:
        raise RuntimeError("No valid EEG data was collected.")

    return class_power, reference_info


def compute_class_statistics(trials: list[TrialPower]) -> dict[str, np.ndarray]:
    """Compute class-wise mean power and task-related effect."""
    rest_power = np.stack([trial.rest for trial in trials], axis=0)   # (n_trials, n_channels)
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
    rest_window: tuple[float, float],
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
        f"Rest window: [{rest_window[0]:g}, {rest_window[1]:g}] s relative to task onset."
    )
    lines.append(
        f"Task window: [{task_window[0]:g}, {task_window[1]:g}] s relative to task onset."
    )
    lines.append("Metric: ERD/ERS (%) = (P_task - P_rest) / P_rest × 100.")
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
        "--rest-window",
        type=float,
        nargs=2,
        default=(-3.5, 0.0),
        metavar=("START", "END"),
        help="Rest window in seconds relative to task onset. Default: -3.5 0",
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
    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    rest_window = (float(args.rest_window[0]), float(args.rest_window[1]))
    task_window = (float(args.task_window[0]), float(args.task_window[1]))

    class_power, info = collect_class_power(
        root_dir=args.root_dir,
        subjects=args.subjects,
        classes=args.classes,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        rest_window=rest_window,
        task_window=task_window,
        eps=args.eps,
    )

    class_results: dict[str, dict] = {}
    all_effect_values = []

    for class_name in args.classes:
        trials = class_power.get(class_name, [])
        if not trials:
            print(f"[WARN] No valid trials for class {class_name!r}; skipping.")
            continue

        stats = compute_class_statistics(trials)
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

    # Use a global symmetric color scale across classes for fair comparison
    all_values = np.concatenate(all_effect_values)
    vmax = float(np.percentile(np.abs(all_values), 98))
    vlim = (-vmax, vmax)

    summary = {
        "root_dir": str(args.root_dir),
        "subjects": args.subjects if args.subjects is not None else "all",
        "classes": args.classes,
        "frequency_band_hz": [args.low_freq, args.high_freq],
        "rest_window_seconds": list(rest_window),
        "task_window_seconds": list(task_window),
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
        rest_window=rest_window,
        task_window=task_window,
    )

    print(f"Summary saved to {output_dir / 'summary.json'}")
    print(f"CSV saved to {output_dir / 'top_electrodes.csv'}")
    print(f"Captions saved to {output_dir / 'captions.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
