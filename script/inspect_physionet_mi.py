"""Inspect PhysioNet Motor Imagery signals and labels with MOABB.

Examples:
    python script/inspect_physionet_mi.py --subjects 1
    python script/inspect_physionet_mi.py --subjects 1 --events left_hand,right_hand
    python script/inspect_physionet_mi.py --subjects 1 --trial-index 0 --save-fig data/physionet_trial0.png
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_subjects(raw: str) -> list[int]:
    subjects: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            subjects.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            subjects.append(int(part))

    subjects = sorted(set(subjects))
    invalid = [subject for subject in subjects if subject < 1 or subject > 109]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"PhysionetMI subject ids must be in [1, 109], got {invalid}"
        )
    return subjects


def parse_events(raw: str) -> list[str]:
    events = [event.strip() for event in raw.split(",") if event.strip()]
    if not events:
        raise argparse.ArgumentTypeError("at least one event is required")
    return events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print PhysionetMI EEG signals, labels, and metadata.",
    )
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=parse_subjects("1"),
        help="subject ids, for example: 1 or 1,2,3 or 1-3",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="MOABB/MNE download directory",
    )
    parser.add_argument(
        "--physionet-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0",
        help="local PhysioNet EEGMMI EDF cache root",
    )
    parser.add_argument(
        "--edf-file",
        type=Path,
        default=None,
        help="read this EDF file directly for raw timeline plotting",
    )
    parser.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="inspect arrays from an existing .npz file instead of loading MOABB",
    )
    parser.add_argument(
        "--events",
        type=parse_events,
        default=parse_events("left_hand,right_hand,hands,feet"),
        help="comma-separated labels to load",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="epoch start time relative to the task cue, in seconds",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=3.0,
        help="epoch end time relative to the task cue, in seconds",
    )
    parser.add_argument(
        "--resample",
        type=float,
        default=None,
        help="optional target sampling rate, for example 128",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=160.0,
        help="sampling rate to use when plotting data loaded with --from-npz",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=0,
        help="which trial to print and plot",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="print a continuous raw EEG segment and skip epoch extraction",
    )
    parser.add_argument(
        "--raw-start",
        type=float,
        default=0.0,
        help="raw EEG segment start time in seconds",
    )
    parser.add_argument(
        "--raw-duration",
        type=float,
        default=None,
        help="raw EEG segment duration in seconds; set this to print continuous EEG",
    )
    parser.add_argument(
        "--raw-run-index",
        type=int,
        default=0,
        help="which raw run to inspect when printing continuous EEG",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="maximum number of time samples to print per channel",
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=8,
        help="maximum number of channels to plot",
    )
    parser.add_argument(
        "--save-fig",
        type=Path,
        default=PROJECT_ROOT / "data" / "physionet_mi_trial.png",
        help="where to save the trial signal plot",
    )
    parser.add_argument(
        "--save-raw-fig",
        type=Path,
        default=PROJECT_ROOT / "data" / "physionet_mi_raw_timeline.png",
        help="where to save the continuous raw EEG timeline plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show the matplotlib window after saving the figure",
    )
    return parser


def moabb_path(download_dir: Path) -> str:
    download_dir = download_dir.resolve()
    try:
        return download_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(download_dir)


def configure_download_dir(download_dir: Path, set_download_dir) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(moabb_path(download_dir))


def print_raw_annotations(dataset, subjects: list[int]) -> None:
    raw_data = dataset.get_data(subjects=subjects[:1])
    subject = subjects[0]
    print("\nRaw EDF annotations from the first subject/run:")
    for session_name, runs in raw_data[subject].items():
        for run_name, raw in runs.items():
            print(f"  session={session_name}, run={run_name}")
            print(f"  raw.info['sfreq'] = {raw.info['sfreq']}")
            print(f"  raw.ch_names[:8] = {raw.ch_names[:8]}")
            print(raw.annotations[:10])
            return


def get_raw_by_index(dataset, subjects: list[int], run_index: int):
    raw_data = dataset.get_data(subjects=subjects[:1])
    subject = subjects[0]
    flat_runs = []

    for session_name, runs in raw_data[subject].items():
        for run_name, raw in runs.items():
            flat_runs.append((session_name, run_name, raw))

    if not flat_runs:
        raise RuntimeError(f"no raw runs found for subject {subject}")
    if run_index < 0 or run_index >= len(flat_runs):
        raise IndexError(f"raw-run-index must be in [0, {len(flat_runs) - 1}]")

    return flat_runs[run_index]


def inspect_raw_object(
    raw,
    subject,
    session_name,
    run_name,
    start_time: float,
    duration: float,
    max_channels: int,
    max_samples: int,
    save_raw_fig: Path,
    show: bool,
) -> None:
    if duration <= 0:
        raise ValueError("raw-duration must be positive")

    session_name, run_name, raw = get_raw_by_index(dataset, subjects, run_index)
    sfreq = float(raw.info["sfreq"])
    start_sample = int(round(start_time * sfreq))
    stop_sample = int(round((start_time + duration) * sfreq))
    start_sample = max(0, start_sample)
    stop_sample = min(raw.n_times, stop_sample)
    if start_sample >= stop_sample:
        raise ValueError(
            f"empty raw segment: start={start_time}s, duration={duration}s, "
            f"raw duration={raw.times[-1]:.3f}s"
        )

    picks = list(range(min(max_channels, len(raw.ch_names))))
    segment = raw.get_data(picks=picks, start=start_sample, stop=stop_sample)
    times = raw.times[start_sample:stop_sample]

    print("\nContinuous raw EEG segment:")
    print(f"  subject = {subject}")
    print(f"  session = {session_name}")
    print(f"  run = {run_name}")
    print(f"  raw.info['sfreq'] = {sfreq} Hz")
    print(f"  raw.n_times = {raw.n_times}")
    print(f"  raw duration = {raw.times[-1]:.3f}s")
    print(f"  segment time = {times[0]:.3f}s to {times[-1]:.3f}s")
    print(f"  segment shape = {segment.shape}  # channels, time samples")
    print(f"  channel names = {[raw.ch_names[pick] for pick in picks]}")

    print(f"\nFirst {min(max_samples, segment.shape[1])} samples per channel:")
    for row_index, pick in enumerate(picks):
        values = segment[row_index, :max_samples]
        print(f"  {raw.ch_names[pick]}: {values}")

    overlapping_annotations = []
    print("\nAnnotations overlapping this raw segment:")
    segment_start = times[0]
    segment_stop = times[-1]
    for annotation in raw.annotations:
        onset = float(annotation["onset"])
        duration_value = float(annotation["duration"])
        offset = onset + duration_value
        if onset <= segment_stop and offset >= segment_start:
            overlapping_annotations.append(annotation)
            print(
                f"  onset={onset:.3f}s, duration={duration_value:.3f}s, "
                f"description={annotation['description']}"
            )
    if not overlapping_annotations:
        print("  no annotation in this segment")

    plot_raw_timeline(
        segment=segment,
        times=times,
        ch_names=[raw.ch_names[pick] for pick in picks],
        annotations=overlapping_annotations,
        save_raw_fig=save_raw_fig,
        show=show,
    )


def print_raw_segment(
    dataset,
    subjects: list[int],
    start_time: float,
    duration: float,
    run_index: int,
    max_channels: int,
    max_samples: int,
    save_raw_fig: Path,
    show: bool,
) -> None:
    session_name, run_name, raw = get_raw_by_index(dataset, subjects, run_index)
    inspect_raw_object(
        raw=raw,
        subject=subjects[0],
        session_name=session_name,
        run_name=run_name,
        start_time=start_time,
        duration=duration,
        max_channels=max_channels,
        max_samples=max_samples,
        save_raw_fig=save_raw_fig,
        show=show,
    )


def find_cached_edf_path(
    subjects: list[int],
    run_index: int,
    physionet_root: Path,
    edf_file: Path | None,
):
    subject = subjects[0]
    if edf_file is not None:
        path = edf_file if edf_file.is_absolute() else PROJECT_ROOT / edf_file
    else:
        subject_dir = physionet_root / f"S{subject:03d}"
        edf_files = sorted(subject_dir.glob("*.edf"))
        if not edf_files:
            return None
        if run_index < 0 or run_index >= len(edf_files):
            raise IndexError(f"raw-run-index must be in [0, {len(edf_files) - 1}]")
        path = edf_files[run_index]

    if not path.exists():
        return None

    return subject, "edf", path.stem, path


def inspect_edf_path(
    path: Path,
    subject: int,
    session_name: str,
    run_name: str,
    start_time: float,
    duration: float,
    max_channels: int,
    max_samples: int,
    save_raw_fig: Path,
    show: bool,
) -> None:
    import pyedflib

    if duration <= 0:
        raise ValueError("raw-duration must be positive")

    reader = pyedflib.EdfReader(str(path))
    try:
        sfreq = float(reader.getSampleFrequency(0))
        n_times = int(reader.getNSamples()[0])
        start_sample = max(0, int(round(start_time * sfreq)))
        stop_sample = min(n_times, int(round((start_time + duration) * sfreq)))
        if start_sample >= stop_sample:
            raw_duration = (n_times - 1) / sfreq
            raise ValueError(
                f"empty raw segment: start={start_time}s, duration={duration}s, "
                f"raw duration={raw_duration:.3f}s"
            )

        n_channels = min(max_channels, reader.signals_in_file)
        ch_names = reader.getSignalLabels()[:n_channels]
        segment = np.vstack(
            [
                reader.readSignal(channel, start=start_sample, n=stop_sample - start_sample)
                for channel in range(n_channels)
            ]
        )
        times = np.arange(start_sample, stop_sample) / sfreq
        onsets, durations, descriptions = reader.readAnnotations()

        segment_start = float(times[0])
        segment_stop = float(times[-1])
        overlapping_annotations = []
        for onset, duration_value, description in zip(onsets, durations, descriptions):
            onset = float(onset)
            duration_value = float(duration_value)
            offset = onset + duration_value
            if onset <= segment_stop and offset >= segment_start:
                mapped_label = map_physionet_annotation(run_name, str(description))
                overlapping_annotations.append(
                    {
                        "onset": onset,
                        "duration": duration_value,
                        "description": str(description),
                        "label": mapped_label,
                    }
                )

        print("\nContinuous raw EEG segment:")
        print(f"  file = {path}")
        print(f"  subject = {subject}")
        print(f"  session = {session_name}")
        print(f"  run = {run_name}")
        print(f"  sfreq = {sfreq} Hz")
        print(f"  n_times = {n_times}")
        print(f"  raw duration = {(n_times - 1) / sfreq:.3f}s")
        print(f"  segment time = {times[0]:.3f}s to {times[-1]:.3f}s")
        print(f"  segment shape = {segment.shape}  # channels, time samples")
        print(f"  channel names = {ch_names}")

        print(f"\nFirst {min(max_samples, segment.shape[1])} samples per channel:")
        for row_index, channel_name in enumerate(ch_names):
            values = segment[row_index, :max_samples]
            print(f"  {channel_name}: {values}")

        print("\nAnnotations overlapping this raw segment:")
        if overlapping_annotations:
            for annotation in overlapping_annotations:
                print(
                    f"  onset={annotation['onset']:.3f}s, "
                    f"duration={annotation['duration']:.3f}s, "
                    f"description={annotation['description']}, "
                    f"label={annotation['label']}"
                )
        else:
            print("  no annotation in this segment")

        plot_raw_timeline(
            segment=segment,
            times=times,
            ch_names=ch_names,
            annotations=overlapping_annotations,
            save_raw_fig=save_raw_fig,
            show=show,
        )
    finally:
        reader.close()


def map_physionet_annotation(run_name: str, description: str) -> str:
    if description == "T0":
        return "rest"

    try:
        run_number = int(run_name[-2:])
    except ValueError:
        return description

    left_right_runs = {3, 4, 7, 8, 11, 12}
    hands_feet_runs = {5, 6, 9, 10, 13, 14}

    if run_number in left_right_runs:
        return {"T1": "left_hand", "T2": "right_hand"}.get(description, description)
    if run_number in hands_feet_runs:
        return {"T1": "hands", "T2": "feet"}.get(description, description)
    return description


def plot_raw_timeline(
    segment: np.ndarray,
    times: np.ndarray,
    ch_names: list[str],
    annotations,
    save_raw_fig: Path,
    show: bool,
) -> None:
    n_channels = segment.shape[0]
    spacing = np.nanmax(np.abs(segment)) * 2.5
    if not np.isfinite(spacing) or spacing == 0:
        spacing = 1.0

    color_map = {
        "T0": "#9ca3af",
        "T1": "#2563eb",
        "T2": "#dc2626",
        "rest": "#9ca3af",
        "left_hand": "#2563eb",
        "right_hand": "#dc2626",
        "hands": "#16a34a",
        "feet": "#9333ea",
    }

    fig, ax = plt.subplots(figsize=(13, 2.4 + n_channels * 0.45))
    ymin = -spacing
    ymax = n_channels * spacing

    for annotation in annotations:
        onset = float(annotation["onset"])
        duration_value = float(annotation["duration"])
        description = str(annotation.get("label", annotation["description"]))
        start = max(onset, float(times[0]))
        stop = min(onset + duration_value, float(times[-1]))
        color = color_map.get(description, "#f59e0b")
        ax.axvspan(start, stop, color=color, alpha=0.16)
        ax.text(
            start,
            ymax,
            description,
            color=color,
            fontsize=9,
            va="bottom",
            ha="left",
        )

    for channel_index in range(n_channels):
        offset = (n_channels - channel_index - 1) * spacing
        ax.plot(times, segment[channel_index] + offset, color="#111827", linewidth=0.75)
        ax.text(times[0], offset, ch_names[channel_index], va="center", ha="right", fontsize=8)

    ax.set_title("Continuous PhysionetMI raw EEG with annotation labels")
    ax.set_xlabel("Raw time (s)")
    ax.set_yticks([])
    ax.set_ylim(ymin, ymax + spacing * 0.4)
    ax.margins(x=0.01)
    fig.tight_layout()

    save_raw_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_raw_fig, dpi=160)
    print(f"\nSaved raw timeline plot: {save_raw_fig}")

    if show:
        plt.show()
    plt.close(fig)


def load_npz_arrays(path: Path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    labels = np.asarray(data["labels"]).astype(str)

    metadata_raw = data.get("metadata")
    if metadata_raw is None:
        metadata = pd.DataFrame(index=range(len(labels)))
    elif metadata_raw.shape == ():
        metadata = pd.DataFrame(metadata_raw.item())
    else:
        metadata = pd.DataFrame(metadata_raw)

    return X, labels, metadata


def plot_trial(
    X: np.ndarray,
    labels: np.ndarray,
    metadata,
    ch_names: list[str],
    sfreq: float,
    trial_index: int,
    max_channels: int,
    save_fig: Path,
    show: bool,
) -> None:
    if trial_index < 0 or trial_index >= len(labels):
        raise IndexError(f"trial-index must be in [0, {len(labels) - 1}]")

    trial = X[trial_index]
    n_channels = min(max_channels, trial.shape[0])
    times = np.arange(trial.shape[1]) / sfreq
    spacing = np.nanmax(np.abs(trial[:n_channels])) * 2.5
    if not np.isfinite(spacing) or spacing == 0:
        spacing = 1.0

    fig, ax = plt.subplots(figsize=(12, 2 + n_channels * 0.45))
    for channel_index in range(n_channels):
        offset = (n_channels - channel_index - 1) * spacing
        channel_name = ch_names[channel_index] if channel_index < len(ch_names) else str(channel_index)
        ax.plot(times, trial[channel_index] + offset, linewidth=0.8)
        ax.text(times[0], offset, channel_name, va="center", ha="right", fontsize=8)

    label = labels[trial_index]
    row = metadata.iloc[trial_index].to_dict()
    subject = row.get("subject", "?")
    session = row.get("session", "?")
    run = row.get("run", "?")
    ax.set_title(
        f"PhysionetMI trial {trial_index}: label={label}, "
        f"subject={subject}, session={session}, run={run}"
    )
    ax.set_xlabel("Time after cue (s)")
    ax.set_yticks([])
    ax.margins(x=0.01)
    fig.tight_layout()

    save_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fig, dpi=160)
    print(f"\nSaved signal plot: {save_fig}")

    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)

    if args.from_npz is not None:
        npz_path = args.from_npz
        if not npz_path.is_absolute():
            npz_path = PROJECT_ROOT / npz_path

        X, labels, metadata = load_npz_arrays(npz_path)
        sfreq = args.resample or args.sfreq
        ch_names = [f"ch_{index + 1:02d}" for index in range(X.shape[1])]

        print(f"Loaded arrays from: {npz_path}")
        print(f"Events requested by CLI are ignored in --from-npz mode.")
        print("\nLoaded arrays:")
        print(f"  X.shape = {X.shape}  # trials, channels, time samples")
        print(f"  labels.shape = {labels.shape}")
        print(f"  metadata.shape = {metadata.shape}")
        print(f"  sampling rate used for plotting = {sfreq} Hz")

        print("\nLabel counts:")
        for label, count in sorted(Counter(labels).items()):
            print(f"  {label}: {count}")

        print("\nFirst 10 labels:")
        print(labels[:10])

        print("\nFirst 10 metadata rows:")
        print(metadata.head(10).to_string(index=False))

        trial_index = args.trial_index
        print(f"\nSelected trial {trial_index}:")
        print(f"  label = {labels[trial_index]}")
        print(f"  metadata = {metadata.iloc[trial_index].to_dict()}")
        print(f"  signal shape = {X[trial_index].shape}  # channels, time samples")
        print(f"  first channel, first 10 samples = {X[trial_index, 0, :10]}")

        plot_trial(
            X=X,
            labels=labels,
            metadata=metadata,
            ch_names=ch_names,
            sfreq=sfreq,
            trial_index=trial_index,
            max_channels=args.max_channels,
            save_fig=args.save_fig,
            show=args.show,
        )
        return 0

    raw_already_printed = False
    if args.raw_duration is not None or args.raw_only:
        duration = args.raw_duration if args.raw_duration is not None else 5.0
        cached_raw = find_cached_edf_path(
            subjects=args.subjects,
            run_index=args.raw_run_index,
            physionet_root=args.physionet_root,
            edf_file=args.edf_file,
        )
        if cached_raw is not None:
            subject, session_name, run_name, path = cached_raw
            inspect_edf_path(
                path=path,
                subject=subject,
                session_name=session_name,
                run_name=run_name,
                start_time=args.raw_start,
                duration=duration,
                max_channels=args.max_channels,
                max_samples=args.max_samples,
                save_raw_fig=args.save_raw_fig,
                show=args.show,
            )
            raw_already_printed = True
            if args.raw_only:
                return 0

    try:
        import moabb
        from moabb import set_download_dir
        from moabb.datasets import PhysionetMI
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        print(
            "error: MOABB is not installed in this Python environment. "
            "Activate the project conda environment or install dependencies from environment.yml.",
            file=sys.stderr,
        )
        print(f"details: {exc}", file=sys.stderr)
        return 1

    moabb.set_log_level("warning")
    configure_download_dir(args.download_dir, set_download_dir)

    dataset = PhysionetMI(
        imagined=True,
        executed=False,
        subjects=args.subjects,
    )

    if (args.raw_duration is not None or args.raw_only) and not raw_already_printed:
        duration = args.raw_duration if args.raw_duration is not None else 5.0
        print_raw_segment(
            dataset=dataset,
            subjects=args.subjects,
            start_time=args.raw_start,
            duration=duration,
            run_index=args.raw_run_index,
            max_channels=args.max_channels,
            max_samples=args.max_samples,
            save_raw_fig=args.save_raw_fig,
            show=args.show,
        )
        if args.raw_only:
            return 0

    paradigm = MotorImagery(
        events=args.events,
        n_classes=len(args.events),
        tmin=args.tmin,
        tmax=args.tmax,
        resample=args.resample,
    )

    print(f"Dataset: {dataset.code}")
    print("Task: imagined motor imagery only")
    print(f"Subjects: {args.subjects}")
    print(f"Events: {args.events}")
    print(f"Epoch window: {args.tmin}s to {args.tmax}s")

    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=args.subjects)
    labels = np.asarray(labels).astype(str)

    sfreq = float(paradigm.resample or dataset.metadata.acquisition.sampling_rate)
    ch_names = list(paradigm.channels or [])

    if not ch_names:
        raw_data = dataset.get_data(subjects=args.subjects[:1])
        first_subject = args.subjects[0]
        first_session = next(iter(raw_data[first_subject].values()))
        first_raw = next(iter(first_session.values()))
        ch_names = first_raw.ch_names

    print("\nLoaded arrays:")
    print(f"  X.shape = {X.shape}  # trials, channels, time samples")
    print(f"  labels.shape = {labels.shape}")
    print(f"  metadata.shape = {metadata.shape}")
    print(f"  sampling rate used for epochs = {sfreq} Hz")

    print("\nLabel counts:")
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label}: {count}")

    print("\nFirst 10 labels:")
    print(labels[:10])

    print("\nFirst 10 metadata rows:")
    print(metadata.head(10).to_string(index=False))

    trial_index = args.trial_index
    print(f"\nSelected trial {trial_index}:")
    print(f"  label = {labels[trial_index]}")
    print(f"  metadata = {metadata.iloc[trial_index].to_dict()}")
    print(f"  signal shape = {X[trial_index].shape}  # channels, time samples")
    print(f"  first channel, first 10 samples = {X[trial_index, 0, :10]}")

    print_raw_annotations(dataset, args.subjects)

    plot_trial(
        X=X,
        labels=labels,
        metadata=metadata,
        ch_names=ch_names,
        sfreq=sfreq,
        trial_index=trial_index,
        max_channels=args.max_channels,
        save_fig=args.save_fig,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
