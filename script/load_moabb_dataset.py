"""Load the PhysioNet Motor Imagery dataset with MOABB.

Examples:
    python script/load_moabb_dataset.py --subjects 1
    python script/load_moabb_dataset.py --subjects 1-5 --save-npz data/physionet_mi_s1_5.npz
    python script/load_moabb_dataset.py --download-only --subjects 1-10
"""

from __future__ import annotations

import argparse
import os

from pathlib import Path
import sys
import moabb.datasets as moabb_datasets
import moabb
from moabb import set_download_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_util import load_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets.yaml"

config = load_config(DEFAULT_CONFIG)
defaults = config.get("defaults", {})
datasets_config = config.get("datasets", [])

moabb.set_log_level("warning")


def path_for_moabb(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(
                "On Windows, Moabb should use a relative download path "
                "inside the project, for example: data/eeg"
            ) from exc

    return path.as_posix()


def parse_int_list(raw: str) -> list[int]:
    """Parse subject list"""
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if start > end:
                raise argparse.ArgumentTypeError(f"invalid range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    subjects = sorted(set(values))
    invalid = [subject for subject in subjects if subject < 1 or subject > 109]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"PhysionetMI subject ids must be in [1, 109], got {invalid}"
        )
    return subjects


def default_subjects(include_subject_88: bool) -> list[int]:
    subjects = list(range(1, 110))
    if not include_subject_88:
        subjects.remove(88)
    return subjects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load EEG data through MOABB.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(defaults["download_dir"]) if defaults.get("download_dir") else None,
        help="MOABB/MNE download directory",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="force MOABB to re-download existing files",
    )
    parser.add_argument(
        "--save-npz",
        type=bool,
        default=True,
        help="save MotorImagery arrays to this .npz file",
    )

    parser.add_argument(
        "--download-only",
        type=bool,
        default=False,
        help="download only the raw files",
    )
    parser.add_argument(
        "--metadata-csv",
        type=bool,
        default=False,
        help="save MotorImagery metadata to this CSV file",
    )
    return parser


def configure_moabb_download_dir(download_dir: str | None) -> None:
    if download_dir is None:
        return
    os.makedirs(download_dir, exist_ok=True)
    set_download_dir(download_dir)
    print(f"downloading to : {download_dir}")


def load_dataset(args: argparse.Namespace, dataset_config: dict) -> None:
    dataset_name = dataset_config["name"]
    DatasetClass = getattr(moabb_datasets, dataset_name)
    subjects = parse_int_list(dataset_config["subjects"])

    dataset = DatasetClass(
        **dataset_config.get("params", {}),
        subjects=subjects,
    )


    if args.download_only:
        dataset.download(
            subject_list=subjects,
            path=args.download_dir,
            force_update=args.force_update,
            update_path=True,
            accept=True,
        )
        print(f"downloaded {dataset_name} subjects: {subjects}")
        return

    from moabb.paradigms import MotorImagery
    import numpy as np

    paradigm = MotorImagery(resample=dataset_config["sample_rate"])
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

    print(f"X shape: {X.shape}")
    print(f"labels shape: {labels}")
    print(f"metadata shape: {metadata.shape}")
    print(f"classes: {sorted(set(labels))}")
    print(f"labels: {labels[:10]}")

    # if args.save_npz:
    #     path = Path(args.download_dir) / f"{dataset_name}.npz"
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     np.savez_compressed(
    #         path,
    #         X=X,
    #         labels=np.asarray(labels),
    #         metadata=metadata.to_dict(orient="list"),
    #         subjects=np.asarray(subjects),
    #     )
    #     print(f"saved arrays: {path}")

    # if args.metadata_csv:
    #     path = Path(args.download_dir) / f"{dataset_name}.csv"
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #     metadata.to_csv(path, index=False)
    #     print(f"saved metadata: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        os.chdir(PROJECT_ROOT)
        download_dir = path_for_moabb(args.download_dir)
        configure_moabb_download_dir(download_dir)
        for dataset_config in datasets_config:
            load_dataset(args, dataset_config)
    except ImportError as exc:
        print(
            "error: MOABB is not installed. Install the conda environment from "
            "environment.yml first.",
            file=sys.stderr,
        )
        print(f"details: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
