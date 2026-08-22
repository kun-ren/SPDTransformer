"""Download BCI Competition IV Dataset 2a with MOABB.

By default, all nine subjects are downloaded into the repository's ``data``
directory.

Examples:
    python script/download_bci_competition_iv_2a.py
    python script/download_bci_competition_iv_2a.py --subjects 1-3,5
    python script/download_bci_competition_iv_2a.py --force-update
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "data"
VALID_SUBJECTS = tuple(range(1, 10))


def parse_subjects(value: str) -> list[int]:
    """Parse a comma-separated list of subject IDs and inclusive ranges."""
    subjects: list[int] = []

    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue

            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError(f"range starts after it ends: {item}")
                subjects.extend(range(start, end + 1))
            else:
                subjects.append(int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid subject list {value!r}: {exc}"
        ) from exc

    subjects = sorted(set(subjects))
    if not subjects:
        raise argparse.ArgumentTypeError("at least one subject must be selected")

    invalid = [subject for subject in subjects if subject not in VALID_SUBJECTS]
    if invalid:
        raise argparse.ArgumentTypeError(
            "BCI Competition IV Dataset 2a subject IDs must be between 1 and "
            f"9; got {invalid}"
        )

    return subjects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download BCI Competition IV Dataset 2a (MOABB BNCI2014_001)."
        )
    )
    parser.add_argument(
        "--subjects",
        type=parse_subjects,
        default=list(VALID_SUBJECTS),
        metavar="LIST",
        help="subject IDs or ranges, for example 1-3,5 (default: 1-9)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"download root (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="download files again even if they already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    download_dir = args.download_dir.expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        import moabb
        from moabb.datasets import BNCI2014_001
    except ImportError as exc:
        print(
            "error: MOABB is not installed. Install the project environment "
            "from environment.yml first.",
            file=sys.stderr,
        )
        print(f"details: {exc}", file=sys.stderr)
        return 1

    moabb.set_log_level("info")
    moabb.set_download_dir(str(download_dir))

    dataset = BNCI2014_001()
    print(f"Downloading {dataset.code} subjects {args.subjects}")
    print(f"Download root: {download_dir}")

    try:
        dataset.download(
            subject_list=args.subjects,
            path=str(download_dir),
            force_update=args.force_update,
            update_path=True,
            accept=True,
        )
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
