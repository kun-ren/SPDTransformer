"""Download ZIP datasets configured in a YAML file.

Default command:

    python script/download_datasets.py

The YAML config controls dataset URLs, output paths, skipping existing files,
forced re-downloads, retry behavior, and optional ZIP extraction.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
import zipfile


DEFAULT_CONFIG = Path("configs") / "datasets.yaml"
DEFAULT_USER_AGENT = "SPDTransformer-dataset-downloader/1.0"


class DownloadError(RuntimeError):
    """Raised when a configured download fails."""

def merged_dataset_config(defaults: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    merged = {**defaults, **dataset}
    if "name" not in merged:
        raise ConfigError("each dataset must define 'name'")
    if "url" not in merged:
        raise ConfigError(f"dataset '{merged['name']}' must define 'url'")
    if "output_dir" not in merged:
        raise ConfigError(f"dataset '{merged['name']}' must define 'output_dir'")
    return merged


def bool_value(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be true or false")
    return value


def int_value(config: dict[str, Any], key: str, default: int, minimum: int = 1) -> int:
    value = config.get(key, default)
    if not isinstance(value, int) or value < minimum:
        raise ConfigError(f"'{key}' must be an integer >= {minimum}")
    return value


def output_filename(config: dict[str, Any]) -> str:
    filename = config.get("filename")
    if filename:
        return str(filename)

    url_path = unquote(urlparse(str(config["url"])).path.rstrip("/"))
    inferred = Path(url_path).name
    if inferred and inferred.endswith(".zip"):
        return inferred
    return f"{config['name']}.zip"


def build_request(url: str, resume_from: int, headers: dict[str, str] | None = None) -> Request:
    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        request_headers.update({str(key): str(value) for key, value in headers.items()})
    if resume_from:
        request_headers["Range"] = f"bytes={resume_from}-"
    return Request(url, headers=request_headers)


def sha256sum(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str | None, chunk_size: int) -> None:
    if not expected_sha256:
        return
    actual = sha256sum(path, chunk_size)
    if actual.lower() != expected_sha256.lower():
        raise DownloadError(
            f"checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def verify_zip_file(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        first_bad = archive.testzip()
    if first_bad is not None:
        raise DownloadError(f"ZIP verification failed in {path}: {first_bad}")


def extract_zip(path: Path, extract_dir: Path, overwrite: bool) -> None:
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(extract_dir)
    print(f"extracted: {path} -> {extract_dir}")


def finalize_zip(config: dict[str, Any], path: Path, chunk_size: int) -> None:
    verify_checksum(path, config.get("sha256"), chunk_size)
    if bool_value(config, "verify_zip", True):
        verify_zip_file(path)
    if bool_value(config, "extract", False):
        output_dir = Path(str(config["output_dir"]))
        extract_dir = Path(str(config.get("extract_dir", output_dir / str(config["name"]))))
        overwrite_extracted = bool_value(config, "overwrite_extracted", True)
        extract_zip(path, extract_dir, overwrite_extracted)


def download_zip(config: dict[str, Any]) -> Path:
    name = str(config["name"])
    url = str(config["url"])
    output_dir = Path(str(config["output_dir"]))
    destination = output_dir / output_filename(config)
    partial = destination.with_suffix(destination.suffix + ".part")

    skip_existing = bool_value(config, "skip_existing", True)
    force = bool_value(config, "force", False)
    retries = int_value(config, "retries", 3)
    timeout = int_value(config, "timeout_seconds", 60)
    chunk_size = int_value(config, "chunk_size_mb", 8) * 1024 * 1024
    headers = config.get("headers")
    if headers is not None and not isinstance(headers, dict):
        raise ConfigError(f"dataset '{name}' headers must be a mapping")

    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    elif destination.exists() and skip_existing:
        print(f"skip existing: {destination}")
        finalize_zip(config, destination, chunk_size)
        return destination

    for attempt in range(1, retries + 1):
        resume_from = partial.stat().st_size if partial.exists() else 0
        mode = "ab" if resume_from else "wb"
        try:
            print(f"downloading {name}: {url}")
            request = build_request(url, resume_from, headers)
            with urlopen(request, timeout=timeout) as response:
                if resume_from and getattr(response, "status", None) != 206:
                    resume_from = 0
                    mode = "wb"
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
            partial.replace(destination)
            finalize_zip(config, destination, chunk_size)
            print(f"downloaded: {destination}")
            return destination
        except (HTTPError, URLError, TimeoutError, OSError, zipfile.BadZipFile, DownloadError) as exc:
            if attempt == retries:
                raise DownloadError(f"failed to download '{name}': {exc}") from exc
            wait_seconds = min(2**attempt, 30)
            print(f"retry {attempt}/{retries - 1} for {name} after {wait_seconds}s: {exc}")
            time.sleep(wait_seconds)

    raise DownloadError(f"failed to download '{name}'")


def select_datasets(config: dict[str, Any], names: list[str] | None) -> list[dict[str, Any]]:
    defaults = config.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise ConfigError("'defaults' must be a mapping")

    selected: list[dict[str, Any]] = []
    requested = set(names or [])
    known_names: set[str] = set()
    for raw_dataset in config["datasets"]:
        if not isinstance(raw_dataset, dict):
            raise ConfigError("each item in 'datasets' must be a mapping")
        dataset = merged_dataset_config(defaults, raw_dataset)
        name = str(dataset["name"])
        known_names.add(name)
        if not requested or name in requested:
            selected.append(dataset)

    missing = requested - known_names
    if missing:
        raise ConfigError(f"unknown dataset name(s): {', '.join(sorted(missing))}")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ZIP dataset files from a YAML configuration.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML dataset config, default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="download only this dataset name; repeat for multiple names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show selected downloads without writing files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        datasets = select_datasets(config, args.datasets)
        if args.dry_run:
            for dataset in datasets:
                destination = Path(str(dataset["output_dir"])) / output_filename(dataset)
                message = f"{dataset['name']}: {dataset['url']} -> {destination}"
                if bool_value(dataset, "extract", False):
                    extract_dir = Path(
                        str(dataset.get("extract_dir", Path(str(dataset["output_dir"])) / str(dataset["name"])))
                    )
                    message += f" ; unzip -> {extract_dir}"
                print(message)
            return 0

        for dataset in datasets:
            download_zip(dataset)
    except (ConfigError, DownloadError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
