#!/usr/bin/env python3
"""Download and reproduce the prepared CRATER Carparts23 dataset.

The public archive contains the original YOLO-segmentation data. This script
verifies that archive, extracts it safely, runs the deterministic CRATER
preparation pipeline, validates the result, and installs the ignored dataset
directories into ``datasets/civilian/carparts23`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import BinaryIO


DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "carparts-seg.zip"
)
ARCHIVE_SHA256 = "e2eea20030d02366b07174bcdbcc843f63791bdd53962a83e00636d15e78e41f"
MANAGED_DIRECTORIES = ("images", "annotations", "manifests", "audit")
SPLITS = ("train", "val", "test")
EXPECTED = {
    "train": {"images": 2585, "carparts23": 14210, "crater_transfer3": 4596},
    "val": {"images": 193, "carparts23": 1177, "crater_transfer3": 374},
    "test": {"images": 192, "carparts23": 1150, "crater_transfer3": 370},
}


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    default_dataset_root = (
        repository_root / "datasets" / "civilian" / "carparts23"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_dataset_root,
        help="Dataset root receiving images, annotations, manifests, and audit",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing archive instead of downloading it",
    )
    parser.add_argument(
        "--similarity-workers",
        type=int,
        default=4,
        help="Worker processes used by visual source grouping (default: 4)",
    )
    return parser.parse_args()


def ensure_destination_is_empty(output_root: Path) -> None:
    existing = [name for name in MANAGED_DIRECTORIES if (output_root / name).exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(
            f"Refusing to overwrite existing dataset directories: {joined}. "
            "Move or remove them explicitly before downloading."
        )


def ensure_preparation_dependencies() -> None:
    required = {"numpy": "numpy", "PIL": "Pillow", "scipy": "scipy"}
    missing = [
        package
        for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Missing dataset preparation dependencies: "
            f"{', '.join(missing)}. Run `python -m pip install -r requirements.txt`."
        )


def copy_with_progress(response: BinaryIO, destination: Path, total: int) -> str:
    digest = hashlib.sha256()
    received = 0
    with destination.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if total:
                percent = received * 100 // total
                print(
                    f"\rDownloading: {received / 1024**2:.1f} MiB "
                    f"of {total / 1024**2:.1f} MiB ({percent}%)",
                    end="",
                    flush=True,
                )
    if total:
        print()
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    print(f"Downloading {DOWNLOAD_URL}")
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "CRATER dataset downloader"},
    )
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length", "0"))
        actual_hash = copy_with_progress(response, destination, total)
    verify_hash(actual_hash)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(actual_hash: str) -> None:
    if actual_hash.lower() != ARCHIVE_SHA256:
        raise ValueError(
            "Archive SHA-256 mismatch: "
            f"expected {ARCHIVE_SHA256}, received {actual_hash.lower()}"
        )
    print(f"Verified archive SHA-256: {actual_hash.lower()}")


def extract_archive(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as file:
        for member in file.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination_root and destination_root not in member_path.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        file.extractall(destination)


def find_source_root(extracted_root: Path) -> Path:
    candidates = [extracted_root, *sorted(path for path in extracted_root.rglob("*") if path.is_dir())]
    matches = [
        path
        for path in candidates
        if all((path / "images" / split).is_dir() for split in SPLITS)
        and all((path / "labels" / split).is_dir() for split in SPLITS)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one extracted dataset root containing "
            f"images/{{train,val,test}} and labels/{{train,val,test}}; found {len(matches)}"
        )
    return matches[0]


def run_preparation(source_root: Path, prepared_root: Path, workers: int) -> None:
    if workers < 1:
        raise ValueError("--similarity-workers must be at least 1")
    prepare_script = Path(__file__).with_name("prepare_carparts23.py")
    if not prepare_script.is_file():
        raise FileNotFoundError(f"Missing preparation script: {prepare_script}")
    command = [
        sys.executable,
        str(prepare_script),
        str(source_root),
        str(prepared_root),
        "--similarity-workers",
        str(workers),
    ]
    print("Preparing leakage-resistant COCO splits. This can take several minutes.")
    subprocess.run(command, check=True)


def normalize_generated_text(prepared_root: Path) -> None:
    for directory in ("annotations", "manifests", "audit"):
        for path in (prepared_root / directory).rglob("*"):
            if path.suffix.lower() not in {".csv", ".json", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            with path.open("w", encoding="utf-8", newline="\n") as file:
                file.write(text)
    print("Normalized generated text files to LF line endings.")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_prepared_dataset(prepared_root: Path) -> None:
    for split, expected in EXPECTED.items():
        image_dir = prepared_root / "images" / split
        image_count = sum(path.is_file() for path in image_dir.iterdir())
        if image_count != expected["images"]:
            raise ValueError(
                f"Unexpected {split} image count: expected {expected['images']}, "
                f"received {image_count}"
            )

        manifest = prepared_root / "manifests" / f"{split}.txt"
        manifest_count = len(manifest.read_text(encoding="utf-8").splitlines())
        if manifest_count != expected["images"]:
            raise ValueError(
                f"Unexpected {split} manifest count: expected {expected['images']}, "
                f"received {manifest_count}"
            )

        for annotation_set in ("carparts23", "crater_transfer3"):
            annotation_path = (
                prepared_root
                / "annotations"
                / f"{annotation_set}_instances_{split}.json"
            )
            document = load_json(annotation_path)
            if len(document["images"]) != expected["images"]:
                raise ValueError(f"Image count mismatch in {annotation_path}")
            instance_count = len(document["annotations"])
            if instance_count != expected[annotation_set]:
                raise ValueError(
                    f"Unexpected instance count in {annotation_path}: expected "
                    f"{expected[annotation_set]}, received {instance_count}"
                )

    report = load_json(prepared_root / "audit" / "audit_report.json")
    report_hash = report["dataset"]["archive_sha256"]
    if report_hash != ARCHIVE_SHA256:
        raise ValueError(f"Audit report contains unexpected archive hash: {report_hash}")
    print("Validated prepared dataset counts and provenance.")


def install_dataset(prepared_root: Path, output_root: Path) -> None:
    installed: list[tuple[Path, Path]] = []
    try:
        for name in MANAGED_DIRECTORIES:
            source = prepared_root / name
            destination = output_root / name
            shutil.move(str(source), str(destination))
            installed.append((source, destination))
    except Exception:
        for source, destination in reversed(installed):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        ensure_destination_is_empty(output_root)
    except FileExistsError as error:
        raise SystemExit(f"error: {error}") from None
    ensure_preparation_dependencies()

    with tempfile.TemporaryDirectory(prefix="crater-car-data-") as temporary:
        temporary_root = Path(temporary)
        if args.archive:
            archive = args.archive.resolve()
            if not archive.is_file():
                raise FileNotFoundError(f"Archive does not exist: {archive}")
            verify_hash(sha256_file(archive))
        else:
            archive = temporary_root / "carparts-seg.zip"
            download_archive(archive)

        extracted_root = temporary_root / "raw"
        extracted_root.mkdir()
        print("Extracting archive")
        extract_archive(archive, extracted_root)
        source_root = find_source_root(extracted_root)

        prepared_root = temporary_root / "prepared"
        run_preparation(source_root, prepared_root, args.similarity_workers)
        normalize_generated_text(prepared_root)
        validate_prepared_dataset(prepared_root)
        install_dataset(prepared_root, output_root)

    print(f"Installed CRATER car-parts data under {output_root}")


if __name__ == "__main__":
    main()
