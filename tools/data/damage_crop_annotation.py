"""Utilities for detector-driven military damage-crop annotation runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps


PART_GROUP_ORDER = ("mobility", "structure", "mission_equipment")
SOURCE6_PART_GROUPS = {
    "wheel_tire": "mobility",
    "wheel": "mobility",
    "track": "mobility",
    "windshield": "structure",
    "door": "structure",
    "engine_bay": "structure",
    "hull": "structure",
    "weapon_station": "mission_equipment",
    "mounted_gun": "mission_equipment",
    "comms_equipment": "mission_equipment",
    "comms_system": "mission_equipment",
}
SEVERITY_NAMES = {1: "none", 2: "moderate", 3: "severe"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MANIFEST_FIELDS = (
    "crop_id",
    "part_group",
    "detector_class",
    "detector_confidence",
    "source_image",
    "source_asset_id",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
    "crop_file",
    "checkpoint_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "third_party" / "YOLOX").is_dir() and (candidate / "configs" / "taxonomy.json").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the CRATER repository root.")


def load_coco_class_names(annotation_file: Path) -> list[str]:
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
    ids = [int(item["id"]) for item in categories]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError(f"Expected contiguous one-based category IDs, found {ids}")
    return [str(item["name"]) for item in categories]


def discover_source_images(
    image_dir: Path,
    catalog_path: Path | None = None,
    local_file_field: str = "local_file",
) -> list[Path]:
    image_dir = image_dir.resolve()
    if catalog_path is None:
        paths = [
            path.resolve()
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    else:
        catalog_path = catalog_path.resolve()
        with catalog_path.open(newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
        paths = []
        for row in rows:
            local_file = (row.get(local_file_field) or "").strip()
            if not local_file:
                continue
            path = (catalog_path.parent / local_file).resolve()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(path)
    unique = sorted(set(paths), key=lambda path: path.as_posix().casefold())
    if not unique:
        raise FileNotFoundError(f"No catalog-approved images found under {image_dir}")
    outside = [path for path in unique if image_dir not in path.parents]
    if outside:
        raise ValueError(f"Catalog references images outside {image_dir}: {outside[:3]}")
    return unique


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def run_yolox_inference(
    *,
    repo_root: Path,
    image_paths: Sequence[Path],
    experiment_file: Path,
    checkpoint_file: Path,
    class_names: Sequence[str],
    confidence: float = 0.25,
    nms_threshold: float | None = None,
    device_name: str = "auto",
) -> tuple[list[dict], dict]:
    """Run YOLOX and return deterministic, unnumbered crop candidates."""
    import cv2
    import torch

    yolox_root = repo_root / "third_party" / "YOLOX"
    for import_root in (repo_root, yolox_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    from yolox.data import ValTransform
    from yolox.exp import get_exp
    from yolox.utils import postprocess

    if not experiment_file.is_file():
        raise FileNotFoundError(f"Missing experiment definition: {experiment_file}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint: {checkpoint_file}")
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")

    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    experiment = get_exp(str(experiment_file), None)
    if experiment.num_classes != len(class_names):
        raise ValueError(
            f"Experiment expects {experiment.num_classes} classes, but {len(class_names)} names were supplied."
        )
    unknown = sorted(set(class_names) - set(SOURCE6_PART_GROUPS))
    if unknown:
        raise ValueError(f"No part-group mapping for detector classes: {unknown}")

    model = experiment.get_model()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.to(device).eval()
    preprocess = ValTransform(legacy=False)
    nms = experiment.nmsthre if nms_threshold is None else nms_threshold

    candidates: list[dict] = []
    unreadable: list[str] = []
    for image_index, image_path in enumerate(image_paths, start=1):
        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            unreadable.append(str(image_path))
            continue
        ratio = min(
            experiment.test_size[0] / raw_bgr.shape[0],
            experiment.test_size[1] / raw_bgr.shape[1],
        )
        network_input, _ = preprocess(raw_bgr, None, experiment.test_size)
        network_input = torch.from_numpy(network_input).unsqueeze(0).float().to(device)
        with torch.inference_mode():
            detections = postprocess(
                model(network_input),
                experiment.num_classes,
                confidence,
                nms,
            )[0]
        if detections is not None:
            source_asset_id = f"file_sha256:{sha256_file(image_path)}"
            detections = detections.detach().cpu()
            boxes = detections[:, :4] / ratio
            scores = detections[:, 4] * detections[:, 5]
            class_ids = detections[:, 6].to(torch.int64)
            for box, score, class_id_tensor in zip(boxes, scores, class_ids):
                class_id = int(class_id_tensor)
                class_name = class_names[class_id]
                x1, y1, x2, y2 = (float(value) for value in box.tolist())
                x1, x2 = sorted((max(0.0, x1), min(float(raw_bgr.shape[1]), x2)))
                y1, y2 = sorted((max(0.0, y1), min(float(raw_bgr.shape[0]), y2)))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                candidates.append(
                    {
                        "part_group": SOURCE6_PART_GROUPS[class_name],
                        "detector_class": class_name,
                        "detector_confidence": float(score),
                        "source_path": image_path.resolve(),
                        "source_image": _relative_or_absolute(image_path, repo_root),
                        "source_asset_id": source_asset_id,
                        "box": (x1, y1, x2, y2),
                    }
                )
        if image_index % 10 == 0 or image_index == len(image_paths):
            print(f"Processed {image_index}/{len(image_paths)} images; {len(candidates)} crops found", flush=True)

    group_order = {name: index for index, name in enumerate(PART_GROUP_ORDER)}
    candidates.sort(
        key=lambda item: (
            group_order[item["part_group"]],
            item["source_image"].casefold(),
            item["detector_class"],
            *(round(value, 3) for value in item["box"]),
            -item["detector_confidence"],
        )
    )
    metadata = {
        "checkpoint": _relative_or_absolute(checkpoint_file, repo_root),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "experiment_file": _relative_or_absolute(experiment_file, repo_root),
        "class_names": list(class_names),
        "confidence": confidence,
        "nms_threshold": nms,
        "test_size": list(experiment.test_size),
        "device": str(device),
        "source_image_count": len(image_paths),
        "unreadable_images": unreadable,
    }
    return candidates, metadata


def _padded_box(box: Sequence[float], width: int, height: int, padding_fraction: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_fraction
    pad_y = (y2 - y1) * padding_fraction
    return (
        max(0, math.floor(x1 - pad_x)),
        max(0, math.floor(y1 - pad_y)),
        min(width, math.ceil(x2 + pad_x)),
        min(height, math.ceil(y2 + pad_y)),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_contact_sheets(run_dir: Path, rows: Sequence[dict], columns: int = 4, rows_per_page: int = 4) -> None:
    tile_width, tile_height = 300, 240
    title_height, caption_height = 36, 42
    page_size = columns * rows_per_page
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["part_group"]].append(row)

    for group in PART_GROUP_ORDER:
        group_rows = by_group[group]
        sheet_dir = run_dir / "contact_sheets" / group
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for page_index in range(0, len(group_rows), page_size):
            page_rows = group_rows[page_index : page_index + page_size]
            canvas = Image.new("RGB", (columns * tile_width, title_height + rows_per_page * tile_height), "white")
            draw = ImageDraw.Draw(canvas)
            page_number = page_index // page_size + 1
            draw.text((10, 10), f"{group} — page {page_number} — severity: 1 none, 2 moderate, 3 severe", fill="black")
            for offset, row in enumerate(page_rows):
                column = offset % columns
                grid_row = offset // columns
                left = column * tile_width
                top = title_height + grid_row * tile_height
                with Image.open(run_dir / row["crop_file"]) as crop:
                    preview = ImageOps.contain(crop.convert("RGB"), (tile_width - 12, tile_height - caption_height - 12))
                image_left = left + (tile_width - preview.width) // 2
                canvas.paste(preview, (image_left, top + caption_height))
                draw.rectangle((left, top, left + tile_width - 1, top + tile_height - 1), outline="#b0b0b0")
                draw.text((left + 6, top + 5), f"{row['crop_id']} | {row['detector_class']}", fill="black")
                draw.text((left + 6, top + 22), f"detector confidence {float(row['detector_confidence']):.3f}", fill="black")
            canvas.save(sheet_dir / f"page_{page_number:03d}.jpg", quality=92)


def create_annotation_run(
    *,
    candidates: Sequence[dict],
    run_dir: Path,
    repo_root: Path,
    inference_metadata: Mapping,
    crop_padding_fraction: float = 0.08,
) -> Path:
    """Materialize crops, contact sheets, templates, and provenance atomically."""
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"Annotation run already exists: {run_dir}. Choose a new RUN_NAME to protect any labels."
        )
    if not 0 <= crop_padding_fraction <= 0.5:
        raise ValueError("crop_padding_fraction must be between 0 and 0.5")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = run_dir.parent / f".{run_dir.name}.building"
    if build_dir.exists():
        raise FileExistsError(
            f"Incomplete build directory exists: {build_dir}. Inspect and remove it before retrying."
        )
    build_dir.mkdir()
    (build_dir / "label_templates").mkdir()
    (build_dir / "label_uploads").mkdir()

    manifest_rows: list[dict] = []
    current_source_path: Path | None = None
    current_source: Image.Image | None = None
    checkpoint_sha256 = str(inference_metadata["checkpoint_sha256"])
    try:
        for index, candidate in enumerate(candidates, start=1):
            crop_id = f"C{index:06d}"
            source_path = Path(candidate["source_path"])
            if source_path != current_source_path:
                if current_source is not None:
                    current_source.close()
                with Image.open(source_path) as opened:
                    current_source = opened.convert("RGB")
                current_source_path = source_path
            crop_box = _padded_box(
                candidate["box"], current_source.width, current_source.height, crop_padding_fraction
            )
            crop = current_source.crop(crop_box)
            group = candidate["part_group"]
            crop_relative = Path("crops") / group / f"{crop_id}.jpg"
            crop_path = build_dir / crop_relative
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path, quality=95)
            x1, y1, x2, y2 = candidate["box"]
            cx1, cy1, cx2, cy2 = crop_box
            manifest_rows.append(
                {
                    "crop_id": crop_id,
                    "part_group": group,
                    "detector_class": candidate["detector_class"],
                    "detector_confidence": f"{candidate['detector_confidence']:.8f}",
                    "source_image": candidate["source_image"],
                    "source_asset_id": candidate["source_asset_id"],
                    "box_x1": f"{x1:.3f}",
                    "box_y1": f"{y1:.3f}",
                    "box_x2": f"{x2:.3f}",
                    "box_y2": f"{y2:.3f}",
                    "crop_x1": cx1,
                    "crop_y1": cy1,
                    "crop_x2": cx2,
                    "crop_y2": cy2,
                    "crop_width": crop.width,
                    "crop_height": crop.height,
                    "crop_file": crop_relative.as_posix(),
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
        if current_source is not None:
            current_source.close()

        _write_csv(build_dir / "crops_manifest.csv", MANIFEST_FIELDS, manifest_rows)
        for group in PART_GROUP_ORDER:
            template_rows = (
                {"crop_id": row["crop_id"], "damage_severity": ""}
                for row in manifest_rows
                if row["part_group"] == group
            )
            _write_csv(
                build_dir / "label_templates" / f"{group}.csv",
                ("crop_id", "damage_severity"),
                template_rows,
            )
        _create_contact_sheets(build_dir, manifest_rows)
        metadata = {
            **dict(inference_metadata),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_dir.name,
            "crop_padding_fraction": crop_padding_fraction,
            "crop_count": len(manifest_rows),
            "crop_counts_by_group": dict(Counter(row["part_group"] for row in manifest_rows)),
            "severity_rubric": {str(key): value for key, value in SEVERITY_NAMES.items()},
            "crop_id_order": "part group, source image, detector class, box coordinates, confidence",
            "note": "Predicted detector crops; not ground-truth component boxes.",
        }
        (build_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        run_dir.parent.joinpath(run_dir.name).parent.mkdir(parents=True, exist_ok=True)
        build_dir.replace(run_dir)
    except Exception:
        if current_source is not None:
            current_source.close()
        raise
    return run_dir


def contact_sheet_paths(run_dir: Path) -> dict[str, list[Path]]:
    return {
        group: sorted((run_dir / "contact_sheets" / group).glob("page_*.jpg"))
        for group in PART_GROUP_ORDER
    }


def validate_and_merge_labels(run_dir: Path, uploaded_csvs: Mapping[str, Path]) -> Path:
    """Validate uploads and join only crops with a nonblank severity label."""
    with (run_dir / "crops_manifest.csv").open(newline="", encoding="utf-8-sig") as source:
        manifest_rows = list(csv.DictReader(source))
    manifest_by_id = {row["crop_id"]: row for row in manifest_rows}
    expected_by_group = {
        group: {row["crop_id"] for row in manifest_rows if row["part_group"] == group}
        for group in PART_GROUP_ORDER
    }
    errors: list[str] = []
    labels: dict[str, int] = {}
    for group in PART_GROUP_ORDER:
        path = Path(uploaded_csvs[group])
        if not path.is_file():
            errors.append(f"{group}: missing upload {path}")
            continue
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != ["crop_id", "damage_severity"]:
                errors.append(f"{group}: columns must be exactly crop_id,damage_severity")
                continue
            rows = list(reader)
        ids = [row["crop_id"].strip() for row in rows]
        duplicates = sorted(crop_id for crop_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            errors.append(f"{group}: duplicate IDs {duplicates[:10]}")
        observed = set(ids)
        unknown = sorted(observed - expected_by_group[group])
        if unknown:
            errors.append(f"{group}: {len(unknown)} unknown IDs, first: {unknown[:10]}")
        for row in rows:
            crop_id = row["crop_id"].strip()
            severity_text = row["damage_severity"].strip()
            if not severity_text:
                continue
            if severity_text not in {"1", "2", "3"}:
                errors.append(f"{group}: {crop_id} has invalid severity {severity_text!r}")
            elif crop_id in expected_by_group[group]:
                labels[crop_id] = int(severity_text)
    if errors:
        raise ValueError("Label validation failed:\n- " + "\n- ".join(errors))

    output_fields = list(MANIFEST_FIELDS) + ["damage_severity", "damage_severity_name"]
    if not labels:
        raise ValueError("Label validation failed: no crops have a severity label.")
    output_rows = []
    for crop_id in sorted(labels):
        severity = labels[crop_id]
        output_rows.append(
            {
                **manifest_by_id[crop_id],
                "damage_severity": severity,
                "damage_severity_name": SEVERITY_NAMES[severity],
            }
        )
    output = run_dir / "validated_damage_labels.csv"
    _write_csv(output, output_fields, output_rows)
    return output


def _split_source_assets(source_asset_ids: Sequence[str], ratios: Sequence[float], seed: int) -> dict[str, str]:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must be three nonnegative values summing to 1")
    assets = sorted(set(source_asset_ids))
    random.Random(seed).shuffle(assets)
    count = len(assets)
    if count < 3:
        return {asset: "train" for asset in assets}
    validation_count = max(1, round(count * ratios[1])) if ratios[1] else 0
    test_count = max(1, round(count * ratios[2])) if ratios[2] else 0
    while validation_count + test_count >= count:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
    assignments = {}
    for index, asset in enumerate(assets):
        if index < test_count:
            split = "test"
        elif index < test_count + validation_count:
            split = "val"
        else:
            split = "train"
        assignments[asset] = split
    return assignments


def create_head_splits(
    *,
    run_dir: Path,
    validated_labels_csv: Path,
    output_dir: Path,
    split_ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Path:
    """Create per-head crop datasets while keeping every source image in one split."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Prepared dataset already exists: {output_dir}")
    with validated_labels_csv.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("No labeled crops are available for splitting.")
    with (run_dir / "crops_manifest.csv").open(newline="", encoding="utf-8-sig") as source:
        total_crop_count = sum(1 for _ in csv.DictReader(source))
    assignments = _split_source_assets([row["source_asset_id"] for row in rows], split_ratios, seed)
    build_dir = output_dir.parent / f".{output_dir.name}.building"
    if build_dir.exists():
        raise FileExistsError(f"Incomplete prepared-dataset build exists: {build_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir()

    label_fields = list(rows[0].keys()) + ["split"]
    split_rows = []
    for group in PART_GROUP_ORDER:
        group_root = build_dir / group
        for split in ("train", "val", "test"):
            (group_root / "images" / split).mkdir(parents=True, exist_ok=True)
        group_rows = []
        for row in rows:
            if row["part_group"] != group:
                continue
            split = assignments[row["source_asset_id"]]
            source_crop = run_dir / row["crop_file"]
            if not source_crop.is_file():
                raise FileNotFoundError(f"Missing crop referenced by labels: {source_crop}")
            destination_crop = group_root / "images" / split / source_crop.name
            shutil.copy2(source_crop, destination_crop)
            group_rows.append({**row, "split": split})
        _write_csv(group_root / "labels.csv", label_fields, group_rows)

    source_images: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        source_images[row["source_asset_id"]].add(row["source_image"])
    for source_asset_id in sorted(assignments):
        split_rows.append(
            {
                "source_asset_id": source_asset_id,
                "split": assignments[source_asset_id],
                "source_images": ";".join(sorted(source_images[source_asset_id])),
            }
        )
    _write_csv(build_dir / "source_split_manifest.csv", ("source_asset_id", "split", "source_images"), split_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_annotation_run": str(run_dir),
        "validated_labels_csv": str(validated_labels_csv.resolve()),
        "split_ratios": list(split_ratios),
        "seed": seed,
        "labeled_crop_count": len(rows),
        "excluded_unlabeled_crop_count": total_crop_count - len(rows),
        "counts_by_head": dict(Counter(row["part_group"] for row in rows)),
        "counts_by_split": dict(Counter(assignments[row["source_asset_id"]] for row in rows)),
    }
    (build_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    build_dir.replace(output_dir)
    return output_dir
