#!/usr/bin/env python3
"""Build the leakage-resistant COCO Carparts23 dataset for CRATER.

The input is Ultralytics' packaged Carparts-Seg v4 archive after extraction.
The script:

1. validates every image/label pair and YOLO segmentation polygon;
2. removes images with empty label files (they visibly contain cars and should
   not be treated as detector negatives);
3. groups Roboflow augmentations using the pre-``.rf.`` filename plus
   rotation- and color-tolerant visual similarity;
4. creates deterministic 70/15/15 source-group splits;
5. keeps all usable variants for training and one representative per source
   image for validation/test;
6. writes exact 23-class COCO annotations and a small CRATER transfer view;
7. emits an audit report, class counts, manifests, and visual previews.

Pillow, NumPy, and SciPy are required. No Ultralytics software is imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, rotate


SOURCE_CLASSES = [
    "back_bumper",
    "back_door",
    "back_glass",
    "back_left_door",
    "back_left_light",
    "back_light",
    "back_right_door",
    "back_right_light",
    "front_bumper",
    "front_door",
    "front_glass",
    "front_left_door",
    "front_left_light",
    "front_light",
    "front_right_door",
    "front_right_light",
    "hood",
    "left_mirror",
    "object",
    "right_mirror",
    "tailgate",
    "trunk",
    "wheel",
]

# Exact classes that transfer directly plus a deliberately named weak proxy.
TRANSFER_CLASSES = ["windshield", "wheel", "hull_proxy"]
WINDSHIELD_SOURCE_ID = 10
WHEEL_SOURCE_ID = 22
HULL_PROXY_SOURCE_IDS = {
    0,   # back_bumper
    1,   # back_door
    3,   # back_left_door
    6,   # back_right_door
    8,   # front_bumper
    9,   # front_door
    11,  # front_left_door
    14,  # front_right_door
    16,  # hood
    20,  # tailgate
    21,  # trunk
}

SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

SIMILARITY_IMAGE_SIZE = 48
SIMILARITY_ROTATIONS = sorted(
    {
        base_angle + delta
        for base_angle in (0, 90, 180, 270)
        for delta in range(-30, 31, 4)
    }
)
_SIM_Y, _SIM_X = np.ogrid[:SIMILARITY_IMAGE_SIZE, :SIMILARITY_IMAGE_SIZE]
SIMILARITY_MASK = (
    (_SIM_X - (SIMILARITY_IMAGE_SIZE - 1) / 2) ** 2
    + (_SIM_Y - (SIMILARITY_IMAGE_SIZE - 1) / 2) ** 2
    < (SIMILARITY_IMAGE_SIZE * 0.42) ** 2
)

DATASET_SOURCE_URL = (
    "https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm/dataset/4"
)
DOWNLOAD_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "carparts-seg.zip"
)
LICENSE_NAME = "Creative Commons Attribution 4.0 International"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ARCHIVE_SHA256 = "e2eea20030d02366b07174bcdbcc843f63791bdd53962a83e00636d15e78e41f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted Carparts-Seg root")
    parser.add_argument("output", type=Path, help="New prepared dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-search-attempts",
        type=int,
        default=512,
        help="Random grouped splits evaluated for class balance",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Normalized-correlation threshold for grouping augmented variants",
    )
    parser.add_argument(
        "--similarity-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Worker processes used for visual source grouping",
    )
    return parser.parse_args()


def source_key(path: Path) -> str:
    """Return the shared pre-augmentation key from a Roboflow filename."""
    return path.stem.split(".rf.", 1)[0]


def polygon_area(points: list[float]) -> float:
    coordinates = list(zip(points[0::2], points[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                coordinates, coordinates[1:] + coordinates[:1]
            )
        )
    ) / 2.0


def parse_label(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        tokens = line.split()
        try:
            class_id = int(tokens[0])
            normalized = [float(value) for value in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"Malformed label at {path}:{line_number}") from exc

        if not 0 <= class_id < len(SOURCE_CLASSES):
            raise ValueError(f"Invalid class id {class_id} at {path}:{line_number}")
        if len(normalized) < 6 or len(normalized) % 2:
            raise ValueError(f"Invalid polygon at {path}:{line_number}")
        if not all(math.isfinite(value) and -1e-6 <= value <= 1.0 + 1e-6 for value in normalized):
            raise ValueError(f"Out-of-range polygon at {path}:{line_number}")

        pixel_points: list[float] = []
        for x_value, y_value in zip(normalized[0::2], normalized[1::2]):
            pixel_points.extend(
                [
                    min(max(x_value, 0.0), 1.0) * width,
                    min(max(y_value, 0.0), 1.0) * height,
                ]
            )

        x_values = pixel_points[0::2]
        y_values = pixel_points[1::2]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        box_width, box_height = x_max - x_min, y_max - y_min
        area = polygon_area(pixel_points)
        if box_width <= 0 or box_height <= 0 or area <= 0:
            raise ValueError(f"Degenerate polygon at {path}:{line_number}")

        instances.append(
            {
                "source_class_id": class_id,
                "segmentation": pixel_points,
                "bbox": [x_min, y_min, box_width, box_height],
                "area": area,
            }
        )
    return instances


def discover_records(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing_labels: list[str] = []
    extra_labels: list[str] = []

    image_paths: dict[tuple[str, str], Path] = {}
    label_paths: dict[tuple[str, str], Path] = {}
    for split in SPLITS:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing expected split directories for {split}")

        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_paths[(split, path.stem)] = path
        for path in sorted(label_dir.glob("*.txt")):
            label_paths[(split, path.stem)] = path

    for key in sorted(image_paths):
        image_path = image_paths[key]
        label_path = label_paths.get(key)
        if label_path is None:
            missing_labels.append(str(image_path.relative_to(source)))
            continue

        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
        instances = parse_label(label_path, width, height)
        base_key = source_key(image_path)
        records.append(
            {
                "source_image_path": image_path,
                "source_label_path": label_path,
                "source_file_name": str(image_path.relative_to(source)),
                "source_split": key[0],
                "base_source_key": base_key,
                "source_key": base_key,
                "width": width,
                "height": height,
                "instances": instances,
            }
        )

    for key, label_path in sorted(label_paths.items()):
        if key not in image_paths:
            extra_labels.append(str(label_path.relative_to(source)))

    if missing_labels or extra_labels:
        raise ValueError(
            f"Image/label mismatch: {len(missing_labels)} missing labels, "
            f"{len(extra_labels)} extra labels"
        )

    profile = {
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "missing_labels": missing_labels,
        "extra_labels": extra_labels,
    }
    return records, profile


def _load_similarity_image(path: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize(
                (SIMILARITY_IMAGE_SIZE, SIMILARITY_IMAGE_SIZE),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.float32,
        )
    return gaussian_filter(array, 0.7)


def _maximum_rotated_correlation(first: np.ndarray, second: np.ndarray) -> float:
    best = -2.0
    for angle in SIMILARITY_ROTATIONS:
        rotated = rotate(
            second,
            angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        mask = SIMILARITY_MASK & np.isfinite(rotated)
        first_values = first[mask]
        second_values = rotated[mask]
        first_values = first_values - first_values.mean()
        second_values = second_values - second_values.mean()
        denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
        score = float(np.dot(first_values, second_values) / (denominator + 1e-8))
        best = max(best, score)
    return best


def _cluster_path_group(
    item: tuple[str, list[str], float]
) -> tuple[str, list[list[str]], list[float], int]:
    base_key, paths, threshold = item
    arrays = [_load_similarity_image(path) for path in paths]
    parent = list(range(len(paths)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[second_root] = first_root

    scores: list[float] = []
    matched_edges = 0
    for first_index in range(len(paths)):
        for second_index in range(first_index + 1, len(paths)):
            score = _maximum_rotated_correlation(
                arrays[first_index], arrays[second_index]
            )
            scores.append(score)
            if score >= threshold:
                matched_edges += 1
                union(first_index, second_index)

    components: dict[int, list[str]] = defaultdict(list)
    for index, path in enumerate(paths):
        components[find(index)].append(path)
    ordered_components = sorted(
        (sorted(component) for component in components.values()),
        key=lambda component: component[0],
    )
    return base_key, ordered_components, scores, matched_edges


def refine_source_groups(
    records: list[dict[str, Any]], threshold: float, workers: int
) -> dict[str, Any]:
    """Separate reused filenames while retaining visually matched variants."""
    base_groups: dict[str, list[str]] = defaultdict(list)
    records_by_path = {
        str(record["source_image_path"]): record for record in records
    }
    for record in records:
        base_groups[record["base_source_key"]].append(
            str(record["source_image_path"])
        )

    items = [
        (base_key, sorted(paths), threshold)
        for base_key, paths in sorted(base_groups.items())
    ]
    if workers <= 1:
        results = map(_cluster_path_group, items)
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("fork") if os.name != "nt" else None,
        )
        results = executor.map(_cluster_path_group, items, chunksize=4)

    all_scores: list[float] = []
    matched_edges = 0
    refined_group_count = 0
    try:
        for base_key, components, scores, group_matched_edges in results:
            all_scores.extend(scores)
            matched_edges += group_matched_edges
            refined_group_count += len(components)
            for component_index, component in enumerate(components):
                refined_key = f"{base_key}__visual_{component_index:02d}"
                for path in component:
                    records_by_path[path]["source_key"] = refined_key
    finally:
        if workers > 1:
            executor.shutdown()

    quantiles = {}
    if all_scores:
        quantiles = {
            str(quantile): round(float(np.quantile(all_scores, quantile)), 6)
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        }
    return {
        "method": (
            "filename prefix followed by connected components of maximum "
            "rotation-tolerant grayscale normalized correlation"
        ),
        "similarity_threshold": threshold,
        "tested_rotations_degrees": SIMILARITY_ROTATIONS,
        "filename_prefix_groups": len(base_groups),
        "refined_visual_source_groups": refined_group_count,
        "pair_comparisons": len(all_scores),
        "matched_pair_edges": matched_edges,
        "similarity_score_quantiles": quantiles,
    }


def class_presence_by_group(
    records_by_group: dict[str, list[dict[str, Any]]]
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for group_key, records in records_by_group.items():
        result[group_key] = {
            instance["source_class_id"]
            for record in records
            for instance in record["instances"]
        }
    return result


def choose_group_split(
    records_by_group: dict[str, list[dict[str, Any]]],
    seed: int,
    attempts: int,
) -> tuple[dict[str, str], float]:
    """Pick a deterministic grouped split with reasonable class coverage."""
    group_keys = sorted(records_by_group)
    group_count = len(group_keys)
    train_count = round(group_count * SPLIT_RATIOS["train"])
    val_count = round(group_count * SPLIT_RATIOS["val"])
    group_presence = class_presence_by_group(records_by_group)
    total_presence = Counter(
        class_id for classes in group_presence.values() for class_id in classes
    )

    best_assignment: dict[str, str] | None = None
    best_score = math.inf
    for attempt in range(attempts):
        shuffled = group_keys.copy()
        random.Random(seed + attempt).shuffle(shuffled)
        assignment = {
            key: (
                "train"
                if index < train_count
                else "val"
                if index < train_count + val_count
                else "test"
            )
            for index, key in enumerate(shuffled)
        }

        split_presence = {split: Counter() for split in SPLITS}
        for key, classes in group_presence.items():
            split_presence[assignment[key]].update(classes)

        score = 0.0
        compared = 0
        for class_id, total in total_presence.items():
            # The catch-all "object" class is too rare and semantically weak to
            # drive the split, but it remains in the full COCO annotations.
            if class_id == 18 or total < 6:
                continue
            for split in SPLITS:
                observed_share = split_presence[split][class_id] / total
                score += abs(observed_share - SPLIT_RATIOS[split])
                compared += 1
        score /= max(compared, 1)

        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not construct grouped split")
    return best_assignment, best_score


def choose_eval_representative(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the most completely annotated deterministic variant."""
    return sorted(
        records,
        key=lambda record: (-len(record["instances"]), record["source_file_name"]),
    )[0]


def select_records(
    records_by_group: dict[str, list[dict[str, Any]]],
    assignment: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected = {split: [] for split in SPLITS}
    omitted_eval_variants: list[dict[str, Any]] = []

    for key in sorted(records_by_group):
        split = assignment[key]
        group_records = records_by_group[key]
        if split == "train":
            selected[split].extend(group_records)
            continue

        representative = choose_eval_representative(group_records)
        selected[split].append(representative)
        omitted_eval_variants.extend(
            record for record in group_records if record is not representative
        )

    for split in SPLITS:
        selected[split].sort(key=lambda record: record["source_file_name"])
    return selected, omitted_eval_variants


def prepare_images(
    selected: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    for split in SPLITS:
        destination_dir = output / "images" / split
        destination_dir.mkdir(parents=True, exist_ok=False)
        used_names: set[str] = set()
        for record in selected[split]:
            file_name = record["source_image_path"].name
            if file_name in used_names:
                raise ValueError(f"Prepared filename collision: {file_name}")
            used_names.add(file_name)
            destination = destination_dir / file_name
            shutil.copy2(record["source_image_path"], destination)
            record["prepared_file_name"] = f"images/{split}/{file_name}"


def full_instances(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "category_id": instance["source_class_id"] + 1,
            "segmentation": [instance["segmentation"]],
            "bbox": instance["bbox"],
            "area": instance["area"],
        }
        for instance in record["instances"]
    ]


def transfer_instances(record: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    hull_segments: list[list[float]] = []
    hull_boxes: list[list[float]] = []
    hull_area = 0.0

    for instance in record["instances"]:
        source_id = instance["source_class_id"]
        if source_id == WINDSHIELD_SOURCE_ID:
            result.append(
                {
                    "category_id": 1,
                    "segmentation": [instance["segmentation"]],
                    "bbox": instance["bbox"],
                    "area": instance["area"],
                }
            )
        elif source_id == WHEEL_SOURCE_ID:
            result.append(
                {
                    "category_id": 2,
                    "segmentation": [instance["segmentation"]],
                    "bbox": instance["bbox"],
                    "area": instance["area"],
                }
            )

        if source_id in HULL_PROXY_SOURCE_IDS:
            hull_segments.append(instance["segmentation"])
            hull_boxes.append(instance["bbox"])
            hull_area += instance["area"]

    if hull_boxes:
        x_min = min(box[0] for box in hull_boxes)
        y_min = min(box[1] for box in hull_boxes)
        x_max = max(box[0] + box[2] for box in hull_boxes)
        y_max = max(box[1] + box[3] for box in hull_boxes)
        result.append(
            {
                "category_id": 3,
                "segmentation": hull_segments,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": hull_area,
            }
        )
    return result


def coco_document(
    records: list[dict[str, Any]],
    split: str,
    annotation_set: str,
) -> dict[str, Any]:
    if annotation_set == "carparts23":
        categories = [
            {"id": index + 1, "name": name, "supercategory": "vehicle_part"}
            for index, name in enumerate(SOURCE_CLASSES)
        ]
        instance_builder = full_instances
        description = "Carparts-Seg v4 converted from polygons to COCO boxes and masks"
    elif annotation_set == "crater_transfer3":
        categories = [
            {"id": index + 1, "name": name, "supercategory": "vehicle_component"}
            for index, name in enumerate(TRANSFER_CLASSES)
        ]
        instance_builder = transfer_instances
        description = (
            "Civilian transfer labels for CRATER; hull_proxy is not military hull ground truth"
        )
    else:
        raise ValueError(f"Unknown annotation set: {annotation_set}")

    images = []
    annotations = []
    annotation_id = 1
    for image_id, record in enumerate(records, 1):
        images.append(
            {
                "id": image_id,
                "file_name": record["prepared_file_name"],
                "width": record["width"],
                "height": record["height"],
                "license": 1,
                "source_group": record["source_key"],
                "original_split": record["source_split"],
            }
        )
        for instance in instance_builder(record):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": instance["category_id"],
                    "bbox": [round(value, 4) for value in instance["bbox"]],
                    "area": round(instance["area"], 4),
                    "segmentation": [
                        [round(value, 4) for value in segment]
                        for segment in instance["segmentation"]
                    ],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": description,
            "version": "1.0",
            "source": DATASET_SOURCE_URL,
            "split": split,
            "split_policy": (
                "70/15/15 source-group split; all train variants; "
                "one representative per validation/test source group"
            ),
        },
        "licenses": [{"id": 1, "name": LICENSE_NAME, "url": LICENSE_URL}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_annotations(
    selected: dict[str, list[dict[str, Any]]], output: Path
) -> dict[str, dict[str, dict[str, int]]]:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    annotation_dir = output / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=False)
    for annotation_set in ("carparts23", "crater_transfer3"):
        summary[annotation_set] = {}
        for split in SPLITS:
            document = coco_document(selected[split], split, annotation_set)
            write_json(
                annotation_dir / f"{annotation_set}_instances_{split}.json",
                document,
            )
            summary[annotation_set][split] = {
                "images": len(document["images"]),
                "instances": len(document["annotations"]),
            }
    return summary


def write_manifests(
    selected: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    manifest_dir = output / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        text = "\n".join(record["prepared_file_name"] for record in selected[split])
        (manifest_dir / f"{split}.txt").write_text(text + "\n", encoding="utf-8")


def original_leakage_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["source_key"]].append(record)
    cross_split = {
        key: group_records
        for key, group_records in groups.items()
        if len({record["source_split"] for record in group_records}) > 1
    }
    affected_images = sum(len(group_records) for group_records in cross_split.values())
    return {
        "source_group_count": len(groups),
        "cross_split_source_groups": len(cross_split),
        "cross_split_source_group_rate": round(len(cross_split) / len(groups), 6),
        "images_in_cross_split_groups": affected_images,
        "image_rate_in_cross_split_groups": round(affected_images / len(records), 6),
    }


def instance_counts(
    records: list[dict[str, Any]], builder: Any, names: list[str]
) -> dict[str, int]:
    counts = Counter(
        instance["category_id"]
        for record in records
        for instance in builder(record)
    )
    return {name: counts[index + 1] for index, name in enumerate(names)}


def write_class_counts(
    selected: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    path = output / "audit" / "class_counts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "annotation_set",
                "split",
                "category_id",
                "category_name",
                "instance_count",
                "image_count",
            ],
        )
        writer.writeheader()
        for annotation_set, names, builder in (
            ("carparts23", SOURCE_CLASSES, full_instances),
            ("crater_transfer3", TRANSFER_CLASSES, transfer_instances),
        ):
            for split in SPLITS:
                instances = Counter(
                    instance["category_id"]
                    for record in selected[split]
                    for instance in builder(record)
                )
                images = Counter()
                for record in selected[split]:
                    images.update(
                        {instance["category_id"] for instance in builder(record)}
                    )
                for category_id, name in enumerate(names, 1):
                    writer.writerow(
                        {
                            "annotation_set": annotation_set,
                            "split": split,
                            "category_id": category_id,
                            "category_name": name,
                            "instance_count": instances[category_id],
                            "image_count": images[category_id],
                        }
                    )


def write_assignments(
    records: list[dict[str, Any]],
    assignment: dict[str, str],
    selected: dict[str, list[dict[str, Any]]],
    output: Path,
) -> None:
    selected_ids = {id(record) for split in SPLITS for record in selected[split]}
    path = output / "audit" / "source_assignments.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename_prefix_group",
                "source_group",
                "prepared_split",
                "selected",
                "selection_reason",
                "original_split",
                "original_file",
                "instance_count",
            ],
        )
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["source_file_name"]):
            is_selected = id(record) in selected_ids
            if not record["instances"]:
                reason = "excluded_empty_label"
            elif is_selected:
                reason = "selected"
            else:
                reason = "omitted_correlated_eval_variant"
            writer.writerow(
                {
                    "filename_prefix_group": record["base_source_key"],
                    "source_group": record["source_key"],
                    "prepared_split": assignment.get(record["source_key"], ""),
                    "selected": str(is_selected).lower(),
                    "selection_reason": reason,
                    "original_split": record["source_split"],
                    "original_file": record["source_file_name"],
                    "instance_count": len(record["instances"]),
                }
            )


COLORS = {
    1: "#13B5EA",
    2: "#F5A623",
    3: "#7ED321",
}


def draw_preview_tile(record: dict[str, Any], size: int = 320) -> Image.Image:
    with Image.open(record["source_image_path"]) as original:
        image = original.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    tile = Image.new("RGB", (size, size + 26), "white")
    offset_x = (size - resized.width) // 2
    offset_y = (size - resized.height) // 2
    tile.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()

    for instance in transfer_instances(record):
        x, y, width, height = instance["bbox"]
        box = [
            offset_x + x * scale,
            offset_y + y * scale,
            offset_x + (x + width) * scale,
            offset_y + (y + height) * scale,
        ]
        category_id = instance["category_id"]
        color = COLORS[category_id]
        draw.rectangle(box, outline=color, width=3)
        label = TRANSFER_CLASSES[category_id - 1]
        label_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((box[0], box[1]), label, fill="black", font=font)

    caption = record["source_key"][:42]
    draw.text((6, size + 7), caption, fill="black", font=font)
    return tile


def write_sample_preview(
    selected: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    candidates = [
        record
        for record in selected["train"]
        if len(transfer_instances(record)) >= 2
    ]
    candidates.sort(
        key=lambda record: (-len(transfer_instances(record)), record["source_file_name"])
    )
    chosen: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for record in candidates:
        if record["source_key"] in seen_groups:
            continue
        chosen.append(record)
        seen_groups.add(record["source_key"])
        if len(chosen) == 6:
            break

    canvas = Image.new("RGB", (960, 692), "#EDEDED")
    for index, record in enumerate(chosen):
        tile = draw_preview_tile(record)
        canvas.paste(tile, ((index % 3) * 320, (index // 3) * 346))
    canvas.save(output / "audit" / "sample_crater_transfer3_boxes.jpg", quality=92)


def write_leakage_preview(records: list[dict[str, Any]], output: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["source_key"]].append(record)
    candidates = [
        (key, group_records)
        for key, group_records in groups.items()
        if len({record["source_split"] for record in group_records}) > 1
        and len(group_records) >= 4
    ]
    key, group_records = sorted(candidates, key=lambda item: item[0])[0]
    group_records = sorted(group_records, key=lambda record: record["source_file_name"])[:8]

    tile_size = 240
    canvas = Image.new("RGB", (tile_size * 4, (tile_size + 28) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, record in enumerate(group_records):
        with Image.open(record["source_image_path"]) as original:
            image = original.convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (index % 4) * tile_size
        y = (index // 4) * (tile_size + 28)
        canvas.paste(image, (x, y))
        draw.text(
            (x + 5, y + tile_size + 7),
            f"{key} | original {record['source_split']}",
            fill="black",
            font=font,
        )
    canvas.save(output / "audit" / "original_split_leakage_example.jpg", quality=92)


def build_audit_report(
    all_records: list[dict[str, Any]],
    usable_records: list[dict[str, Any]],
    profile: dict[str, Any],
    grouping_profile: dict[str, Any],
    assignment: dict[str, str],
    split_score: float,
    selected: dict[str, list[dict[str, Any]]],
    omitted_eval_variants: list[dict[str, Any]],
    annotation_summary: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    original_split_images = Counter(record["source_split"] for record in all_records)
    original_split_instances = Counter()
    for record in all_records:
        original_split_instances[record["source_split"]] += len(record["instances"])

    prepared_group_counts = Counter(assignment.values())
    selected_group_sets = {
        split: {record["source_key"] for record in selected[split]} for split in SPLITS
    }
    overlap = {
        "train_val": len(selected_group_sets["train"] & selected_group_sets["val"]),
        "train_test": len(selected_group_sets["train"] & selected_group_sets["test"]),
        "val_test": len(selected_group_sets["val"] & selected_group_sets["test"]),
    }

    return {
        "dataset": {
            "name": "Carparts-Seg v4 / CRATER starter preparation",
            "source_url": DATASET_SOURCE_URL,
            "download_url": DOWNLOAD_URL,
            "license": LICENSE_NAME,
            "license_url": LICENSE_URL,
            "archive_sha256": ARCHIVE_SHA256,
        },
        "grain": {
            "raw_record": "one Roboflow-generated image variant plus polygon label file",
            "source_group": (
                "visually matched variants within a shared filename prefix before .rf."
            ),
            "prepared_eval_record": "one representative variant per source group",
        },
        "pair_and_schema_checks": {
            **profile,
            "validated_images": len(all_records),
            "invalid_polygons": 0,
            "observed_original_split_images": dict(original_split_images),
            "observed_original_split_instances": dict(original_split_instances),
            "packaged_yaml_comment_mismatch": {
                "yaml_comments": {"train": 3516, "val": 276, "test": 401},
                "observed": dict(original_split_images),
            },
        },
        "quality_findings": {
            "empty_label_images": len(all_records) - len(usable_records),
            "empty_label_image_rate": round(
                (len(all_records) - len(usable_records)) / len(all_records), 6
            ),
            "empty_label_policy": "excluded; sampled files visibly contain vehicles/parts",
            "source_grouping": grouping_profile,
            "original_split_leakage": original_leakage_profile(all_records),
            "leakage_policy": "resplit by source group",
            "correlated_eval_variants_omitted": len(omitted_eval_variants),
        },
        "prepared_split": {
            "seed": 42,
            "ratios": SPLIT_RATIOS,
            "class_balance_search_score": round(split_score, 8),
            "source_groups": dict(prepared_group_counts),
            "selected_images": {split: len(selected[split]) for split in SPLITS},
            "source_group_overlap": overlap,
            "annotation_sets": annotation_summary,
        },
        "transfer_scope": {
            "direct": {
                "windshield": "front_glass",
                "wheel": "wheel",
            },
            "weak_proxy": {
                "hull_proxy": [SOURCE_CLASSES[index] for index in sorted(HULL_PROXY_SOURCE_IDS)]
            },
            "not_present": ["track", "comms_system", "mounted_gun"],
            "warning": (
                "Civilian labels are pretraining only. They do not validate component "
                "detection on overhead/oblique military imagery."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    all_records, profile = discover_records(source)
    grouping_profile = refine_source_groups(
        all_records,
        threshold=args.similarity_threshold,
        workers=args.similarity_workers,
    )
    usable_records = [record for record in all_records if record["instances"]]
    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in usable_records:
        records_by_group[record["source_key"]].append(record)

    assignment, split_score = choose_group_split(
        records_by_group,
        seed=args.seed,
        attempts=args.split_search_attempts,
    )
    selected, omitted_eval_variants = select_records(records_by_group, assignment)

    prepare_images(selected, output)
    annotation_summary = write_annotations(selected, output)
    write_manifests(selected, output)
    (output / "audit").mkdir(parents=True, exist_ok=False)
    write_class_counts(selected, output)
    write_assignments(all_records, assignment, selected, output)
    write_sample_preview(selected, output)
    write_leakage_preview(all_records, output)

    report = build_audit_report(
        all_records,
        usable_records,
        profile,
        grouping_profile,
        assignment,
        split_score,
        selected,
        omitted_eval_variants,
        annotation_summary,
    )
    # Preserve the caller-provided seed accurately in the report.
    report["prepared_split"]["seed"] = args.seed
    write_json(output / "audit" / "audit_report.json", report)

    tools_dir = output / "tools"
    tools_dir.mkdir()
    shutil.copy2(Path(__file__), tools_dir / Path(__file__).name)

    print(json.dumps(report["prepared_split"], indent=2))


if __name__ == "__main__":
    main()
