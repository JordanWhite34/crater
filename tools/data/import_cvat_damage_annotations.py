"""Validate and import human CVAT component-damage annotations.

The source images remain in the immutable Humvee component export. This tool
stores a privacy-sanitized CVAT XML record and a flat, canonical damage-box
inventory. It deliberately does not assign train/validation/test splits or
materialize crops; source grouping must be resolved before either operation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "datasets"
    / "military"
    / "damage"
    / "source"
    / "humvee_cvat_damage_v1"
)
DEFAULT_SOURCE_IMAGE_DIR = (
    REPOSITORY_ROOT / "datasets" / "military" / "components" / "source" / "humvee"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_SOURCE_IMAGE_DIR / "source_manifest.csv"
DEFAULT_COMPONENT_ANNOTATIONS = DEFAULT_SOURCE_IMAGE_DIR / "instances_default.json"

SOURCE_LABELS = {
    "wheel_tire",
    "windshield",
    "door",
    "engine_bay",
    "weapons_station",
    "comms_equipment",
}
LABEL_ALIASES = {"weapons_station": "weapon_station"}
PART_GROUPS = {
    "wheel_tire": "mobility",
    "windshield": "structure",
    "door": "structure",
    "engine_bay": "structure",
    "weapon_station": "mission_equipment",
    "comms_equipment": "mission_equipment",
}
DAMAGE_LEVELS = {
    "intact": "no_visible_damage",
    "degraded": "possible_damage",
    "destroyed": "severe_visible_damage",
    "unknown": "unobservable",
}
CSV_FIELDS = (
    "annotation_id",
    "cvat_image_id",
    "cvat_box_index",
    "source_image",
    "source_image_sha256",
    "component_class_source",
    "component_class",
    "component_group",
    "damage_state_source",
    "damage_level",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "box_width",
    "box_height",
    "box_area",
    "occluded",
    "annotation_source",
    "split",
)
FORBIDDEN_XML_MARKERS = (b"<!DOCTYPE", b"<!ENTITY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cvat_xml(path: Path) -> tuple[ET.ElementTree, bytes]:
    raw = path.read_bytes()
    upper = raw.upper()
    if any(marker in upper for marker in FORBIDDEN_XML_MARKERS):
        raise ValueError("CVAT XML must not contain DTD or entity declarations.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError(f"Invalid CVAT XML: {error}") from error
    if root.tag != "annotations":
        raise ValueError(f"Expected <annotations> root, found <{root.tag}>.")
    return ET.ElementTree(root), raw


def _required_text(parent: ET.Element, path: str) -> str:
    node = parent.find(path)
    if node is None or not (node.text or "").strip():
        raise ValueError(f"Missing required CVAT field: {path}")
    return (node.text or "").strip()


def _find_metadata(root: ET.Element) -> ET.Element:
    metadata = root.find("./meta/job")
    if metadata is None:
        metadata = root.find("./meta/task")
    if metadata is None:
        raise ValueError("CVAT XML is missing job/task metadata.")
    return metadata


def _load_source_manifest(path: Path, image_dir: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    required = {"file_name", "sha256", "width", "height"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Source manifest must contain {sorted(required)}.")
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["file_name"].strip()
        if name in by_name:
            raise ValueError(f"Duplicate source-manifest filename: {name}")
        image_path = image_dir / name
        if not image_path.is_file():
            raise ValueError(f"Source image is missing: {image_path}")
        expected_hash = row["sha256"].strip().lower()
        actual_hash = sha256_file(image_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Source image hash mismatch: {name}")
        by_name[name] = row
    return by_name


def _validate_label_schema(metadata: ET.Element) -> None:
    label_nodes = metadata.findall("./labels/label")
    labels = {_required_text(label, "name") for label in label_nodes}
    if labels != SOURCE_LABELS:
        raise ValueError(
            "CVAT labels do not match the expected source taxonomy: "
            f"expected {sorted(SOURCE_LABELS)}, found {sorted(labels)}"
        )
    for label in label_nodes:
        name = _required_text(label, "name")
        attribute_nodes = label.findall("./attributes/attribute")
        damage_attributes = [
            attribute
            for attribute in attribute_nodes
            if _required_text(attribute, "name") == "damage_state"
        ]
        if len(damage_attributes) != 1 or len(attribute_nodes) != 1:
            raise ValueError(f"{name} must define exactly one damage_state attribute.")
        values = {
            value.strip()
            for value in _required_text(damage_attributes[0], "values").splitlines()
            if value.strip()
        }
        if values != set(DAMAGE_LEVELS):
            raise ValueError(
                f"{name} damage_state values must be {sorted(DAMAGE_LEVELS)}, "
                f"found {sorted(values)}"
            )


def _annotation_id(
    image_hash: str,
    component_class: str,
    coordinates: Iterable[float],
    prefix: str = "HCVAT",
) -> str:
    coordinate_text = ",".join(f"{value:.2f}" for value in coordinates)
    identity = f"{image_hash}|{component_class}|{coordinate_text}".encode("utf-8")
    return f"{prefix}_{hashlib.sha256(identity).hexdigest()[:16].upper()}"


def extract_cvat_damage_rows(
    tree: ET.ElementTree,
    source_manifest: dict[str, dict[str, str]],
    annotation_id_prefix: str = "HCVAT",
) -> tuple[list[dict[str, str]], dict[str, object]]:
    root = tree.getroot()
    version = _required_text(root, "version")
    if version != "1.1":
        raise ValueError(f"Expected CVAT format version 1.1, found {version}.")
    metadata = _find_metadata(root)
    mode = _required_text(metadata, "mode")
    if mode != "annotation":
        raise ValueError(f"Expected CVAT annotation mode, found {mode}.")
    _validate_label_schema(metadata)

    image_nodes = root.findall("image")
    declared_size = int(_required_text(metadata, "size"))
    if declared_size != len(image_nodes):
        raise ValueError(
            f"CVAT metadata declares {declared_size} images, found {len(image_nodes)}."
        )
    image_names = [(image.get("name") or "").strip() for image in image_nodes]
    if len(image_names) != len(set(image_names)):
        raise ValueError("CVAT XML contains duplicate image names.")
    if set(image_names) != set(source_manifest):
        missing = sorted(set(source_manifest) - set(image_names))
        unknown = sorted(set(image_names) - set(source_manifest))
        raise ValueError(
            "CVAT/source image sets differ: "
            f"missing={missing[:5]}, unknown={unknown[:5]}"
        )

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    empty_images = 0
    for image in sorted(image_nodes, key=lambda node: int(node.get("id", "-1"))):
        image_name = (image.get("name") or "").strip()
        source_row = source_manifest[image_name]
        width = int(image.get("width", "0"))
        height = int(image.get("height", "0"))
        if width != int(source_row["width"]) or height != int(source_row["height"]):
            raise ValueError(f"Image dimensions do not match the source manifest: {image_name}")
        unsupported = [child.tag for child in image if child.tag != "box"]
        if unsupported:
            raise ValueError(
                f"{image_name} contains unsupported annotations: {sorted(set(unsupported))}"
            )
        boxes = image.findall("box")
        if not boxes:
            empty_images += 1
        for box_index, box in enumerate(boxes, start=1):
            source_label = (box.get("label") or "").strip()
            if source_label not in SOURCE_LABELS:
                raise ValueError(f"Unexpected box label {source_label!r} in {image_name}.")
            component_class = LABEL_ALIASES.get(source_label, source_label)
            if component_class not in PART_GROUPS:
                raise ValueError(f"No CRATER part-group mapping for {source_label}.")
            attribute_nodes = box.findall("attribute")
            damage_nodes = [
                attribute
                for attribute in attribute_nodes
                if (attribute.get("name") or "").strip() == "damage_state"
            ]
            if len(damage_nodes) != 1 or len(attribute_nodes) != 1:
                raise ValueError(
                    f"{image_name} box {box_index} must have exactly one damage_state attribute."
                )
            damage_state = (damage_nodes[0].text or "").strip()
            if damage_state not in DAMAGE_LEVELS:
                raise ValueError(
                    f"{image_name} box {box_index} has invalid damage_state {damage_state!r}."
                )
            try:
                x1 = float(box.get("xtl", "nan"))
                y1 = float(box.get("ytl", "nan"))
                x2 = float(box.get("xbr", "nan"))
                y2 = float(box.get("ybr", "nan"))
            except ValueError as error:
                raise ValueError(f"Non-numeric box coordinates in {image_name}.") from error
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                raise ValueError(f"Non-finite box coordinates in {image_name}.")
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                raise ValueError(
                    f"Out-of-bounds box in {image_name}: {(x1, y1, x2, y2)}"
                )
            occluded = box.get("occluded", "0")
            if occluded not in {"0", "1"}:
                raise ValueError(
                    f"{image_name} box {box_index} has invalid occluded value {occluded!r}."
                )
            annotation_source = (box.get("source") or "").strip()
            if not annotation_source:
                raise ValueError(f"{image_name} box {box_index} has no annotation source.")
            image_hash = source_row["sha256"].strip().lower()
            annotation_id = _annotation_id(
                image_hash,
                component_class,
                (x1, y1, x2, y2),
                prefix=annotation_id_prefix,
            )
            if annotation_id in seen_ids:
                raise ValueError(f"Duplicate annotation identity: {annotation_id}")
            seen_ids.add(annotation_id)
            box_width = x2 - x1
            box_height = y2 - y1
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "cvat_image_id": image.get("id", ""),
                    "cvat_box_index": str(box_index),
                    "source_image": image_name,
                    "source_image_sha256": image_hash,
                    "component_class_source": source_label,
                    "component_class": component_class,
                    "component_group": PART_GROUPS[component_class],
                    "damage_state_source": damage_state,
                    "damage_level": DAMAGE_LEVELS[damage_state],
                    "box_x1": f"{x1:.2f}",
                    "box_y1": f"{y1:.2f}",
                    "box_x2": f"{x2:.2f}",
                    "box_y2": f"{y2:.2f}",
                    "box_width": f"{box_width:.2f}",
                    "box_height": f"{box_height:.2f}",
                    "box_area": f"{box_width * box_height:.2f}",
                    "occluded": occluded,
                    "annotation_source": annotation_source,
                    "split": "",
                }
            )
    return rows, {
        "cvat_version": version,
        "cvat_metadata_kind": metadata.tag,
        "cvat_job_or_task_id": _required_text(metadata, "id"),
        "cvat_mode": mode,
        "cvat_updated": _required_text(metadata, "updated"),
        "cvat_dumped": _required_text(root, "./meta/dumped"),
        "image_count": len(image_nodes),
        "box_count": len(rows),
        "empty_image_count": empty_images,
    }


def sanitize_cvat_tree(tree: ET.ElementTree) -> ET.ElementTree:
    root_copy = ET.fromstring(ET.tostring(tree.getroot(), encoding="utf-8"))
    metadata = _find_metadata(root_copy)
    for tag in ("owner", "assignee"):
        node = metadata.find(tag)
        if node is not None:
            metadata.remove(node)
    for segment in metadata.findall("./segments/segment"):
        url = segment.find("url")
        if url is not None:
            segment.remove(url)
    ET.indent(root_copy, space="  ")
    return ET.ElementTree(root_copy)


def write_cvat_xml(tree: ET.ElementTree, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
    os.replace(temporary, path)


def write_damage_box_csv(rows: list[dict[str, str]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _component_box_comparison(path: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    images = {int(item["id"]): str(item["file_name"]) for item in payload["images"]}

    def key(image: str, label: str, coordinates: Iterable[float]) -> str:
        normalized = LABEL_ALIASES.get(label, label)
        return "|".join((image, normalized, *(f"{value:.2f}" for value in coordinates)))

    old_keys = []
    for annotation in payload["annotations"]:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        old_keys.append(
            key(
                images[int(annotation["image_id"])],
                categories[int(annotation["category_id"])],
                (x, y, x + width, y + height),
            )
        )
    new_keys = [
        key(
            row["source_image"],
            row["component_class"],
            (
                float(row["box_x1"]),
                float(row["box_y1"]),
                float(row["box_x2"]),
                float(row["box_y2"]),
            ),
        )
        for row in rows
    ]
    if len(old_keys) != len(set(old_keys)) or len(new_keys) != len(set(new_keys)):
        raise ValueError("Duplicate component boxes prevent deterministic comparison.")
    old_set = set(old_keys)
    new_set = set(new_keys)
    return {
        "previous_box_count": len(old_keys),
        "current_box_count": len(new_keys),
        "exact_unchanged_box_count": len(old_set & new_set),
        "exact_added_or_adjusted_box_count": len(new_set - old_set),
        "exact_removed_or_adjusted_box_count": len(old_set - new_set),
    }


def import_cvat_damage_annotations(
    annotations_path: Path,
    source_image_dir: Path,
    source_manifest_path: Path,
    component_annotations_path: Path,
    output_dir: Path,
    dataset_version: str = "humvee-cvat-damage-v1",
) -> dict[str, object]:
    annotations_path = annotations_path.resolve()
    source_image_dir = source_image_dir.resolve()
    source_manifest_path = source_manifest_path.resolve()
    component_annotations_path = component_annotations_path.resolve()
    output_dir = output_dir.resolve()
    for required in (annotations_path, source_manifest_path, component_annotations_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_image_dir.is_dir():
        raise FileNotFoundError(source_image_dir)

    tree, raw_xml = parse_cvat_xml(annotations_path)
    source_manifest = _load_source_manifest(source_manifest_path, source_image_dir)
    rows, summary = extract_cvat_damage_rows(tree, source_manifest)
    comparison = _component_box_comparison(component_annotations_path, rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_xml_path = output_dir / "annotations.xml"
    csv_path = output_dir / "damage_box_annotations.csv"
    manifest_path = output_dir / "import_manifest.json"
    write_cvat_xml(sanitize_cvat_tree(tree), sanitized_xml_path)
    write_damage_box_csv(rows, csv_path)

    class_counts = Counter(row["component_class"] for row in rows)
    source_damage_counts = Counter(row["damage_state_source"] for row in rows)
    canonical_damage_counts = Counter(row["damage_level"] for row in rows)
    class_damage_counts: dict[str, Counter[str]] = {}
    for row in rows:
        class_damage_counts.setdefault(row["component_class"], Counter()).update(
            [row["damage_level"]]
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "status": "validated_source_inventory_not_training_ready",
        "original_export_filename": annotations_path.name,
        "original_export_container": annotations_path.parent.name,
        "original_export_sha256": hashlib.sha256(raw_xml).hexdigest(),
        "sanitized_annotations_sha256": sha256_file(sanitized_xml_path),
        "damage_box_annotations_sha256": sha256_file(csv_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_component_annotations_sha256": sha256_file(component_annotations_path),
        **summary,
        "class_counts": dict(sorted(class_counts.items())),
        "class_damage_counts": {
            component_class: dict(sorted(counts.items()))
            for component_class, counts in sorted(class_damage_counts.items())
        },
        "source_damage_state_counts": dict(sorted(source_damage_counts.items())),
        "canonical_damage_level_counts": dict(sorted(canonical_damage_counts.items())),
        "class_mapping": dict(sorted(LABEL_ALIASES.items())),
        "damage_mapping": dict(sorted(DAMAGE_LEVELS.items())),
        "component_box_comparison": comparison,
        "split_status": "unassigned_pending_authoritative_source_grouping",
        "semantic_review_note": (
            "CVAT defines intact as the default attribute value; the export cannot "
            "distinguish explicitly confirmed intact boxes from untouched defaults."
        ),
        "privacy_sanitization": [
            "removed CVAT owner metadata",
            "removed CVAT assignee metadata",
            "removed CVAT segment URL",
        ],
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="CVAT for images 1.1 XML export")
    parser.add_argument("--source-image-dir", type=Path, default=DEFAULT_SOURCE_IMAGE_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument(
        "--component-annotations",
        type=Path,
        default=DEFAULT_COMPONENT_ANNOTATIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-version", default="humvee-cvat-damage-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = import_cvat_damage_annotations(
        annotations_path=args.annotations,
        source_image_dir=args.source_image_dir,
        source_manifest_path=args.source_manifest,
        component_annotations_path=args.component_annotations,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
    )
    print(f"Validated {manifest['image_count']} images and {manifest['box_count']} boxes.")
    print(f"Imported source inventory: {args.output_dir.resolve()}")
    print("Training split remains unassigned pending authoritative source grouping.")


if __name__ == "__main__":
    main()
