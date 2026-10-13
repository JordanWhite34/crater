"""Create reviewed dominant-component annotations for the synthetic unlabeled set.

The coordinates below were reviewed against the generated images. Each image was
sampled for one dominant component, so this inventory records one conservative
box per image rather than claiming that every visible vehicle component has been
exhaustively annotated.
"""

from __future__ import annotations

import csv
import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from tools.data.import_cvat_damage_annotations import (
    DAMAGE_LEVELS,
    LABEL_ALIASES,
    PART_GROUPS,
    extract_cvat_damage_rows,
    write_damage_box_csv,
)


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "datasets" / "military" / "humvee_unlabeled_v1" / "images"
OUT_DIR = ROOT / "datasets" / "military" / "damage" / "source" / "synthetic_unlabeled_review_v1"


# (file_name, canonical class, source damage state, normalized x1, y1, x2, y2)
REVIEW = [
    ("01_comms_possible.png", "comms_equipment", "degraded", .69, .44, .80, .72),
    ("02_weapon_station_severe.png", "weapon_station", "destroyed", .36, .02, .78, .39),
    ("03_windshield_severe.png", "windshield", "destroyed", .26, .17, .74, .49),
    ("04_door_possible.png", "door", "degraded", .06, .32, .42, .84),
    ("05_engine_bay_unobservable.png", "engine_bay", "unknown", .42, .34, .77, .72),
    ("06_wheel_tire_severe.png", "wheel_tire", "destroyed", .34, .52, .59, .95),
    ("07_comms_intact.png", "comms_equipment", "intact", .62, .34, .86, .79),
    ("08_comms_severe.png", "comms_equipment", "destroyed", .29, .26, .73, .66),
    ("09_weapon_possible.png", "weapon_station", "degraded", .24, .05, .72, .34),
    ("10_weapon_intact.png", "weapon_station", "intact", .22, .02, .70, .30),
    ("11_windshield_possible.png", "windshield", "degraded", .27, .25, .77, .55),
    ("12_door_severe.png", "door", "destroyed", .38, .28, .78, .82),
    ("13_wheel_possible.png", "wheel_tire", "degraded", .27, .40, .70, .97),
    ("14_engine_severe.png", "engine_bay", "destroyed", .20, .18, .88, .78),
    ("15_desert_comms_severe.png", "comms_equipment", "destroyed", .48, .10, .63, .38),
    ("16_forest_weapon_possible.png", "weapon_station", "degraded", .42, .18, .66, .43),
    ("17_snow_windshield_severe.png", "windshield", "destroyed", .42, .42, .65, .62),
    ("18_urban_door_possible.png", "door", "degraded", .52, .43, .68, .67),
    ("19_farm_wheel_severe.png", "wheel_tire", "destroyed", .34, .60, .50, .87),
    ("20_river_engine_unobservable.png", "engine_bay", "unknown", .48, .35, .73, .63),
    ("21_coastal_comms_possible.png", "comms_equipment", "degraded", .52, .22, .70, .45),
    ("22_grass_weapon_severe.png", "weapon_station", "destroyed", .45, .17, .68, .42),
    ("23_mountain_wheel_possible.png", "wheel_tire", "degraded", .53, .57, .66, .77),
    ("24_urban_wreck_mixed.png", "comms_equipment", "destroyed", .46, .22, .68, .46),
    ("25_comms_severe_muddy.png", "comms_equipment", "destroyed", .37, .14, .62, .42),
    ("26_comms_possible_highway.png", "comms_equipment", "degraded", .40, .12, .63, .40),
    ("27_comms_severe_snow.png", "comms_equipment", "destroyed", .39, .13, .64, .43),
    ("28_comms_intact_training.png", "comms_equipment", "intact", .39, .16, .63, .43),
    ("29_weapon_severe_rubble.png", "weapon_station", "destroyed", .42, .16, .66, .46),
    ("30_weapon_severe_field.png", "weapon_station", "destroyed", .45, .16, .67, .44),
    ("31_weapon_possible_sand.png", "weapon_station", "degraded", .42, .14, .65, .43),
    ("32_windshield_severe_rain.png", "windshield", "destroyed", .38, .36, .67, .58),
    ("33_door_possible_grass.png", "door", "degraded", .51, .42, .68, .66),
    ("34_wheel_severe_mud.png", "wheel_tire", "destroyed", .58, .54, .72, .75),
    ("35_wheel_possible_rock.png", "wheel_tire", "degraded", .52, .60, .63, .79),
    ("36_engine_unobservable_smoke.png", "engine_bay", "unknown", .50, .31, .70, .57),
]


COLORS = {
    "wheel_tire": "#ca9e1c",
    "windshield": "#01a1a7",
    "engine_bay": "#5d424c",
    "weapons_station": "#acfdac",
    "door": "#d66b2e",
    "comms_equipment": "#7d5cff",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_label(canonical: str) -> str:
    return "weapons_station" if canonical == "weapon_station" else canonical


def build_source_manifest() -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for file_name, *_ in REVIEW:
        path = IMAGE_DIR / file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        manifest[file_name] = {
            "file_name": file_name,
            "sha256": sha256_file(path),
            "size_bytes": str(path.stat().st_size),
            "width": str(width),
            "height": str(height),
            "source_batch": "synthetic_unlabeled_v1",
            "domain": "synthetic",
            "annotation_status": "reviewed_dominant_component",
        }
    return manifest


def build_xml(manifest: dict[str, dict[str, str]]) -> ET.ElementTree:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    fields = {
        "id": "synthetic-unlabeled-review-v1",
        "name": "synthetic unlabeled Humvee review v1",
        "size": str(len(REVIEW)),
        "mode": "annotation",
        "overlap": "0",
        "bugtracker": "",
        "created": "2026-09-10 00:00:00+00:00",
        "updated": "2026-09-10 00:00:00+00:00",
        "subset": "default",
        "start_frame": "0",
        "stop_frame": str(len(REVIEW) - 1),
        "frame_filter": "",
    }
    for key, value in fields.items():
        ET.SubElement(task, key).text = value
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    ET.SubElement(segment, "id").text = "synthetic-unlabeled-review-v1"
    ET.SubElement(segment, "start").text = "0"
    ET.SubElement(segment, "stop").text = str(len(REVIEW) - 1)
    labels = ET.SubElement(task, "labels")
    for label in ("wheel_tire", "windshield", "engine_bay", "weapons_station", "door", "comms_equipment"):
        node = ET.SubElement(labels, "label")
        ET.SubElement(node, "name").text = label
        ET.SubElement(node, "color").text = COLORS[label]
        ET.SubElement(node, "type").text = "any"
        attributes = ET.SubElement(node, "attributes")
        attribute = ET.SubElement(attributes, "attribute")
        ET.SubElement(attribute, "name").text = "damage_state"
        ET.SubElement(attribute, "mutable").text = "False"
        ET.SubElement(attribute, "input_type").text = "radio"
        ET.SubElement(attribute, "default_value").text = "intact"
        ET.SubElement(attribute, "values").text = "intact\ndegraded\ndestroyed\nunknown"
    ET.SubElement(root.find("meta"), "dumped").text = "2026-09-10 00:00:00+00:00"

    for image_id, (file_name, canonical, damage_state, nx1, ny1, nx2, ny2) in enumerate(REVIEW):
        row = manifest[file_name]
        width, height = int(row["width"]), int(row["height"])
        image = ET.SubElement(root, "image", id=str(image_id), name=file_name, width=str(width), height=str(height))
        box = ET.SubElement(
            image,
            "box",
            label=source_label(canonical),
            source="manual_visual_review",
            xtl=f"{nx1 * width:.2f}",
            ytl=f"{ny1 * height:.2f}",
            xbr=f"{nx2 * width:.2f}",
            ybr=f"{ny2 * height:.2f}",
            occluded="0",
        )
        ET.SubElement(box, "attribute", name="damage_state").text = damage_state
    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_source_manifest(rows: list[dict[str, str]], path: Path) -> None:
    fields = ("file_name", "sha256", "size_bytes", "width", "height", "source_batch", "domain", "annotation_status")
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_review_csv(rows: list[dict[str, str]], path: Path) -> None:
    fields = ("file_name", "component_class", "component_group", "damage_level", "box_x1", "box_y1", "box_x2", "box_y2", "annotation_source", "qa_status")
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_source_manifest()
    manifest_rows = [manifest[file_name] for file_name, *_ in REVIEW]
    write_source_manifest(manifest_rows, OUT_DIR / "source_manifest.csv")
    xml_path = OUT_DIR / "annotations.xml"
    build_xml(manifest).write(xml_path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
    from tools.data.import_cvat_damage_annotations import parse_cvat_xml

    tree, _ = parse_cvat_xml(xml_path)
    rows, summary = extract_cvat_damage_rows(tree, manifest, annotation_id_prefix="SREVIEW")
    write_damage_box_csv(rows, OUT_DIR / "damage_box_annotations.csv")
    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "file_name": row["source_image"],
                "component_class": row["component_class"],
                "component_group": row["component_group"],
                "damage_level": row["damage_level"],
                "box_x1": row["box_x1"],
                "box_y1": row["box_y1"],
                "box_x2": row["box_x2"],
                "box_y2": row["box_y2"],
                "annotation_source": row["annotation_source"],
                "qa_status": "reviewed_dominant_component",
            }
        )
    write_review_csv(review_rows, OUT_DIR / "label_review.csv")
    print(f"Validated {summary['image_count']} images and {summary['box_count']} reviewed boxes.")
    print(f"Wrote reviewed annotations to {OUT_DIR}")


if __name__ == "__main__":
    main()
