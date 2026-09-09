"""Validate and import the versioned synthetic Humvee damage source.

The importer copies the generated PNGs byte-for-byte, stores a privacy-sanitized
CVAT XML record, and emits the same canonical damage-box CSV contract used by
the real Humvee source. Generation prompts are provenance only; human CVAT
attributes remain the sole source of damage labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

if __package__:
    from tools.data.import_cvat_damage_annotations import (
        DAMAGE_LEVELS,
        LABEL_ALIASES,
        extract_cvat_damage_rows,
        parse_cvat_xml,
        sanitize_cvat_tree,
        sha256_file,
        write_cvat_xml,
        write_damage_box_csv,
    )
else:
    from import_cvat_damage_annotations import (
        DAMAGE_LEVELS,
        LABEL_ALIASES,
        extract_cvat_damage_rows,
        parse_cvat_xml,
        sanitize_cvat_tree,
        sha256_file,
        write_cvat_xml,
        write_damage_box_csv,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "datasets"
    / "military"
    / "damage"
    / "source"
    / "synthetic_humvee_damage_v1"
)
GENERATION_SESSION_ID = "01a04411-3958-7240-9f6b-677f4af4af60"
GENERATOR = "OpenAI image generation tool"
GENERATOR_MODEL = "not recorded"


@dataclass(frozen=True)
class GenerationSpec:
    prompt_id: str
    image_name: str
    vehicle_family: str
    intended_condition: str

    @property
    def generation_group_id(self) -> str:
        return f"{GENERATION_SESSION_ID}:{self.prompt_id}"


GENERATION_SPECS = (
    GenerationSpec("01_m1114_intact", "exec-0df7a36d-2319-49f4-9d86-ebd130b25d3f.png", "M1114", "all tracked components intact"),
    GenerationSpec("02_m1151_intact", "exec-648b95f6-2e68-4b27-a8df-18094e847e9d.png", "M1151", "all tracked components intact"),
    GenerationSpec("03_m1114_flat_tires", "exec-8a201254-3ae8-48ab-a77c-bdcae250f6b4.png", "M1114", "two severely deflated tires"),
    GenerationSpec("04_m1151_windshield_cracked", "exec-9c293926-2fc4-4432-9b4c-eb4ca9fe3f4c.png", "M1151", "extensively cracked windshield"),
    GenerationSpec("05_m1114_door_damage", "exec-e561515d-18c0-4b32-b0c5-a2ff442f4e9b.png", "M1114", "dented and perforated armored door"),
    GenerationSpec("06_m1151_engine_damage", "exec-ecd5ba16-f4b5-4258-8ded-eafa009e14d2.png", "M1151", "buckled hood and soot-blackened engine bay"),
    GenerationSpec("07_m1114_windshield_shattered", "exec-9570f9f9-aa99-43b1-a0f2-daf38916087f.png", "M1114", "shattered windshield"),
    GenerationSpec("08_m1151_destroyed_tires", "exec-a064c47e-b677-44c6-a092-8ff20ae00455.png", "M1151", "two shredded front tires"),
    GenerationSpec("09_m1114_hood_burn", "exec-7a769a2d-162b-4b3e-8684-c6a4e4781d3d.png", "M1114", "scorched hood and engine bay"),
    GenerationSpec("10_m1151_door_missing", "exec-449dfdae-8d70-42eb-a4a2-8538caf81c49.png", "M1151", "right rear armored door torn off"),
    GenerationSpec("11_m1114_weapon_station_damaged", "exec-7e536cca-c9ec-481d-96e9-b46ffe127aa3.png", "M1114", "bent gun ring and damaged weapon"),
    GenerationSpec("12_m1151_comms_damaged", "exec-fd6916d4-b56f-4f06-bd76-b539b605967f.png", "M1151", "broken antennas and damaged radio equipment"),
    GenerationSpec("13_m1114_two_flat_door_damage", "exec-54237c45-196d-4f8d-8c01-c9b955a3352c.png", "M1114", "two deflated tires and dented armored door"),
    GenerationSpec("14_m1151_cracked_windshield_broken_comms", "exec-516d3535-eb1a-46d6-904b-388f8b60a3a9.png", "M1151", "cracked windshield and snapped antennas"),
    GenerationSpec("15_m1114_burned_engine_flat_tire", "exec-597297fc-4c95-44be-a5f6-45cfadb18626.png", "M1114", "destroyed front tire and fire-damaged engine bay"),
    GenerationSpec("16_m1151_windshield_out_door_damage", "exec-2196d25a-90b2-4d9b-9a5c-f9e5770f49e8.png", "M1151", "missing windshield and damaged front door"),
    GenerationSpec("17_m1114_weapon_comms_damage", "exec-40053e27-68a5-4d2b-9e08-f4019573d5b4.png", "M1114", "damaged weapon station and communications equipment"),
    GenerationSpec("18_m1151_three_flat_tires", "exec-4d666161-0a25-4376-bcb5-6d09dd53b6f8.png", "M1151", "three destroyed tires"),
    GenerationSpec("19_m1114_destroyed_engine", "exec-16aa96fa-5269-4ea5-9ecf-e6516adbe73f.png", "M1114", "burned-out engine bay"),
    GenerationSpec("20_m1151_destroyed_cab_glass", "exec-11bc9a67-edc6-499c-9632-b137fd4bb405.png", "M1151", "windshield and front side windows blown out"),
    GenerationSpec("21_m1114_destroyed_door", "exec-0ce2c907-2f6e-4e19-9935-9ec1e14ef7ff.png", "M1114", "front armored door twisted open"),
    GenerationSpec("22_m1151_destroyed_weapon_station", "exec-9aa52de3-07f4-4778-b38b-36c23aa321e8.png", "M1151", "destroyed weapon station"),
    GenerationSpec("23_m1114_destroyed_comms", "exec-51224d86-d526-4c70-a9de-6c70c9067ff4.png", "M1114", "destroyed communications equipment"),
    GenerationSpec("24_m1151_wreck_tires_engine", "exec-ffbcb6d4-ff4c-428e-8830-980521f67d09.png", "M1151", "shredded tires, cracked windshield, burned engine, damaged comms"),
    GenerationSpec("25_m1114_intact_light_scorch", "exec-67a994cf-f5ea-47c6-b7c3-ac74c0679002.png", "M1114", "tracked components intact with light exterior scorch"),
    GenerationSpec("26_m1151_intact_no_weapon", "exec-dee60ab3-ff34-47d7-ad62-d5c093d5a990.png", "M1151", "tracked components intact with empty weapon station"),
    GenerationSpec("27_m1114_windshield_tire_door", "exec-0a0b7d16-0de8-43d1-bd03-991938d29339.png", "M1114", "destroyed tire, cracked windshield, and dented door"),
    GenerationSpec("28_m1151_engine_comms_weapon", "exec-ad1f1220-9c51-4b4e-af6a-74d2d4546116.png", "M1151", "damaged engine bay, weapon station, and communications equipment"),
    GenerationSpec("29_m1114_destroyed_cab_and_comms", "exec-5832f4af-dd13-41aa-a551-4d4dae5bd329.png", "M1114", "missing windshield, damaged doors, and destroyed communications equipment"),
    GenerationSpec("30_m1151_heavily_damaged_wreck", "exec-a1e315b5-18ca-402a-9944-16723ad6a51d.png", "M1151", "heavily damaged multi-component wreck"),
)

SOURCE_MANIFEST_FIELDS = (
    "file_name",
    "sha256",
    "size_bytes",
    "width",
    "height",
    "annotation_count",
    "domain",
    "source_type",
    "generator",
    "generator_model",
    "generation_session_id",
    "prompt_id",
    "vehicle_family",
    "intended_condition",
    "generation_group_id",
    "parent_source_image",
    "review_status",
    "split",
)


def _inspect_source_images(
    source_image_dir: Path,
    generation_specs: Iterable[GenerationSpec],
) -> tuple[dict[str, dict[str, str]], list[GenerationSpec]]:
    specs = list(generation_specs)
    names = [spec.image_name for spec in specs]
    if not specs or len(names) != len(set(names)):
        raise ValueError("Generation specifications must contain unique image names.")
    prompt_ids = [spec.prompt_id for spec in specs]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Generation specifications must contain unique prompt IDs.")

    source_files = {path.name: path for path in source_image_dir.glob("*.png")}
    if set(source_files) != set(names):
        missing = sorted(set(names) - set(source_files))
        unknown = sorted(set(source_files) - set(names))
        raise ValueError(
            "Synthetic image/specification sets differ: "
            f"missing={missing[:5]}, unknown={unknown[:5]}"
        )

    manifest: dict[str, dict[str, str]] = {}
    seen_hashes: set[str] = set()
    for spec in specs:
        path = source_files[spec.image_name]
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
        except Exception as error:
            raise ValueError(f"Invalid synthetic image: {path}") from error
        if image_format != "PNG":
            raise ValueError(f"Expected PNG source image: {path}")
        image_hash = sha256_file(path)
        if image_hash in seen_hashes:
            raise ValueError(f"Duplicate synthetic image content: {path.name}")
        seen_hashes.add(image_hash)
        manifest[path.name] = {
            "file_name": path.name,
            "sha256": image_hash,
            "size_bytes": str(path.stat().st_size),
            "width": str(width),
            "height": str(height),
            "annotation_count": "0",
            "domain": "synthetic",
            "source_type": "text_to_image",
            "generator": GENERATOR,
            "generator_model": GENERATOR_MODEL,
            "generation_session_id": GENERATION_SESSION_ID,
            "prompt_id": spec.prompt_id,
            "vehicle_family": spec.vehicle_family,
            "intended_condition": spec.intended_condition,
            "generation_group_id": spec.generation_group_id,
            "parent_source_image": "",
            "review_status": "cvat_annotations_received_pending_synthetic_qa",
            "split": "",
        }
    return manifest, specs


def _write_source_manifest(
    rows: list[dict[str, str]],
    path: Path,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=SOURCE_MANIFEST_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _copy_images(
    source_image_dir: Path,
    output_image_dir: Path,
    manifest: dict[str, dict[str, str]],
) -> None:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        path.name
        for path in output_image_dir.glob("*.png")
        if path.name not in manifest
    )
    if unexpected:
        raise ValueError(
            "Output image directory contains files outside this dataset version: "
            f"{unexpected[:5]}"
        )
    for name, row in manifest.items():
        destination = output_image_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source_image_dir / name, temporary)
        if sha256_file(temporary) != row["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Copied image hash mismatch: {name}")
        os.replace(temporary, destination)


def import_synthetic_cvat_damage_annotations(
    annotations_path: Path,
    source_image_dir: Path,
    output_dir: Path,
    generation_specs: Iterable[GenerationSpec] = GENERATION_SPECS,
    dataset_version: str = "synthetic-humvee-damage-v1",
) -> dict[str, object]:
    annotations_path = annotations_path.resolve()
    source_image_dir = source_image_dir.resolve()
    output_dir = output_dir.resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)
    if not source_image_dir.is_dir():
        raise FileNotFoundError(source_image_dir)

    source_manifest, specs = _inspect_source_images(source_image_dir, generation_specs)
    tree, _ = parse_cvat_xml(annotations_path)
    rows, summary = extract_cvat_damage_rows(
        tree,
        source_manifest,
        annotation_id_prefix="SCVAT",
    )

    annotation_counts = Counter(row["source_image"] for row in rows)
    for name, row in source_manifest.items():
        row["annotation_count"] = str(annotation_counts[name])

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    sanitized_xml_path = output_dir / "annotations.xml"
    csv_path = output_dir / "damage_box_annotations.csv"
    source_manifest_path = output_dir / "source_manifest.csv"
    import_manifest_path = output_dir / "import_manifest.json"
    _copy_images(source_image_dir, image_dir, source_manifest)
    write_cvat_xml(sanitize_cvat_tree(tree), sanitized_xml_path)
    write_damage_box_csv(rows, csv_path)
    manifest_rows = [source_manifest[spec.image_name] for spec in specs]
    _write_source_manifest(manifest_rows, source_manifest_path)

    class_counts = Counter(row["component_class"] for row in rows)
    source_damage_counts = Counter(row["damage_state_source"] for row in rows)
    canonical_damage_counts = Counter(row["damage_level"] for row in rows)
    class_damage_counts: dict[str, Counter[str]] = {}
    for row in rows:
        class_damage_counts.setdefault(row["component_class"], Counter()).update(
            [row["damage_level"]]
        )
    dimensions = Counter(
        f"{row['width']}x{row['height']}" for row in source_manifest.values()
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "status": "validated_synthetic_source_inventory_not_training_ready",
        "domain": "synthetic",
        "source_type": "text_to_image",
        "generator": GENERATOR,
        "generator_model": GENERATOR_MODEL,
        "generation_session_id": GENERATION_SESSION_ID,
        "original_export_filename": annotations_path.name,
        "original_export_sha256": sha256_file(annotations_path),
        "sanitized_annotations_sha256": sha256_file(sanitized_xml_path),
        "damage_box_annotations_sha256": sha256_file(csv_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_image_count": len(source_manifest),
        "source_image_total_bytes": sum(
            int(row["size_bytes"]) for row in source_manifest.values()
        ),
        "source_image_unique_sha256_count": len(
            {row["sha256"] for row in source_manifest.values()}
        ),
        "image_dimensions": dict(sorted(dimensions.items())),
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
        "split_status": "unassigned_training_only_candidate",
        "evaluation_policy": "exclude synthetic images from validation and test sets",
        "prompt_label_policy": (
            "Intended generation conditions are provenance only; human CVAT "
            "damage attributes are the sole training-label source."
        ),
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
    temporary_manifest = import_manifest_path.with_suffix(
        import_manifest_path.suffix + ".tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, import_manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="CVAT for images 1.1 XML export")
    parser.add_argument("--source-image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-version", default="synthetic-humvee-damage-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = import_synthetic_cvat_damage_annotations(
        annotations_path=args.annotations,
        source_image_dir=args.source_image_dir,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
    )
    print(f"Validated {manifest['image_count']} images and {manifest['box_count']} boxes.")
    print(f"Imported synthetic source inventory: {args.output_dir.resolve()}")
    print("Synthetic images remain training-only candidates pending visual QA.")


if __name__ == "__main__":
    main()
