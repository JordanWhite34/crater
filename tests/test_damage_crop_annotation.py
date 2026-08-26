import csv
import json
import shutil
import unittest
from pathlib import Path

from PIL import Image

from tools.data.damage_crop_annotation import (
    PART_GROUP_ORDER,
    create_annotation_run,
    create_head_splits,
    discover_source_images,
    validate_and_merge_labels,
)


class DamageCropAnnotationTest(unittest.TestCase):
    def setUp(self):
        self.root = Path("tests/.tmp_damage_crop_annotation")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_create_and_validate_annotation_run(self):
        root = self.root
        source_dir = root / "source"
        source_dir.mkdir()
        source_image = source_dir / "vehicle.jpg"
        Image.new("RGB", (160, 120), "navy").save(source_image)

        candidates = []
        for index, group in enumerate(PART_GROUP_ORDER):
            candidates.append(
                {
                    "part_group": group,
                    "detector_class": ("wheel_tire", "windshield", "weapon_station")[index],
                    "detector_confidence": 0.9 - index * 0.1,
                    "source_path": source_image,
                    "source_image": "source/vehicle.jpg",
                    "source_asset_id": "file_sha256:test",
                    "box": (10 + index * 20, 15, 40 + index * 20, 55),
                }
            )
        run_dir = root / "annotation_runs" / "test_run"
        metadata = {"checkpoint_sha256": "abc123", "confidence": 0.25}
        create_annotation_run(
            candidates=candidates,
            run_dir=run_dir,
            repo_root=root,
            inference_metadata=metadata,
        )

        self.assertTrue((run_dir / "crops_manifest.csv").is_file())
        self.assertEqual(len(list((run_dir / "crops").rglob("*.jpg"))), 3)
        self.assertEqual(len(list((run_dir / "contact_sheets").rglob("*.jpg"))), 3)

        uploads = {}
        for severity, group in enumerate(PART_GROUP_ORDER, start=1):
            template = run_dir / "label_templates" / f"{group}.csv"
            upload = run_dir / "label_uploads" / f"{group}.csv"
            with template.open(newline="", encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            with upload.open("w", newline="", encoding="utf-8-sig") as destination:
                writer = csv.DictWriter(destination, fieldnames=("crop_id", "damage_severity"))
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            "crop_id": row["crop_id"],
                            "damage_severity": "" if group == "mission_equipment" else severity,
                        }
                    )
            uploads[group] = upload

        merged = validate_and_merge_labels(run_dir, uploads)
        with merged.open(newline="", encoding="utf-8-sig") as source:
            merged_rows = list(csv.DictReader(source))
        self.assertEqual([row["damage_severity_name"] for row in merged_rows], ["none", "moderate"])
        prepared = create_head_splits(
            run_dir=run_dir,
            validated_labels_csv=merged,
            output_dir=run_dir / "prepared_datasets" / "v1",
        )
        self.assertEqual(len(list(prepared.rglob("C*.jpg"))), 2)
        with (prepared / "mission_equipment" / "labels.csv").open(newline="", encoding="utf-8-sig") as source:
            self.assertEqual(list(csv.DictReader(source)), [])
        with self.assertRaises(FileExistsError):
            create_annotation_run(
                candidates=candidates,
                run_dir=run_dir,
                repo_root=root,
                inference_metadata=metadata,
            )

    def test_catalog_discovery_uses_only_local_rows(self):
        root = self.root
        image_dir = root / "images"
        image_dir.mkdir()
        Image.new("RGB", (20, 20)).save(image_dir / "present.jpg")
        catalog = root / "manifest.csv"
        with catalog.open("w", newline="", encoding="utf-8-sig") as destination:
            writer = csv.DictWriter(destination, fieldnames=("subset_local_file",))
            writer.writeheader()
            writer.writerow({"subset_local_file": "images/present.jpg"})
            writer.writerow({"subset_local_file": ""})
        self.assertEqual(
            discover_source_images(image_dir, catalog, local_file_field="subset_local_file"),
            [(image_dir / "present.jpg").resolve()],
        )

    def test_notebook_code_cells_compile(self):
        notebook_path = Path(__file__).resolve().parents[1] / "experiments/damage/military_damage_crop_labeling.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{notebook_path.name}:cell-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
