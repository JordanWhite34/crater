import csv
import hashlib
import json
import shutil
import unittest
from pathlib import Path

from PIL import Image

from tools.data.import_cvat_damage_annotations import import_cvat_damage_annotations


SOURCE_LABELS = (
    "wheel_tire",
    "windshield",
    "engine_bay",
    "weapons_station",
    "comms_equipment",
    "door",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(root: Path, unsafe_prefix: str = "") -> tuple[Path, Path, Path, Path]:
    image_dir = root / "images"
    image_dir.mkdir()
    first = image_dir / "first.jpg"
    second = image_dir / "second.jpg"
    Image.new("RGB", (100, 80), "navy").save(first)
    Image.new("RGB", (120, 90), "gray").save(second)

    manifest = root / "source_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("file_name", "sha256", "width", "height"),
        )
        writer.writeheader()
        writer.writerow(
            {"file_name": first.name, "sha256": sha256(first), "width": 100, "height": 80}
        )
        writer.writerow(
            {"file_name": second.name, "sha256": sha256(second), "width": 120, "height": 90}
        )

    component_annotations = root / "instances_default.json"
    component_annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": first.name},
                    {"id": 2, "file_name": second.name},
                ],
                "categories": [
                    {"id": 1, "name": "wheel_tire"},
                    {"id": 2, "name": "weapon_station"},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [10, 12, 30, 35]},
                ],
            }
        ),
        encoding="utf-8",
    )

    label_xml = "".join(
        f"""
        <label><name>{label}</name><type>any</type><attributes><attribute>
          <name>damage_state</name><values>intact\ndegraded\ndestroyed\nunknown</values>
        </attribute></attributes></label>
        """
        for label in SOURCE_LABELS
    )
    annotations = root / "annotations.xml"
    annotations.write_text(
        unsafe_prefix
        + f"""<?xml version="1.0" encoding="utf-8"?>
<annotations><version>1.1</version><meta><job><id>42</id><size>2</size><updated>2026-09-01T00:00:00Z</updated>
<mode>annotation</mode><segments><segment><url>https://example.invalid/private</url></segment></segments>
<owner><username>owner</username><email>private@example.com</email></owner>
<assignee><username>worker</username><email>worker@example.com</email></assignee>
<labels>{label_xml}</labels></job><dumped>2026-09-01T00:00:00Z</dumped></meta>
<image id="0" name="first.jpg" width="100" height="80">
  <box label="wheel_tire" source="manual" occluded="0" xtl="10" ytl="12" xbr="40" ybr="47">
    <attribute name="damage_state">degraded</attribute>
  </box>
</image>
<image id="1" name="second.jpg" width="120" height="90">
  <box label="weapons_station" source="manual" occluded="1" xtl="20" ytl="15" xbr="80" ybr="70">
    <attribute name="damage_state">destroyed</attribute>
  </box>
</image></annotations>
""",
        encoding="utf-8",
    )
    return annotations, image_dir, manifest, component_annotations


class ImportCvatDamageAnnotationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_import_cvat_damage_annotations")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_import_validates_maps_and_sanitizes(self) -> None:
        fixture_root = self.root / "valid"
        fixture_root.mkdir()
        annotations, image_dir, source_manifest, component_annotations = write_fixture(
            fixture_root
        )
        output_dir = fixture_root / "output"

        manifest = import_cvat_damage_annotations(
            annotations,
            image_dir,
            source_manifest,
            component_annotations,
            output_dir,
        )

        self.assertEqual(manifest["image_count"], 2)
        self.assertEqual(manifest["box_count"], 2)
        self.assertEqual(
            manifest["canonical_damage_level_counts"],
            {"possible_damage": 1, "severe_visible_damage": 1},
        )
        with (output_dir / "damage_box_annotations.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(
            [row["component_class"] for row in rows],
            ["wheel_tire", "weapon_station"],
        )
        self.assertEqual(
            [row["component_group"] for row in rows],
            ["mobility", "mission_equipment"],
        )
        self.assertEqual(
            [row["damage_level"] for row in rows],
            ["possible_damage", "severe_visible_damage"],
        )
        self.assertTrue(all(not row["split"] for row in rows))

        sanitized = (output_dir / "annotations.xml").read_text(encoding="utf-8")
        self.assertNotIn("private@example.com", sanitized)
        self.assertNotIn("worker@example.com", sanitized)
        self.assertNotIn("example.invalid/private", sanitized)
        self.assertNotIn("<owner>", sanitized)
        self.assertNotIn("<assignee>", sanitized)
        self.assertEqual(sanitized.count("<box "), 2)
        self.assertEqual(sanitized.count("damage_state"), 8)

    def test_import_rejects_dtd(self) -> None:
        fixture_root = self.root / "unsafe"
        fixture_root.mkdir()
        annotations, image_dir, source_manifest, component_annotations = write_fixture(
            fixture_root,
            unsafe_prefix='<!DOCTYPE annotations [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>\n',
        )
        with self.assertRaisesRegex(ValueError, "DTD or entity"):
            import_cvat_damage_annotations(
                annotations,
                image_dir,
                source_manifest,
                component_annotations,
                fixture_root / "output",
            )


if __name__ == "__main__":
    unittest.main()
