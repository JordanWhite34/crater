import csv
import shutil
import unittest
from pathlib import Path

from PIL import Image

from tools.data.import_cvat_damage_annotations import SOURCE_LABELS
from tools.data.import_synthetic_cvat_damage_annotations import (
    GenerationSpec,
    import_synthetic_cvat_damage_annotations,
)


def write_fixture(root: Path) -> tuple[Path, Path, tuple[GenerationSpec, ...]]:
    image_dir = root / "source"
    image_dir.mkdir()
    first = image_dir / "first.png"
    second = image_dir / "second.png"
    Image.new("RGB", (100, 80), "navy").save(first)
    Image.new("RGB", (120, 90), "gray").save(second)

    specs = (
        GenerationSpec("01_first", first.name, "M1114", "damaged tire"),
        GenerationSpec("02_second", second.name, "M1151", "damaged weapon station"),
    )
    label_xml = "".join(
        f"""
        <label><name>{label}</name><type>any</type><attributes><attribute>
          <name>damage_state</name><default_value>intact</default_value>
          <values>intact\ndegraded\ndestroyed\nunknown</values>
        </attribute></attributes></label>
        """
        for label in sorted(SOURCE_LABELS)
    )
    annotations = root / "annotations.xml"
    annotations.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<annotations><version>1.1</version><meta><task><id>42</id><size>2</size>
<updated>2026-09-01T00:00:00Z</updated><mode>annotation</mode>
<segments><segment><url>https://example.invalid/private</url></segment></segments>
<owner><username>owner</username><email>private@example.com</email></owner>
<assignee><username>worker</username><email>worker@example.com</email></assignee>
<labels>{label_xml}</labels></task><dumped>2026-09-01T00:00:00Z</dumped></meta>
<image id="0" name="first.png" width="100" height="80">
  <box label="wheel_tire" source="manual" occluded="0" xtl="10" ytl="12" xbr="40" ybr="47">
    <attribute name="damage_state">degraded</attribute>
  </box>
</image>
<image id="1" name="second.png" width="120" height="90">
  <box label="weapons_station" source="manual" occluded="1" xtl="20" ytl="15" xbr="80" ybr="70">
    <attribute name="damage_state">destroyed</attribute>
  </box>
</image></annotations>
""",
        encoding="utf-8",
    )
    return annotations, image_dir, specs


class ImportSyntheticCvatDamageAnnotationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_import_synthetic_cvat_damage_annotations")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_import_copies_images_maps_labels_and_preserves_lineage(self) -> None:
        annotations, image_dir, specs = write_fixture(self.root)
        output_dir = self.root / "output"

        manifest = import_synthetic_cvat_damage_annotations(
            annotations,
            image_dir,
            output_dir,
            generation_specs=specs,
        )

        self.assertEqual(manifest["image_count"], 2)
        self.assertEqual(manifest["box_count"], 2)
        self.assertEqual(manifest["domain"], "synthetic")
        self.assertEqual(manifest["source_image_unique_sha256_count"], 2)
        self.assertEqual(
            manifest["canonical_damage_level_counts"],
            {"possible_damage": 1, "severe_visible_damage": 1},
        )
        self.assertEqual(
            sorted(path.name for path in (output_dir / "images").glob("*.png")),
            ["first.png", "second.png"],
        )

        with (output_dir / "damage_box_annotations.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            box_rows = list(csv.DictReader(source))
        self.assertTrue(all(row["annotation_id"].startswith("SCVAT_") for row in box_rows))
        self.assertEqual(
            [row["component_class"] for row in box_rows],
            ["wheel_tire", "weapon_station"],
        )
        self.assertTrue(all(not row["split"] for row in box_rows))

        with (output_dir / "source_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            image_rows = list(csv.DictReader(source))
        self.assertEqual([row["prompt_id"] for row in image_rows], ["01_first", "02_second"])
        self.assertEqual([row["annotation_count"] for row in image_rows], ["1", "1"])
        self.assertTrue(all(row["domain"] == "synthetic" for row in image_rows))
        self.assertTrue(all(not row["parent_source_image"] for row in image_rows))

        sanitized = (output_dir / "annotations.xml").read_text(encoding="utf-8")
        self.assertNotIn("private@example.com", sanitized)
        self.assertNotIn("worker@example.com", sanitized)
        self.assertNotIn("example.invalid/private", sanitized)

    def test_import_rejects_unlisted_source_image(self) -> None:
        annotations, image_dir, specs = write_fixture(self.root)
        Image.new("RGB", (32, 32), "red").save(image_dir / "unexpected.png")

        with self.assertRaisesRegex(ValueError, "image/specification sets differ"):
            import_synthetic_cvat_damage_annotations(
                annotations,
                image_dir,
                self.root / "output",
                generation_specs=specs,
            )


if __name__ == "__main__":
    unittest.main()
