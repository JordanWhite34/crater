# Synthetic Humvee damage source v1

This directory is the validated source inventory for 30 independently generated
HMMWV images and their human CVAT component-damage annotations. It is a
synthetic, training-only candidate source. It is not an evaluation set or a
prepared crop dataset.

## Inventory

- 30 unique, decodable RGB PNG images (77,564,902 bytes total)
- 251 manual component boxes; every image has at least one box
- CVAT for images 1.1, task `2559206`
- no duplicate filenames, duplicate image hashes, duplicate boxes, invalid
  labels, missing attributes, or out-of-bounds coordinates
- 28 images at 1536 x 1024, one at 1628 x 966, and one at 1499 x 1049

| Source state | CRATER level | Count | Share |
| --- | --- | ---: | ---: |
| `intact` | `no_visible_damage` | 188 | 74.9% |
| `degraded` | `possible_damage` | 22 | 8.8% |
| `destroyed` | `severe_visible_damage` | 31 | 12.4% |
| `unknown` | `unobservable` | 10 | 4.0% |

| Component class | Total | Intact | Degraded | Destroyed | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wheel_tire` | 88 | 59 | 9 | 10 | 10 |
| `door` | 60 | 50 | 5 | 5 | 0 |
| `engine_bay` | 31 | 23 | 2 | 6 | 0 |
| `windshield` | 30 | 21 | 4 | 5 | 0 |
| `weapon_station` | 28 | 22 | 1 | 5 | 0 |
| `comms_equipment` | 14 | 13 | 1 | 0 | 0 |

CVAT used the source spelling `weapons_station`; the canonical CSV normalizes
it to the established Humvee class `weapon_station` while retaining both
values. Damage mappings match the real Humvee CVAT source.

## Files

- `images/`: byte-for-byte copies of the 30 generated PNGs, versioned with Git
  LFS
- `annotations.xml`: privacy-sanitized CVAT export
- `damage_box_annotations.csv`: canonical, box-level inventory with stable
  `SCVAT_*` IDs and an intentionally blank `split`
- `source_manifest.csv`: image hashes, dimensions, annotation counts, prompt
  lineage, synthetic-domain marker, and grouping fields
- `import_manifest.json`: dataset-level hashes, validation counts, mappings,
  and use policy

## Provenance and use policy

The images were independently generated in OpenAI image-generation session
`01a04411-3958-7240-9f6b-677f4af4af60`. The exact generator model was not
recorded. They are text-to-image assets, not edits of the real Humvee source;
therefore `parent_source_image` is blank and each prompt has its own generation
group.

`intended_condition` records what generation requested. It is provenance only:
training labels come exclusively from the human CVAT `damage_state` attribute.
This distinction matters because a requested feature can be absent, malformed,
or not visibly boxable. For example, this export has only one degraded
communications-equipment box and no destroyed communications-equipment boxes,
even though some prompts requested destroyed communications equipment.

Before materializing crops:

1. visually review component identity, box fit, damage-state fit, and generation
   artifacts;
2. explicitly confirm boxes left at CVAT's default `intact` value;
3. keep every synthetic image out of validation and test sets;
4. retain `domain=synthetic` and generation lineage in every derived crop; and
5. decide whether the class imbalance and communications-equipment gap require
   more generation or labeling.

The blank split is intentional. Synthetic images may later be admitted to the
training partition after QA, but they must never make real-image validation or
test metrics look better.

## Reproduce the import

From the repository root:

```powershell
python tools\data\import_synthetic_cvat_damage_annotations.py `
  "<path-to-synthetic-folder>\annotations.xml" `
  --source-image-dir "<path-to-synthetic-folder>"
```

The importer validates the exact 30-file generation specification, image
decodability and uniqueness, XML schema, filename/dimension agreement, label
taxonomy, attributes, and box geometry before replacing generated artifacts.
