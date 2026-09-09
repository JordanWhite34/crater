# Humvee CVAT component-damage annotations v1

Status: validated source annotation inventory; not training-ready until source
grouping and split assignment are complete.

This source preserves the September 1, 2026 export of CVAT job `4366557`. The
job covers the same 224 images versioned under
`datasets/military/components/source/humvee/` and adds a human `damage_state`
attribute to every component box.

## Files

- `annotations.xml`: CVAT for images 1.1 XML with account ownership, assignment,
  and segment URL metadata removed. Image names, box coordinates, labels,
  damage attributes, job timing, and non-personal job metadata are unchanged.
- `damage_box_annotations.csv`: flat, validated box inventory with source-image
  hashes, normalized component classes, CRATER component groups, original CVAT
  damage states, and canonical CRATER damage levels.
- `import_manifest.json`: file hashes, validation counts, mappings, comparison
  with the previous component annotations, and split status.

The original local export is identified by SHA-256
`93ea8863bd489104017381a742c6545834b3174877f8330626523c772b8d574a`.
It is not versioned because it contains CVAT account metadata.

## Label mappings

The received CVAT class `weapons_station` is normalized to the established
Humvee source class `weapon_station`. All other component names are preserved.

| CVAT damage state | Canonical CRATER damage level |
| --- | --- |
| `intact` | `no_visible_damage` |
| `degraded` | `possible_damage` |
| `destroyed` | `severe_visible_damage` |
| `unknown` | `unobservable` |

The original values remain in `damage_state_source`; the mapping does not erase
the annotator's terminology.

## Validated inventory

- 224 source images; every filename, dimension, and file hash matches the
  versioned Humvee source.
- 2,240 in-bounds manual boxes; 2 images contain no boxes.
- No missing damage attributes, unknown labels, duplicate annotation IDs, or
  unsupported CVAT shapes.
- 2,135 intact, 32 degraded, 42 destroyed, and 31 unknown boxes.
- The previous component COCO export had 2,226 boxes. At two-decimal coordinate
  precision, 2,210 are unchanged, 30 are new or adjusted, and 16 were removed
  or adjusted. The previous detector source remains intact as its own version.

Damage coverage is not balanced. `comms_equipment` has 205 intact boxes and no
degraded, destroyed, or unknown boxes; `weapon_station` has only three damaged
boxes. Use these labels as validated source data, not as evidence that every
damage head has adequate training support.

CVAT defines `intact` as the default `damage_state`. The XML proves that every
box has a valid exported value, but it cannot distinguish an explicitly
confirmed intact box from an untouched default. Treat semantic review of the
2,135 intact boxes as a separate QA gate unless the annotation procedure
already required every box to be reviewed.

## Reproduce the import

From the repository root, run:

```powershell
python tools\data\import_cvat_damage_annotations.py `
  "<path-to-CVAT-export>\annotations.xml"
```

The importer rejects DTD/entity declarations, unexpected annotation shapes,
missing or altered source images, unknown labels or damage values, duplicate
annotations, and out-of-bounds boxes. It verifies every source image against
`source_manifest.csv` before writing outputs.

## Training boundary

The CSV intentionally leaves `split` blank. Authoritative source, scene,
vehicle, and near-duplicate grouping is incomplete for the Humvee collection.
Assign groups and splits before materializing crops; all boxes from one source
image and all related source images must remain in the same split.
