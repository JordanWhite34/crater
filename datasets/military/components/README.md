# Military component dataset

This directory separates immutable source deliveries from the prepared
six-class COCO dataset consumed by `yolox_s_crater6.py`.

## Source inventory

### Humvee component export

Local path: `source/humvee/`

- 224 images: 222 JPEG and 2 PNG files (110,157,833 bytes total)
- 2,226 COCO bounding-box annotations
- 3 images with no annotations
- source annotation file: `instances_default.json`
- tracked image integrity/provenance inventory: `source_manifest.csv`
- annotation SHA-256:
  `6CF31604CB44C214F0E36EFFDDCCC53047B1B016D16C2A6289FEE0A671A54637`

The source export stays byte-for-byte intact. Its image names are relative to
the export directory, so the JSON and images intentionally remain together.
The 224 source images are versioned through Git LFS; the COCO JSON, README, and
manifest use normal Git. The manifest records each image's hash, dimensions,
size, and annotation count. Its blank provenance and grouping columns are
intentional work items, not evidence that those values are unknown in
principle. Fill those fields before treating the dataset as provenance-complete
or redistributing it beyond the approved repository scope.

## Relationship to the canonical taxonomy

The initialization comparison intentionally preserves the received labels.
They are not the canonical CRATER six-class taxonomy; the table below records
the relationship for context, not as an automatic mapping:

| Received label | Instances | CRATER class | Disposition |
| --- | ---: | --- | --- |
| `wheel_tire` | 802 | `wheel` | likely rename; confirm annotation scope |
| `windshield` | 282 | `windshield` | exact name match |
| `door` | 413 | none | do not remap without a taxonomy decision |
| `engine_bay` | 269 | none | do not remap without a taxonomy decision |
| `weapon_station` | 255 | `mounted_gun` | possible scope mismatch; review boxes |
| `comms_equipment` | 205 | `comms_system` | likely rename; confirm annotation scope |

Do not relabel `door` or `engine_bay` as `hull` or `track`. A Humvee has no
track instances, and those boxes do not establish the canonical `hull` extent.

## Release-quality preparation gates

Before fine-tuning:

1. Add source URL, license, vehicle/scene identity, and near-duplicate grouping
   metadata. The current COCO `info` and `licenses` records are empty.
2. Preserve the source categories for this comparison and review the three
   images with no boxes as intentional negatives or annotation omissions.
3. Split by source asset, scene, vehicle, and sequence before any augmentation.
4. Generate `annotations/humvee_source6_instances_{train,val,test}.json` files
   that point back to the versioned source images.
5. Validate image references, box bounds, category IDs, class coverage, and
   cross-split leakage before launching training.

The comparison notebook applies an interim leakage-resistant grouping based on
Commons upload sequences plus visually confirmed photo-series links. It checks
that no group crosses splits without changing the source export. Authoritative
provenance fields are still required before treating results as release-quality.
A later canonical CRATER detector is a separate annotation and experiment
lineage.
