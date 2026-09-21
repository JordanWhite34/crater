# Synthetic Humvee additions v2

This is the validated, training-only import of the 36 PNGs supplied in
`Desktop/synth_label` and CVAT task `2585774`. The original CVAT export was
imported on 2026-09-17 with the repository's synthetic-damage importer; owner,
assignee, and segment URL metadata were removed from the stored XML.

## Inventory

- 36 unique PNG images and 244 manual boxes
- six component labels, normalized to the existing Humvee taxonomy
- 187 `no_visible_damage`, 27 `possible_damage`, 22
  `severe_visible_damage`, and 8 `unobservable` boxes
- one box-free image, `08_comms_severe.png`; because its filename suggests a
  positive communications-damage scene, it is excluded from both training tasks
  pending a corrected annotation rather than treated as a detector negative
- no synthetic image is eligible for validation or test metrics

The filename stem is retained as the prompt/provenance identifier because a
separate prompt manifest was not supplied. Filename intent is not used as a
training label; the human CVAT box attributes are authoritative.

## Files

- `images/`: byte-for-byte PNG copies
- `annotations.xml`: privacy-sanitized CVAT for-images 1.1 export
- `damage_box_annotations.csv`: canonical boxes and damage labels
- `source_manifest.csv`: hashes, dimensions, provenance, and group IDs
- `import_manifest.json`: validation summary and artifact hashes

## Reproduce the import

```powershell
python tools\data\import_synthetic_cvat_damage_annotations.py `
  "<path-to-export>\annotations.xml" `
  --source-image-dir "<path-to-synth_label>" `
  --output-dir datasets\military\damage\source\synthetic_humvee_additions_v2 `
  --dataset-version synthetic-humvee-additions-v2 `
  --infer-specs-from-filenames `
  --generation-session-id cvat-task-2585774
```

The end-to-end benchmark notebook displays the supplied annotations before
training and keeps both synthetic releases training-only.
