# Military damage image staging

This directory stages component-damage classification data separately from the
military component detector export.

## Human Humvee damage boxes

`source/humvee_cvat_damage_v1/` is the validated CVAT for images 1.1 export for
the 224 versioned Humvee source images. It contains 2,240 manual component boxes
and a human `damage_state` attribute on every box. The import preserves the
received values and adds a canonical mapping:

| Source value | CRATER level | Count |
| --- | --- | ---: |
| `intact` | `no_visible_damage` | 2,135 |
| `degraded` | `possible_damage` | 32 |
| `destroyed` | `severe_visible_damage` | 42 |
| `unknown` | `unobservable` | 31 |

The source is usable for ground-truth crop generation after grouped splits are
assigned. It is not yet training-ready: provenance grouping remains incomplete,
damage labels are highly imbalanced, and all 205 communications-equipment boxes
are currently intact. Do not mix its ground-truth boxes with the separate
detector-predicted crop pilot.

Because `intact` was CVAT's default attribute value, the file format alone
cannot prove that every intact box was actively reviewed. Confirm that semantic
QA before using the intact population as classifier ground truth.

Reproduce the validated, privacy-sanitized source inventory with
`tools/data/import_cvat_damage_annotations.py`. The importer verifies all 224
source-image hashes before writing the canonical CSV.

## Synthetic Humvee damage source

`source/synthetic_humvee_damage_v1/` contains 30 independently generated
text-to-image HMMWV scenes and 251 human CVAT component boxes. Every box has an
`intact`, `degraded`, `destroyed`, or `unknown` attribute, mapped through the
same canonical contract as the real Humvee source. The images, sanitized XML,
box CSV, source manifest, and import manifest are versioned; PNGs use Git LFS.

This is a validated source inventory, not a prepared dataset. The labels are
74.9% intact, and communications equipment has 13 intact boxes, one degraded
box, and no destroyed boxes. Generation-prompt intent is retained only as
provenance because requested damage is not guaranteed to appear or be visibly
boxable. Review component identity, box fit, damage-state fit, and generation
artifacts before admitting any crop to training.

These are independent generations rather than edits of real source images, so
they have no real-image parent. Keep them out of validation and test data, and
retain their synthetic-domain and generation-group fields in every derivative.
See the [source README](source/synthetic_humvee_damage_v1/README.md) for the
verified counts and reproduction command.

## Wikimedia Commons candidate catalog

`source/wikimedia_commons_candidates/` is a provenance-rich search and review
queue for real military vehicles across the six CRATER component classes. The
raw `manifest.csv` retains all search results for audit. `curated_manifest.csv`
removes metadata-confirmed false positives, applies conservative candidate
damage levels, and assigns known photo-series groups. The catalog is not a
training set: every retained record still requires visual review, component
boxes/crops, final damage/cue labels, and a leakage-safe split.

The catalog is reproducible with:

```powershell
python tools\data\download_commons_military_damage.py --per-query 12 --max-images 150 --catalog-only
python tools\data\curate_commons_military_damage.py
```

Omit `--catalog-only` to fetch missing image files. Wikimedia may throttle bulk
media downloads, so URL-only rows are expected and are explicitly identified by
`download_status`. Each row includes its Commons page, media URL, author,
license, description, SHA-256 when downloaded, source asset ID, candidate
components, review status, and grouping fields. The candidate component and
damage fields come from search/metadata and must not be treated as annotations.

`source/commons_labeling_subset_v1/` is the fixed small annotation pilot used by
the damage notebook. Its 16 images span 5 intact, 7 moderate-damage, and 4
severe-damage source scenes. `selection_manifest.csv` records the source page,
license, author, image hash, candidate components, damage band, and the reason
each image was selected. The bands guide coverage only; humans label damage at
the predicted-crop level.

## Predicted-crop annotation runs

The damage-labeling notebook writes immutable-by-name annotation runs under
`annotation_runs/predicted_crops/<run_name>/`. A run contains:

- `crops_manifest.csv` with detector, source, box, checkpoint, and crop lineage;
- clean crop images under `crops/<part_group>/`;
- ID-overlaid review pages under `contact_sheets/<part_group>/`;
- blank two-column CSVs under `label_templates/`; and
- a `label_uploads/` directory for the three completed group CSVs.

Choose a new run name when the source collection, checkpoint, confidence
threshold, or crop policy changes. The workflow refuses to overwrite an
existing run so completed human labels cannot be lost accidentally.

After the three group CSVs are uploaded, blank or omitted crop IDs are excluded.
Only nonblank severities enter `validated_damage_labels.csv` and the prepared
per-head datasets under `prepared_datasets/<name>/`. Splits are assigned by
source-image hash, so crops from one source image cannot cross train,
validation, and test.
