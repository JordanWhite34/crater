# Military damage image staging

This directory stages component-damage classification data separately from the
military component detector export.

## Synthetic pilot

`source/synthetic_humvee/` contains image-generation edits derived from the
immutable Humvee component source images. These are candidate training assets,
not ground truth. Review every image for edit locality, component identity,
damage-label fit, and generation artifacts before admitting it to a prepared
split.

The current pilot contains 30 full images: two source scenes for each of
`windshield`, `wheel_tire`, `weapon_station`, `engine_bay`, and `door`, with
`no_visible_damage`, `possible_damage`, and `severe_visible_damage` versions
for each scene.

All variants derived from one source image must inherit the same split and
group ID. Never split an original and its generated variants across train,
validation, or test.

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
