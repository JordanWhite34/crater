# Commons damage-labeling subset v1

This fixed 16-image pilot is the default input to
`experiments/damage/military_damage_crop_labeling.ipynb`.

Coverage was selected before crop labeling:

- 5 intact source images;
- 7 moderate-damage candidates; and
- 4 severe-damage candidates.

The images include wheeled and tracked vehicles, body and windshield damage,
external communications equipment, and mounted weapons. These source-level
bands are coverage aids, not crop labels. The human-provided crop CSVs are the
only damage labels consumed by the prepared per-head datasets.

`selection_manifest.csv` preserves the exact Commons page, author, license,
source URLs, local SHA-256, selection order, candidate components, and selection
rationale for every file. Reproduce or verify the subset with:

```powershell
python tools\data\prepare_commons_damage_labeling_subset.py
```
