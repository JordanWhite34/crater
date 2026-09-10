# Commons damage-labeling subset v1

This is a historical 16-image crop-labeling pilot. Its images and provenance are
preserved, but it is not used by the active CVAT damage-classification notebook.

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
rationale for every file. The retired preparation scripts and notebook are in
`outputs/repo_cleanup/before_simplification_20260909_122427.zip` on the original
machine and in Git history for tracked files.
