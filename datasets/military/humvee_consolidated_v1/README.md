# Consolidated Humvee image inventory v1

This directory consolidates the source-level real and synthetic Humvee images
currently in the repository into one flat inventory. Original datasets are left
unchanged.

## Inventory

| source | images |
| --- | ---: |
| real Humvee component collection | 224 |
| real Commons damage-labeling subset | 16 |
| real Wikimedia Commons candidates | 4 |
| synthetic Humvee damage v1 | 30 |
| synthetic gap v1 | 6 |
| synthetic gap v2 | 8 |
| synthetic drone v1 | 10 |
| **total** | **298** |

Files are renamed with a provenance prefix, for example
`real_components__commons_101805120.jpg` and
`synthetic_drone_v1__15_desert_comms_severe.png`. The original per-batch
synthetic label files remain authoritative for intended component and damage
labels; the synthetic images are training-only candidates pending visual QA.

The consolidated directory contains source images only, not derived crops or
model-predicted crops. Keep real and synthetic images separated by provenance
when assigning train/validation/test splits; synthetic images must not enter
validation or test sets.
