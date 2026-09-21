# Humvee images awaiting labeling

This folder contains all 36 synthetic images currently awaiting human visual
annotation: the earlier gap/drone sets plus the latest 12-image expansion based
heavily on real military Humvee imagery.

They are intentionally separate from labeled source inventories. Use the
per-batch metadata to guide sampling, but treat every image here as unlabeled
until component boxes and damage attributes are reviewed in CVAT. Keep these
images out of validation and test splits.

Included sources:

- `synthetic_humvee_gap_v1`: 6 images
- `synthetic_humvee_gap_v2`: 8 images
- `synthetic_humvee_drone_v1`: 10 images
- `synthetic_unlabeled_batch_v2`: 12 images

Reviewed labels for all 36 images are available in
`datasets/military/damage/source/synthetic_unlabeled_review_v1`.
