# Damage-classification experiments

Damage classification begins only after a labeled component-crop dataset
exists. Keep it separate from detection so detector AP and damage accuracy
remain independently interpretable.

The intended lineage is:

1. train crop classifiers on civilian damaged/undamaged parts;
2. fine-tune those checkpoints on military component crops; and
3. evaluate both crop-level performance and the end-to-end detector-to-crop
   pipeline.

Use the canonical levels and component groups in `configs/taxonomy.json`.
Initial models should use three broad groups: `mobility`, `structure`, and
`mission_equipment`. The civilian data has no direct mission-equipment analogue,
so that head starts with military data rather than pretending transfer exists.

Each `labels.csv` should minimally contain:

```text
crop_id,source_image_id,component_instance_id,component_class,component_group,damage_level,source_domain,split
```

Generate training crops from ground-truth boxes. Predicted detector crops belong
in a separate end-to-end evaluation set because they include localization and
missed-detection error.

## Detector-driven crop labeling notebook

`military_damage_crop_labeling.ipynb` creates a deliberately separate
annotation run from predicted Humvee detector boxes. It loads the selected
fine-tuned checkpoint, maps its received six classes into `mobility`,
`structure`, and `mission_equipment`, assigns stable crop IDs, and displays all
crops on paginated contact sheets.

By default the notebook uses the fixed 16-image `commons_labeling_subset_v1`
pilot: 5 intact, 7 moderate-damage, and 4 severe-damage source images selected
to cover wheeled mobility, tracked mobility, structure, communications, and
mounted weapons. The exact Commons pages, licenses, hashes, and selection
rationale are preserved in the subset manifest.

The pilot has been executed through detector inference, crop export, and review
sheet generation. It produced three mobility pages, two structure pages, and
one mission-equipment page. The human label uploads and final validation/split
cell remain unexecuted, so there is no validated damage dataset or trained
damage classifier yet. Generated annotation-run assets are ignored and must be
restored or reproduced when continuing the pilot.

Each run is written under
`datasets/military/damage/annotation_runs/predicted_crops/<run_name>/` and
contains clean crops, a provenance manifest, contact sheets, and one blank
two-column label template per part group. Completed uploads use the requested
numeric rubric: `1` none, `2` moderate, and `3` severe. Validation requires
valid IDs and values, but blank or omitted rows are allowed. It writes only
nonblank labels to `validated_damage_labels.csv`; unlabeled crops and their
images are excluded from the prepared datasets. The final notebook cell assigns
source images—not individual crops—to deterministic train/validation/test
splits and creates a separate image/label tree for each head. This does not
change the older canonical `possible_damage` taxonomy implicitly.

These predicted crops must remain identifiable as detector-derived data. They
are useful for human damage labeling and later pipeline experiments, but they
do not become ground-truth component boxes merely because a human labels their
damage severity.
