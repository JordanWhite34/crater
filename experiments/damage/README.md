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
