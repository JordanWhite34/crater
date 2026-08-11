# Architecture and experiment boundaries

## 1. Civilian component detector

YOLOX-S learns localization features from the 23-class Carparts23 dataset. This
stage produces a general vehicle-part detector checkpoint; it does not establish
performance on military vehicles.

Primary metrics are COCO AP50:95, AP50, recall, and per-class AP on the civilian
validation and test splits.

## 2. Military component detector

Create a new six-class head for `windshield`, `wheel`, `track`, `hull`,
`comms_system`, and `mounted_gun`. Initialize every shape-compatible YOLOX
weight from the selected civilian detector and fine-tune on military imagery.

Measure performance by component, vehicle family, viewpoint, object size, and
visibility. Record the exact civilian parent checkpoint in every military run.

## 3. Damage classifiers

The detector yields component boxes. A crop generator turns a ground-truth or
predicted box into a classifier input while retaining provenance. Initial damage
models operate on the broad groups `mobility`, `structure`, and
`mission_equipment` using the four labels in `configs/taxonomy.json`.

Civilian crop training and military crop fine-tuning are separate runs. There is
no civilian mission-equipment analogue, so the project must report that missing
transfer path explicitly.

Evaluate crop classifiers with confusion matrices, macro F1, per-class recall,
calibration, and abstention/unobservable behavior. Then evaluate the connected
detector-to-classifier pipeline separately; its errors include bad boxes and
missed components.

## 4. Operational-effect reasoning

A later configurable Bayesian or noisy-OR layer can turn component observations
into vehicle-level effect estimates. It consumes perception outputs but is not
trained as part of either vision model. This separation preserves an auditable
chain from detections and visible cues to inferred effects.

## Data-leakage rule

Assign source assets, scenes, vehicles, and video sequences to splits before
augmentation or crop generation. All derivatives inherit the source split.
Never allow crops or near-duplicate frames from one source group to cross train,
validation, and test boundaries.
