# CRATER roadmap

Updated: August 14, 2026

## Stage 1: civilian detector pretraining

Status: complete; `carparts23-yolox-s-coco-v1` promoted.

- Completed: reproduced the prepared 23-class Carparts23 data.
- Completed: trained the 100-epoch YOLOX-S scratch baseline (validation
  AP50:95 `0.422`, AP50 `0.642`).
- Completed: trained YOLOX-S from official COCO initialization under the same
  split and schedule (validation AP50:95 `0.546`, AP50 `0.688`).
- Completed: selected the COCO-initialized run based on its validation
  AP50:95 improvement of `0.124` over scratch. Most classes improved, but the
  `object` class remained at zero AP.
- Completed: evaluated the selected checkpoint once on the held-out test split
  (AP50:95 `0.580`, AP50 `0.737`, AR100 `0.779`).
- Completed: promoted the versioned parent checkpoint with data/split lineage
  and a [per-class error-analysis report](reports/carparts23-stage1.md).
- Known limitation: `object` has 5 train, 2 validation, and 0 test instances;
  defer its removal or relabeling to a versioned dataset/model change.

Deliverable: versioned civilian detector checkpoint with reproducible metrics.

## Stage 2: military component detection

Status: experiment scaffolded; dataset required.

- Collect licensed low-altitude oblique/aerial military vehicle images.
- Annotate the six classes from `configs/taxonomy.json` using the military guide.
- Group source assets and near-duplicates before splitting.
- Initialize `yolox_s_crater6.py` from the selected civilian checkpoint.
- Evaluate by class, family, viewpoint, size, and visibility.

Deliverable: six-class military detector and error-analysis report.

## Stage 3: civilian damage classification

Status: data contract defined; dataset and training code required.

- Source civilian damaged and undamaged vehicle-part imagery with provenance.
- Split source images first, then create ground-truth component crops.
- Label damage level and visible cues using the damage guide.
- Train mobility and structure classifiers; record that mission equipment has no
  civilian transfer source.
- Measure confusion, class recall, calibration, and abstention.

Deliverable: civilian crop-classifier checkpoints and calibrated metrics.

## Stage 4: military damage fine-tuning

Status: blocked on military damage crops.

- Generate military crops without crossing source splits.
- Fine-tune compatible civilian damage checkpoints.
- Train mission-equipment damage directly on military data.
- Compare civilian initialization against training from scratch.
- Evaluate both clean ground-truth crops and predicted detector crops.

Deliverable: military damage classifiers with explicit parent lineage.

## Stage 5: end-to-end assessment

- Connect the military detector, crop preprocessing, damage models, and an
  uncertainty-aware operational reasoning layer.
- Produce structured JSON and visual overlays.
- Preserve unknown, unobservable, and abstained outcomes.
- Evaluate end-to-end failure modes and inference cost before edge deployment.

Deliverable: reproducible single-image research prototype.
