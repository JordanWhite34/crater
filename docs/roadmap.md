# CRATER roadmap

Updated: August 21, 2026

## Stage 1: civilian detector pretraining

Status: baseline complete; pretrained comparison remains.

- Reproduce the prepared 23-class Carparts23 data.
- Retain the completed YOLOX-S scratch run as a baseline.
- Train YOLOX-S from official COCO initialization under the same split and
  schedule.
- Select between scratch and pretrained runs using validation AP and per-class
  behavior; evaluate the selected model once on the civilian test split.
- Save the parent checkpoint and a concise error-analysis report.

Deliverable: versioned civilian detector checkpoint with reproducible metrics.

## Stage 2: military component detection

Status: experiment scaffolded; first Humvee source export received but not yet
training-ready.

- Record source/license and grouping metadata for the received 224-image
  Humvee export; continue collecting other military vehicle families.
- Preserve the six received labels for the controlled Humvee comparison; do
  not silently remap them to the canonical CRATER taxonomy.
- Create leakage-resistant training, validation, and test splits.
- Run the controlled random-versus-Carparts initialization notebook using the
  received export taxonomy; treat image-level split results as exploratory.
- Treat a later canonical CRATER six-class dataset as a separate annotation
  effort governed by `configs/taxonomy.json` and the military guide.
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
