# CRATER roadmap

Updated: September 9, 2026

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

Status: exploratory initialization comparison complete; release-quality
dataset, evaluation, and model promotion remain.

- Completed: preserved the 224-image, 2,226-box source export and its six
  received labels without silently remapping them to canonical CRATER classes.
- Completed: created a deterministic interim photo-series split with 156 train,
  34 validation, and 34 test images across 116 groups; the automated check
  found no group crossing splits.
- Completed: ran matched 100-epoch scratch and
  `carparts23-yolox-s-coco-v1`-initialized experiments. Validation AP50:95 was
  `0.1297` from scratch and `0.4784` from the civilian checkpoint, a `0.3487`
  absolute improvement. The initialized run is selected for pilot inference.
- Completed: visually smoke-tested the selected checkpoint on three test-split
  images after selection. This is not a full quantitative test evaluation.
- Pending: restore or reproduce the selected checkpoint, record its SHA-256 and
  complete run lineage, and preserve it with the experiment evidence.
- Pending: record authoritative source URL, license, vehicle/scene identity,
  and near-duplicate grouping metadata; review the three images without boxes.
- Pending: run one fixed aggregate held-out test evaluation and publish
  per-class, viewpoint, size, and visibility error analysis.
- Pending: continue collecting other military vehicle families.
- Treat a later canonical CRATER six-class dataset as a separate annotation
  effort governed by `configs/taxonomy.json` and the military guide.

Current evidence: the civilian checkpoint materially improves exploratory
Humvee validation performance; see the
[initialization comparison report](reports/humvee-source6-initialization-comparison.md).

Deliverable: provenance-complete canonical six-class military detector,
versioned checkpoint, and error-analysis report.

## Stage 3: civilian damage classification

Status: data contract defined; source dataset and training code required.

- Source civilian damaged and undamaged vehicle-part imagery with provenance.
- Split source images first, then create ground-truth component crops.
- Label damage level and visible cues using the damage guide.
- Train mobility and structure classifiers; record that mission equipment has no
  civilian transfer source.
- Measure confusion, class recall, calibration, and abstention.

Deliverable: civilian crop-classifier checkpoints and calibrated metrics.

## Stage 4: military damage fine-tuning

Status: real and synthetic damage-box inventories imported; training-ready
grouped crops, synthetic QA, and additional damaged examples remain.

- Completed: added the [two-stage damage notebook](../experiments/damage/part_damage_classification.ipynb)
  and tested its preparation/training helpers. It trains a real-image ResNet18
  baseline per component group, then fine-tunes from those weights using
  synthetic crops, retaining real-only validation and test splits. No production
  damage training has run; grouping and annotation review still precede it.
- Completed: compared `Desktop/synthetic` with the existing synthetic import;
  all 30 images and the recorded original annotation-export hash match.
- Completed: built the detector-driven crop annotation workflow and ran its
  16-image pilot through crop and contact-sheet generation.
- Completed: validated and imported a 224-image CVAT job with 2,240 manual
  component boxes and a damage state on every box. Canonical mapping yields
  2,135 `no_visible_damage`, 32 `possible_damage`, 42
  `severe_visible_damage`, and 31 `unobservable` instances.
- Completed: validated and imported a separate 30-image synthetic HMMWV source
  with 251 manual CVAT boxes, byte-level image lineage, and prompt provenance.
  It contains 188 `no_visible_damage`, 22 `possible_damage`, 31
  `severe_visible_damage`, and 10 `unobservable` boxes.
- Pending: visually QA the synthetic images and annotations for component
  identity, box fit, label fit, and generation artifacts. Admit accepted crops
  to training only; do not use synthetic images for validation or test metrics.
- Pending: complete authoritative source/scene/vehicle grouping, assign splits,
  and materialize ground-truth crops without crossing source groups.
- Pending: add damaged communications-equipment examples and expand the sparse
  degraded/destroyed classes before treating the data as training-scale. The
  synthetic source adds only one degraded communications-equipment box and no
  destroyed communications-equipment boxes.
- Pending: complete the separate 16-image predicted-crop pilot label uploads;
  the final label and prepared-split notebook cell has not run.
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
