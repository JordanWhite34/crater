# CRATER

CRATER is an interpretable vehicle-component and visible-damage research
pipeline. The model lineage is intentionally staged so each transfer step can
be measured on its own:

```text
civilian images -> 23-class component detector
                         |
                         v transfer shared detector weights
military images -> 6-class component detector -> component crops
                                                     |
             civilian damage crops -> damage classifier
                                                     |
                                                     v fine-tune
                                      military damage classifier
```

Detector outputs answer **which parts are visible and where**. Damage models
answer **what visible damage is present in a part crop**. Keeping those datasets,
experiments, metrics, and checkpoints separate makes failures attributable.

## Current status

- The leakage-resistant civilian Carparts23 dataset preparation is complete.
- Both 100-epoch civilian YOLOX-S runs are complete. The scratch baseline
  reached validation COCO AP50:95 `0.422` and AP50 `0.642`; the run initialized
  from official YOLOX-S COCO weights reached AP50:95 `0.546` and AP50 `0.688`.
- The COCO-initialized checkpoint is promoted as
  `carparts23-yolox-s-coco-v1`. Held-out test COCO AP50:95 is `0.580`, AP50 is
  `0.737`, and AR100 is `0.779`; see the
  [Stage 1 detector report](docs/reports/carparts23-stage1.md).
- The `object` class is a known dataset limitation: 5 train instances, 2
  validation instances, and no test instances. It remains only for checkpoint
  compatibility and should be removed or relabeled in a future dataset version.
- The exploratory Humvee initialization comparison is complete on the received
  224-image, six-label source export. Under the same 100-epoch schedule, scratch
  initialization reached validation AP50:95 `0.1297`; initialization from
  `carparts23-yolox-s-coco-v1` reached `0.4784`, an absolute gain of `0.3487`.
  See the [Humvee comparison report](docs/reports/humvee-source6-initialization-comparison.md).
- The Humvee result is transfer evidence, not a release-quality military model.
  The split uses interim photo-series grouping, authoritative provenance is
  incomplete, the received labels are not the canonical CRATER taxonomy, and a
  full held-out test evaluation and versioned checkpoint promotion remain.
- A 16-image military damage-labeling pilot has generated detector-derived
  crops and review sheets. Human crop labels have not been completed, and no
  civilian or military damage classifier has been trained.

The scratch checkpoint is retained for comparison. The required transfer
lineage is the promoted COCO-initialized civilian checkpoint followed by
military fine-tuning. Generated datasets, checkpoints, logs, annotation runs,
and outputs remain ignored local artifacts. The approved Humvee source images
are the explicit exception and are versioned through Git LFS. The exploratory
Humvee run outputs must be restored or reproduced before downstream work; the
executed notebook records their results but does not version the checkpoints.

## Repository map

```text
configs/
  taxonomy.json                 canonical classes, damage levels, and groups
datasets/                       ignored by default; selected sources use LFS
docs/
  architecture.md               stage boundaries and evaluation policy
  roadmap.md                    work sequence and deliverables
  annotation/                   military component and damage-label rules
  reports/                      versioned stage results and evidence tables
experiments/
  detection/                    civilian pretraining and military fine-tuning
  damage/                       crop-classification experiment contract
outputs/                        ignored runs grouped by task and domain
third_party/YOLOX/              pinned upstream detector implementation
tools/data/                     reproducible dataset preparation tools
weights/                        ignored external initialization checkpoints
```

## Reproduce the civilian detector

Install dataset-preparation dependencies and create the local dataset:

```powershell
python -m pip install -r requirements.txt
python tools\data\download_carparts23.py
```

In a CUDA environment, install the pinned YOLOX checkout and train:

```powershell
python -m pip install -v -e third_party\YOLOX

python third_party\YOLOX\tools\train.py `
  -f experiments\detection\yolox_s_carparts23.py `
  -expn yolox_s_carparts23_pretrained `
  -d 1 -b 8 --fp16 `
  -c weights\yolox_s.pth `
  max_epoch 100 warmup_epochs 5 no_aug_epochs 15 eval_interval 5
```

Increase or decrease batch size only for available GPU memory. Do not use test
metrics for checkpoint or hyperparameter selection.

For the military fine-tuning command and expected dataset contract, see
`experiments/detection/README.md`. The canonical class names live in
`configs/taxonomy.json`; experiment code should not invent alternate spellings.

## Artifact convention

```text
outputs/detection/civilian/<run>/
outputs/detection/humvee_source6/<run>/    exploratory received-taxonomy runs
outputs/detection/military/<run>/
outputs/damage/civilian/<run>/
outputs/damage/military/<run>/
```

A run is meaningful only with its experiment file, starting checkpoint, data
version, split manifest, and validation metrics. Preserve those together when a
checkpoint is promoted to the next stage.

This is a research prototype and decision-support project, not an operationally
validated or autonomous battle-damage assessment system.
