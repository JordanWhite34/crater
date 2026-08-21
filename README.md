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
- The 23-class YOLOX-S scratch baseline is complete: validation COCO AP50:95
  `0.422` and AP50 `0.642` after 100 epochs.
- A first unsplit Humvee component export is staged with its source taxonomy;
  provenance and grouped splits remain before military fine-tuning.
- Civilian and military damage-classification datasets are not yet built.

The scratch checkpoint is retained for comparison. The recommended transfer
lineage is a second civilian run initialized from official YOLOX-S COCO weights,
followed by military fine-tuning from the better validation-selected civilian
checkpoint.

## Repository map

```text
configs/
  taxonomy.json                 canonical classes, damage levels, and groups
datasets/                       ignored by default; selected sources use LFS
docs/
  architecture.md               stage boundaries and evaluation policy
  roadmap.md                    work sequence and deliverables
  annotation/                   military component and damage-label rules
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
outputs/detection/military/<run>/
outputs/damage/civilian/<run>/
outputs/damage/military/<run>/
```

A run is meaningful only with its experiment file, starting checkpoint, data
version, split manifest, and validation metrics. Preserve those together when a
checkpoint is promoted to the next stage.

This is a research prototype and decision-support project, not an operationally
validated or autonomous battle-damage assessment system.
