# Detection experiments

This directory contains the two-stage component detector lineage:

1. `yolox_s_carparts23.py` learns civilian vehicle-part features from 23
   Carparts23 classes.
2. `yolox_s_crater6.py` changes the head to the six military CRATER classes and
   initializes all compatible weights from the selected civilian checkpoint.

Both use the pinned YOLOX implementation in `third_party/YOLOX`. The local
`standard_coco_evaluator.py` keeps COCO evaluation portable on Windows.

## Humvee initialization comparison

`humvee_initialization_comparison.ipynb` is the executed training record for the
received Humvee export. It creates a deterministic group-aware split and
launches the standard YOLOX trainer twice: once from random weights and once
from the promoted `carparts23-yolox-s-coco-v1` checkpoint.

The split keeps related Commons upload sequences and visually confirmed photo
series together. Its automated check rejects any group crossing train,
validation, and test. Because authoritative source/scene/vehicle metadata is
still missing, this remains an exploratory comparison rather than a
release-quality benchmark. The notebook also verifies the promoted checkpoint's
local path and SHA-256 before starting either run.

## Civilian detector results

Both civilian runs completed for 100 epochs on the same leakage-resistant
train/validation split and schedule:

| Initialization | Validation AP50:95 | Validation AP50 |
| --- | ---: | ---: |
| Scratch | 0.422 | 0.642 |
| Official YOLOX-S COCO weights | **0.546** | **0.688** |

The COCO-initialized checkpoint achieved held-out test AP50:95 `0.580`, AP50
`0.737`, AP75 `0.675`, and AR100 `0.779`. It is promoted as
`carparts23-yolox-s-coco-v1`; use that version as the parent for military
fine-tuning. The `object` class is a documented dataset limitation with 5 train,
2 validation, and 0 test instances. Full lineage and per-class results are in
the [Stage 1 report](../../docs/reports/carparts23-stage1.md).

Generated runs and the promoted checkpoint live under
`outputs/detection/civilian/` and are intentionally ignored by Git.

## Exploratory Humvee results

The received source export contains 224 images and 2,226 boxes under six source
labels. The notebook created 116 interim photo groups and split them into 156
train images / 1,598 boxes, 34 validation images / 322 boxes, and 34 test images
/ 306 boxes. No group crossed splits.

Both runs used seed `42`, batch size 8, mixed precision, 100 epochs, 5 warmup
epochs, 15 no-augmentation epochs, and evaluation every 5 epochs.

| Initialization | Best epoch | Validation AP50:95 |
| --- | ---: | ---: |
| Random | 92 | 0.1297 |
| `carparts23-yolox-s-coco-v1` | 91 | **0.4784** |

The `0.3487` absolute validation improvement supports using the civilian model
as the military initialization. After selection, the initialized checkpoint
was also used for qualitative inference on three test-split images. Those
visualizations are a smoke test, not aggregate test metrics.

This comparison retains the received labels (`wheel_tire`, `windshield`,
`door`, `engine_bay`, `weapon_station`, and `comms_equipment`). It is not the
canonical CRATER six-class deliverable. The grouping is based on Commons upload
sequences and manually linked photo series; missing authoritative provenance
prevents release-quality interpretation. See the
[comparison report](../../docs/reports/humvee-source6-initialization-comparison.md).

Run checkpoints and logs are ignored local artifacts under
`outputs/detection/humvee_source6/`. They are not present merely because the
notebook outputs are versioned. Restore or reproduce the selected checkpoint,
record its SHA-256, and complete one fixed held-out evaluation before promotion.

## Reproduce civilian pretraining

```powershell
python third_party\YOLOX\tools\train.py `
  -f experiments\detection\yolox_s_carparts23.py `
  -expn yolox_s_carparts23_pretrained `
  -d 1 -b 8 --fp16 `
  -c weights\yolox_s.pth `
  max_epoch 100 warmup_epochs 5 no_aug_epochs 15 eval_interval 5
```

## Military fine-tuning

After creating `datasets/military/components`, initialize from the promoted
civilian checkpoint:

```powershell
python third_party\YOLOX\tools\train.py `
  -f experiments\detection\yolox_s_crater6.py `
  -expn yolox_s_crater6_finetune `
  -d 1 -b 8 --fp16 `
  -c outputs\detection\civilian\promoted\carparts23-yolox-s-coco-v1\carparts23-yolox-s-coco-v1.pth
```

YOLOX skips the incompatible 23-class prediction tensors and loads the shared
backbone, neck, and compatible head weights. Select checkpoints using validation
AP only. Evaluate the selected checkpoint on the test annotations once.
