# Detection experiments

This directory contains the two-stage component detector lineage:

1. `yolox_s_carparts23.py` learns civilian vehicle-part features from 23
   Carparts23 classes.
2. `yolox_s_crater6.py` changes the head to the six military CRATER classes and
   initializes all compatible weights from the selected civilian checkpoint.

Both use the pinned YOLOX implementation in `third_party/YOLOX`. The local
`standard_coco_evaluator.py` keeps COCO evaluation portable on Windows.

## Current results

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
