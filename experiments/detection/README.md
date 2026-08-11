# Detection experiments

This directory contains the two-stage component detector lineage:

1. `yolox_s_carparts23.py` learns civilian vehicle-part features from 23
   Carparts23 classes.
2. `yolox_s_crater6.py` changes the head to the six military CRATER classes and
   initializes all compatible weights from the selected civilian checkpoint.

Both use the pinned YOLOX implementation in `third_party/YOLOX`. The local
`standard_coco_evaluator.py` keeps COCO evaluation portable on Windows.

## Civilian pretraining

```powershell
python third_party\YOLOX\tools\train.py `
  -f experiments\detection\yolox_s_carparts23.py `
  -expn yolox_s_carparts23_pretrained `
  -d 1 -b 8 --fp16 `
  -c weights\yolox_s.pth `
  max_epoch 100 warmup_epochs 5 no_aug_epochs 15 eval_interval 5
```

## Military fine-tuning

After creating `datasets/military/components`, initialize from the selected
civilian checkpoint:

```powershell
python third_party\YOLOX\tools\train.py `
  -f experiments\detection\yolox_s_crater6.py `
  -expn yolox_s_crater6_finetune `
  -d 1 -b 8 --fp16 `
  -c outputs\detection\civilian\<civilian-run>\best_ckpt.pth
```

YOLOX skips the incompatible 23-class prediction tensors and loads the shared
backbone, neck, and compatible head weights. Select checkpoints using validation
AP only. Evaluate the selected checkpoint on the test annotations once.
