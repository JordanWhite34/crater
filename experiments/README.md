# CRATER training experiments

This directory contains CRATER-owned model configurations. The YOLOX source in
`third_party/YOLOX` is an upstream dependency and should not contain
project-specific dataset paths, class definitions, or training policy.

## Experiment progression

1. `yolox_s_carparts23.py`: pretrain YOLOX-S on all 23 civilian car-part
   classes.
2. `yolox_s_crater6.py`: later, initialize from the best 23-class checkpoint and
   fine-tune a new six-class detection head on military CRATER annotations.

Only create the second file when the six-class dataset exists. A separate
experiment file preserves the baseline and makes the transfer-learning boundary
reviewable.

## What YOLOX already provides

Do not write a second training loop or a new COCO parser. The pinned YOLOX code
already provides:

- `COCODataset` for images, COCO bounding boxes, and class IDs;
- `TrainTransform` for image preprocessing, HSV augmentation, horizontal flips,
  resizing, and padded training targets;
- `MosaicDetection` for mosaic, MixUp, translation, scaling, and shear;
- `Trainer` for optimization, learning-rate scheduling, mixed precision, EMA,
  checkpointing, and periodic evaluation; and
- `COCOEvaluator` for validation metrics.

The experiment file selects and configures those components. Keeping this
boundary lets future experiments change policy without forking the framework.

## First experiment contract

Implement `yolox_s_carparts23.py` by inheriting from `yolox.exp.Exp`.

### 1. Model and paths

In `Exp.__init__`, call `super().__init__()` and then define:

| Setting | Value |
| --- | --- |
| `depth` | `0.33` |
| `width` | `0.50` |
| `num_classes` | `23` |
| `data_dir` | repository root, derived from `Path(__file__)` |
| `train_ann` | `carparts23_instances_train.json` |
| `val_ann` | `carparts23_instances_val.json` |
| `test_ann` | `carparts23_instances_test.json` |
| `output_dir` | `<repository root>/outputs/yolox` |

Do not hard-code a user-specific absolute path. From this directory, the
repository root is `Path(__file__).resolve().parents[1]`.

### 2. Baseline preprocessing and augmentation

Begin with YOLOX-S defaults so the first run is an interpretable baseline:

| Setting | Initial value |
| --- | --- |
| `input_size`, `test_size` | `(640, 640)` |
| `hsv_prob` | `1.0` |
| `flip_prob` | `0.5` |
| `mosaic_prob` | `1.0` |
| `mixup_prob` | `1.0` |
| `degrees` | `10.0` |
| `translate` | `0.1` |
| `mosaic_scale` | `(0.1, 2.0)` |
| `mixup_scale` | `(0.5, 1.5)` |
| `shear` | `2.0` |

These values are inherited from the base experiment. Listing them here records
the baseline without needlessly repeating every default in code. Override a
value only when an experiment intentionally changes it.

### 3. Dataset construction

Override `get_dataset(cache=False, cache_type="ram")`. Construct and return a
YOLOX `COCODataset` using:

- `data_dir=self.data_dir`;
- `json_file=self.train_ann`;
- `name=""`;
- `img_size=self.input_size`;
- a `TrainTransform` using `self.flip_prob` and `self.hsv_prob`; and
- the received cache arguments.

The empty `name` is intentional. CRATER's COCO `file_name` values already
contain `images/train/...`; the stock loader otherwise prepends `train2017/` and
looks in the wrong location.

Override `get_eval_dataset(**kwargs)` in the same way, selecting `self.val_ann`
normally and `self.test_ann` when `testdev=True`. Use `name=""`,
`img_size=self.test_size`, and `ValTransform`. The inherited data loaders,
trainer, and evaluator can then remain unchanged.

## Preflight checks

Before a full run, verify the experiment in this order:

1. Import the experiment and instantiate `Exp`.
2. Confirm the training dataset length is `2585` and it reports 23 classes.
3. Fetch `dataset[0]` to exercise image loading, annotation conversion, and
   preprocessing together.
4. Run one short training smoke test before committing to the full schedule.

The first three checks should fail immediately if an annotation filename,
image path, category count, or transform contract is wrong.

## Environment and training command

Use a dedicated CUDA-enabled environment. Install the correct PyTorch build for
the training machine first, then install the pinned YOLOX submodule in editable
mode:

```powershell
python -m pip install -v -e third_party\YOLOX
```

Download the official pretrained YOLOX-S checkpoint to
`weights/yolox_s.pth`. After the experiment passes its preflight checks, run it
from the repository root:

```powershell
python third_party\YOLOX\tools\train.py `
  -f experiments\yolox_s_carparts23.py `
  -d 1 `
  -b 8 `
  --fp16 `
  -c weights\yolox_s.pth
```

Batch size is a machine-level choice, not experiment identity. Start at 8 and
increase it only after measuring GPU memory. Do not use `--cache` for the first
run; add RAM or disk caching after the uncached pipeline is verified.

Keep the test annotations out of model selection. Tune against validation data,
then evaluate the selected checkpoint on the test split once.
