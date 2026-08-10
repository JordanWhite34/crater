# CRATER Car-Parts Starter Dataset v1

This is a cleaned civilian pretraining dataset for CRATER's component detector.
It is ready for a COCO-compatible detector such as TorchVision Faster R-CNN or
Mask R-CNN and does **not** require the Ultralytics package.

## What is included

| Split | Images | Independent source groups | 23-class instances |
| --- | ---: | ---: | ---: |
| Train | 2,585 | 899 | 14,210 |
| Validation | 193 | 193 | 1,177 |
| Test | 192 | 192 | 1,150 |

The validation and test sets contain one representative per visually matched
source group. Training retains all usable variants for augmentation.

Two COCO annotation views are provided:

- `annotations/carparts23_instances_{split}.json`: the original 23 exterior-part
  classes. Use this for the primary civilian detector pretraining run.
- `annotations/crater_transfer3_instances_{split}.json`: a convenient transfer
  view containing `windshield`, `wheel`, and `hull_proxy`. The hull label is a
  union of civilian body panels and is explicitly **not** military hull ground
  truth.

The transfer view contains:

| Class | Train | Validation | Test | CRATER relevance |
| --- | ---: | ---: | ---: | --- |
| Windshield | 1,448 | 108 | 104 | Direct transfer from `front_glass` |
| Wheel | 612 | 76 | 78 | Direct transfer |
| Hull proxy | 2,536 | 190 | 188 | Weak civilian body-region proxy |

`track`, `comms_system`, and `mounted_gun` do not occur in this civilian data.

## Cleaning performed

- Verified all 3,833 image/label pairs and every segmentation polygon.
- Excluded 135 empty label files (3.52%); visual sampling showed vehicles and
  parts were present, so treating them as detector negatives would be harmful.
- Detected visually matched augmentation groups within reused filenames using
  rotation-tolerant normalized correlation.
- Found that 324 matched source groups crossed the published train/validation/
  test boundaries, affecting 1,403 images.
- Rebuilt deterministic 70/15/15 source-group splits with seed 42.
- Omitted 728 correlated validation/test variants and retained one representative
  per source group.
- Confirmed zero source-group overlap among the prepared splits.

The grouping is an automated conservative heuristic, not a manually verified
identity benchmark. See `audit/audit_report.json`,
`audit/source_assignments.csv`, and the audit images for the exact evidence and
decisions.

## Recommended use

1. Pretrain a Faster R-CNN/ResNet-50-FPN detector on the full 23-class COCO
   annotations.
2. Replace its predictor with the six CRATER classes: `windshield`, `wheel`,
   `track`, `hull`, `comms_system`, and `mounted_gun`.
3. Fine-tune and evaluate on a separately labeled, low-altitude oblique/aerial
   military bridge set.

Do not interpret performance on this civilian dataset as CRATER performance.
It does not validate overhead military imagery or three of the six target
components.

## Layout

```text
images/{train,val,test}/
annotations/
manifests/
audit/
utils/download_car_data.py
utils/prepare_carparts_seg.py
MILITARY_ANNOTATION_GUIDE.md
THIRD_PARTY_NOTICES.md
requirements.txt
```

All image paths in the COCO files are relative to this dataset root.

## Training

YOLOX training code is pinned as a Git submodule under `third_party/YOLOX`.
CRATER-owned experiment configurations belong in `experiments/`; pretrained
weights and generated runs belong in the ignored `weights/` and `outputs/`
directories. See [`experiments/README.md`](experiments/README.md) for the
23-class YOLOX-S baseline workflow and the planned transfer-learning boundary.

## Reproducing the preparation

Install the preparation dependencies and run the dataset downloader from the
repository root:

```powershell
python -m pip install -r requirements.txt
python utils\download_car_data.py
```

The downloader verifies the upstream archive SHA-256, safely extracts it,
recreates the deterministic cleaned splits, validates their expected counts,
and installs `images/`, `annotations/`, `manifests/`, and `audit/`. It refuses
to overwrite any existing dataset directory.
