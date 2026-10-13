# CRATER

Part detection with YOLOX-S, followed by visible-damage classification with ResNet18.

## Run

Use one Python environment for both notebooks. Install a matching PyTorch and
torchvision build for your GPU, then install the remaining dependencies:

```powershell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name crater --display-name "CRATER"
python -m jupyter lab
```

Select that environment's CRATER kernel. For a fresh environment, first run
`python -m venv .venv` and use `.venv\Scripts\python.exe` in place of
`python`. On a fresh clone, run `git submodule update --init --recursive` and
`git lfs pull` to retrieve YOLOX and the versioned source images. Install the
detector framework in the same environment with:

```powershell
python -m pip install -e third_party/YOLOX
```

| Task | Notebook |
| --- | --- |
| Show part boxes and predicted damage on a few images | [Quick demo](experiments/demo.ipynb) |
| Prepare splits, train the part detector, inspect detections | [Part detection](experiments/detection/humvee_initialization_comparison.ipynb) |
| Review part boxes, prepare crops, train and evaluate damage classifiers | [Damage assessment](experiments/damage/part_damage_classification.ipynb) |

Run notebook cells in order. The detector notebook compares scratch training with
Carparts initialization; its last cell can independently inspect an existing
detector. Training cells launch training, so skip them when inspecting saved
detector results. The damage notebook validates and reuses completed stages.
Use a new damage `RUN_NAME` when changing data or settings.

Notebook outputs are cleared from the working copies to keep them small.
Checkpoints, metrics and predictions live in `outputs/`. Existing executed
notebooks and removed legacy files are backed up in
`outputs/repo_cleanup/before_simplification_20260909_122427.zip` on this machine.

## Models and data

The detector retains the six labels from your Humvee export:
`wheel_tire`, `windshield`, `door`, `engine_bay`, `weapon_station`,
and `comms_equipment`. Class order comes from the COCO annotation file.

Damage models use three groups: mobility, structure, and mission equipment.
They start from ImageNet, train on reviewed real part crops, then fine-tune on
synthetic crops. Both stages select checkpoints using real validation data.
If synthetic training does not improve validation, its best checkpoint remains
the real baseline at epoch 0; `last.pt` contains the final adapted model.

See [labeling rules](docs/labeling.md) for the four damage labels and grouping rules.

| Input or result | Location |
| --- | --- |
| Real images and detector annotations | `datasets/military/components/source/humvee/` |
| Human damage boxes | `datasets/military/damage/source/humvee_cvat_damage_v1/` |
| Synthetic images and human damage boxes | `datasets/military/damage/source/synthetic_humvee_damage_v1/` |
| Editable image/group review | `datasets/military/damage/review/humvee_real_synthetic_v1.csv` |
| Selected part detector | `outputs/detection/humvee_source6/carparts_initialized_seed42/best_ckpt.pth` |
| Damage run | `outputs/damage/humvee_cvat/humvee_damage_real_then_synthetic_v1/` |

Completed detection and damage runs exist locally. The damage run selected the
real baseline for all three groups; synthetic fine-tuning did not improve
validation. Read `validation_comparison.csv`, `validation_history.csv`,
`selected_checkpoints.json`, and `test_final/comparison.json` in the damage run
for results. Damage-class coverage is sparse, particularly for mission equipment.

The damage evaluation uses human boxes. Combined detector-to-damage performance
has not yet been evaluated.

## Supporting code

- `experiments/detection/`: two YOLOX configurations and the Windows-compatible COCO evaluator.
- `experiments/damage/training.py`: damage training, checkpoint recovery and evaluation.
- `tools/data/`: Carparts download/preparation, real/synthetic CVAT import, damage crop preparation.
- `configs/taxonomy.json`: existing taxonomy and damage label contract.
- `tests/`: annotation import and classifier checks.
- `third_party/YOLOX/`: pinned detector implementation; see [third-party notices](THIRD_PARTY_NOTICES.md).

The civilian detector's promoted weights are the starting point for Humvee
fine-tuning. To reproduce that parent, prepare Carparts23 and train:

```powershell
python tools/data/download_carparts23.py
python third_party/YOLOX/tools/train.py -f experiments/detection/yolox_s_carparts23.py -expn yolox_s_carparts23_pretrained -d 1 -b 8 --fp16 -c weights/yolox_s.pth max_epoch 100 warmup_epochs 5 no_aug_epochs 15 eval_interval 5
```

The Humvee notebook expects the promoted parent at
`outputs/detection/civilian/promoted/carparts23-yolox-s-coco-v1/carparts23-yolox-s-coco-v1.pth`
and checks its hash. Restore that artifact when moving to another computer.

To import new annotations, use the CVAT importers in `tools/data/`; each exposes
`--help`. Existing source inventories are already imported. Retain their source
manifests and XML exports. Review CSVs, generated crops and model weights are
local files and need to be copied separately between machines.

## Check

```powershell
python -m pytest tests
```

Keep related images in the same split, synthetic images in training only, and
test data out of model selection. Damage describes visible condition; it does
not establish vehicle operability.
