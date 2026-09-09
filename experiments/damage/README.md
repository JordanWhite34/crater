# Damage-classification experiments

## Run on another computer

Install Git LFS and Python, then clone this branch and retrieve the labeled
images. The damage notebook does not require the old civilian detector
checkpoints under `runs/`; it trains its own models from ImageNet initialization.

From PowerShell on the other computer:

```powershell
git lfs install
git clone --branch feature/humvee-training-comparison https://github.com/JordanWhite34/crater.git
cd crater
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/damage/requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name crater --display-name "CRATER"
.\.venv\Scripts\python.exe -m jupyter lab experiments/damage/part_damage_classification.ipynb
```

Select the **CRATER** kernel. The requirements include pandas, which is needed
before the first code cell can run. For NVIDIA GPU training, install the matching
torch/torchvision build using the [PyTorch installer](https://pytorch.org/get-started/locally/)
inside this environment before installing the remaining requirements. The
notebook prints whether it is using CUDA or CPU. ImageNet initialization needs
internet access on its first training run to download the weights.

The notebook recreates the pending review CSV on a fresh clone. If you have
already completed review on another machine, copy
`datasets/military/damage/review/humvee_real_synthetic_v1.csv` to the same relative
path before running the review cells; that working file is ignored by Git.
Generated crops and trained checkpoints are also local artifacts.

## Real data followed by synthetic fine-tuning

Open [part_damage_classification.ipynb](part_damage_classification.ipynb) to
prepare the human CVAT part crops, train a real-image baseline for each component
group, then continue training from those checkpoints on the synthetic crops.
The synthetic stage uses a lower learning rate. Both stages select epochs using
the same real validation split; a final paired evaluation uses real test images
only. The original checkpoint is preserved, and a synthetic stage that does not
improve validation retains the original model as its epoch-zero best checkpoint.

The notebook uses ImageNet-initialized ResNet18 models with the four canonical
damage levels. This is an exploratory direct military baseline; it does not
claim the planned civilian-damage pretraining has happened. Training code is in
`training.py`; source verification, grouped splitting and crop preparation are
in `tools/data/prepare_damage_classification.py`.

The desktop synthetic folder was checked on September 9, 2026: all 30 PNGs match
the repository images byte-for-byte, and `annotations.xml` matches the recorded
original-export hash. The existing import contains 251 human-labeled boxes, so
no duplicate dataset was copied.

Install `experiments/damage/requirements.txt` in the chosen notebook environment
and follow Setup in the notebook. Its first run creates
`datasets/military/damage/review/humvee_real_synthetic_v1.csv`. Review the proposed
photo groups and annotations, then mark each source image `accept` or `reject`.
The groups initially reuse the detector pilot's interim photo-series heuristic;
they must be checked against scene/vehicle identity before acceptance. The
review covers the 222 real images with damage boxes and 30 synthetic images;
the original real source inventory contains 224 images including two without
damage boxes. Pending review rows stop crop preparation. Synthetic images never
enter validation or test, and all real group derivatives share one split.

Generated artifacts go under `outputs/damage/humvee_cvat/<RUN_NAME>/`, including
the review snapshot, hashed crop manifest, class-coverage table, both stages'
best/last checkpoints, training histories, selected-checkpoint hashes and final
test predictions/metrics. Existing training runs refuse overwrites. Keep a new
run name for changed data or settings. Mission-equipment damage coverage is
especially sparse; a head with fewer than two observed training classes must be
deferred or supplied more reviewed data.

Validation at creation: all 13 classifier and related annotation tests passed,
as did Jupyter's notebook-format validator. Every notebook cell executed on
isolated fixture data, including all three heads, both training stages and the
paired evaluation. Production inventory and visual-review cells also executed;
preparation correctly stopped at the 252 pending review rows. Fixture outputs
are not model-performance evidence. The production notebook has not trained
models: source-group and semantic-label review remains pending. After completing
that review, execute the notebook in order, or run it top-to-bottom from the
repository root in the configured environment:

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=-1 experiments/damage/part_damage_classification.ipynb
```

Do not rerun a completed training directory; choose a new `RUN_NAME` for a new
experiment. To inspect existing results, load the saved checkpoints and JSON/CSV
artifacts rather than repeating training or using test results for selection.

## Planned civilian-to-military lineage

Damage classification begins only after a labeled component-crop dataset
exists. Keep it separate from detection so detector AP and damage accuracy
remain independently interpretable.

The intended lineage is:

1. train crop classifiers on civilian damaged/undamaged parts;
2. fine-tune those checkpoints on military component crops; and
3. evaluate both crop-level performance and the end-to-end detector-to-crop
   pipeline.

Use the canonical levels and component groups in `configs/taxonomy.json`.
Initial models should use three broad groups: `mobility`, `structure`, and
`mission_equipment`. The civilian data has no direct mission-equipment analogue,
so that head starts with military data rather than pretending transfer exists.

Each `labels.csv` should minimally contain:

```text
crop_id,source_image_id,component_instance_id,component_class,component_group,damage_level,source_domain,split
```

Generate training crops from ground-truth boxes. Predicted detector crops belong
in a separate end-to-end evaluation set because they include localization and
missed-detection error.

## Human ground-truth box source

`datasets/military/damage/source/humvee_cvat_damage_v1/damage_box_annotations.csv`
contains 2,240 validated human boxes across the 224 Humvee source images. Use
this inventory as the ground-truth crop source after authoritative image groups
and splits are assigned. Its `split` field is intentionally blank to prevent an
accidental image-random training split.

The importer retains CVAT's `intact`, `degraded`, `destroyed`, and `unknown`
values in `damage_state_source` and maps them to the canonical four-level CRATER
taxonomy in `damage_level`. It also normalizes CVAT's `weapons_station` spelling
to the established source class `weapon_station` while preserving both fields.

This source is strongly imbalanced toward intact components and contains no
damaged communications-equipment examples. Establish the split and class
coverage plan before selecting a classifier or sampling policy.

## Synthetic ground-truth box source

`datasets/military/damage/source/synthetic_humvee_damage_v1/` contains 30
generated full images and 251 human CVAT component boxes. It uses the same
box-CSV columns and canonical damage mappings as the real Humvee source, while
`source_manifest.csv` carries `domain=synthetic`, prompt lineage, and one
generation group per independently generated image.

Use this source only as optional training augmentation after visual QA. Do not
include it in validation or test partitions, and do not train from
`intended_condition`; that field records generation intent, while
`damage_level` records the human annotation. The source remains imbalanced and
has only one degraded communications-equipment box and no destroyed examples.

## Detector-driven crop labeling notebook

`military_damage_crop_labeling.ipynb` creates a deliberately separate
annotation run from predicted Humvee detector boxes. It loads the selected
fine-tuned checkpoint, maps its received six classes into `mobility`,
`structure`, and `mission_equipment`, assigns stable crop IDs, and displays all
crops on paginated contact sheets.

By default the notebook uses the fixed 16-image `commons_labeling_subset_v1`
pilot: 5 intact, 7 moderate-damage, and 4 severe-damage source images selected
to cover wheeled mobility, tracked mobility, structure, communications, and
mounted weapons. The exact Commons pages, licenses, hashes, and selection
rationale are preserved in the subset manifest.

The pilot has been executed through detector inference, crop export, and review
sheet generation. It produced three mobility pages, two structure pages, and
one mission-equipment page. The human label uploads and final validation/split
cell remain unexecuted, so there is no validated damage dataset or trained
damage classifier yet. Generated annotation-run assets are ignored and must be
restored or reproduced when continuing the pilot.

Each run is written under
`datasets/military/damage/annotation_runs/predicted_crops/<run_name>/` and
contains clean crops, a provenance manifest, contact sheets, and one blank
two-column label template per part group. Completed uploads use the requested
numeric rubric: `1` none, `2` moderate, and `3` severe. Validation requires
valid IDs and values, but blank or omitted rows are allowed. It writes only
nonblank labels to `validated_damage_labels.csv`; unlabeled crops and their
images are excluded from the prepared datasets. The final notebook cell assigns
source images—not individual crops—to deterministic train/validation/test
splits and creates a separate image/label tree for each head. This does not
change the older canonical `possible_damage` taxonomy implicitly.

These predicted crops must remain identifiable as detector-derived data. They
are useful for human damage labeling and later pipeline experiments, but they
do not become ground-truth component boxes merely because a human labels their
damage severity.
