# Carparts23 Stage 1 detector report

Date: August 14, 2026

Status: complete; `carparts23-yolox-s-coco-v1` is promoted as the civilian
parent checkpoint for Stage 2.

## Technical summary

- Official YOLOX-S COCO initialization improved validation COCO AP50:95 from
  `0.422` to `0.546` and AP50 from `0.642` to `0.688` under the same 100-epoch
  schedule and leakage-resistant split.
- The selected checkpoint achieved held-out test AP50:95 `0.580`, AP50 `0.737`,
  AP75 `0.675`, and AR100 `0.779`. The test split was used only after selection.
- COCO initialization improved validation AP for 21 of 22 measurable classes.
  `back_right_door` decreased by 2.705 AP points; `object` was not measurable.
- The `object` class is a dataset/taxonomy defect: it has only 5 train and 2
  validation instances, no test instances, zero validation AP, and undefined
  test AP. It is not used by the civilian-to-military transfer mapping.
- Stage 1 is ready to hand off to military fine-tuning, with the `object` class
  retained only for checkpoint compatibility. A future Carparts23 dataset
  version should remove or explicitly relabel those seven instances.

## Aggregate results

| Model or split | AP50:95 | AP50 | AP75 | AR100 |
| --- | ---: | ---: | ---: | ---: |
| Scratch validation | 0.422 | 0.642 | — | — |
| COCO-initialized validation | 0.546 | 0.688 | 0.635 | 0.758 |
| Promoted checkpoint, held-out test | **0.580** | **0.737** | **0.675** | **0.779** |

The 0.124 AP50:95 validation gain is a 29.3% relative improvement over the
scratch baseline. Test metrics are final reporting metrics, not another model
selection signal. Their higher values do not prove the test population is
easier without a controlled split-composition analysis.

## Per-class findings

The best held-out classes were `front_glass` (91.568 AP), `hood` (85.339),
`front_door` (83.820), and `front_bumper` (82.388). The weakest measurable
classes were `front_right_light` (31.957), `wheel` (34.498),
`front_left_light` (36.531), `back_right_light` (39.017), and
`back_right_door` (39.541).

The largest validation gains from COCO initialization were `tailgate`
(+28.823 AP), `trunk` (+24.815), `hood` (+20.054), `front_door` (+18.338), and
`back_bumper` (+18.295). `back_right_door` was the only measurable regression
(-2.705 AP); its validation result is based on only 18 instances, so the
direction is worth monitoring during transfer but is not evidence of a broad
initialization failure.

Exact instance counts and all per-class results are preserved in
[`carparts23-stage1-data.csv`](carparts23-stage1-data.csv).

## Scope, data, and metric definitions

- Dataset: deterministic Carparts-Seg v4 preparation, archive SHA-256
  `e2eea20030d02366b07174bcdbcc843f63791bdd53962a83e00636d15e78e41f`.
- Split policy: source-grouped 70/15/15 preparation with seed `42`, preventing
  known augmented variants from crossing splits.
- Split sizes: 2,585 train images / 14,210 boxes; 193 validation images / 1,177
  boxes; 192 test images / 1,150 boxes.
- Model: YOLOX-S, 23-class head, 640-pixel evaluation size, 8.95M parameters.
- Selection metric: COCO bounding-box AP averaged over IoU 0.50:0.95 on the
  validation split. Test AP is averaged over the 22 classes with ground-truth
  test instances; COCOeval excludes the unmeasurable `object` category.

## Methodology and robustness checks

Both training runs used the same prepared split, 100 epochs, batch size 8,
mixed precision, 5 warmup epochs, 15 no-augmentation epochs, and evaluation
every 5 epochs. The selected run started from official YOLOX-S COCO weights;
incompatible 80-class prediction tensors were skipped while compatible model
weights were loaded.

The promoted `best_ckpt.pth` records best validation AP50:95
`0.5457462741069473` at epoch 100. The held-out test evaluation used the same
experiment configuration and the repository's standard pycocotools evaluator
with `--test`; no test-driven model or threshold changes were made.

Data-quality checks reconciled image and box counts against the downloader's
expected values, verified the source archive hash from the generated audit
report, and confirmed the split-specific annotation hashes recorded in the
promotion manifest. The checkpoint copy was verified byte-for-byte by SHA-256.

## Limitations and uncertainty

- This result measures civilian car-part detection and does not establish
  performance on military vehicles, aerial viewpoints, damage, or operational
  effects.
- `object` is statistically unmeasurable and semantically ambiguous. Removing
  it now would make the promoted 23-class head incompatible, so remediation is
  deferred to a versioned dataset/model change.
- Several classes have small validation/test samples, especially `tailgate`
  (9/9), `trunk` (14/15), and side-specific doors/lights. Per-class rankings
  for those categories are less stable than aggregate AP.
- Inference timing includes a slow first batch and Windows evaluation overhead;
  it is not an edge-deployment benchmark.

## Promoted checkpoint and lineage

- Version: `carparts23-yolox-s-coco-v1`
- Local checkpoint:
  `outputs/detection/civilian/promoted/carparts23-yolox-s-coco-v1/carparts23-yolox-s-coco-v1.pth`
- Checkpoint SHA-256:
  `5ab8c99a5a67ca253aea6f1d4a6974f827b5e5bd16aca227306c129a7fd94d47`
- Starting COCO checkpoint SHA-256:
  `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30`
- CRATER source commit: `f2e18633e3c4640cbc5f98056aead5bdd940e030`
- YOLOX source commit: `6ddff4824372906469a7fae2dc3206c7aa4bbaee`
- Evaluation log:
  `outputs/detection/civilian/yolox_s_carparts23_stage1_test/val_log.txt`
- Evaluation-log SHA-256:
  `eba89d5565ee823c261fadd870ef564045bebdd8d934b2724ec31a06cc55f377`

The checkpoint and raw logs remain ignored local artifacts; their hashes and
lineage are tracked here and in the local `promotion.json` beside the promoted
checkpoint.

## Recommended next steps

1. Begin Stage 2 data collection and annotation using the six military classes.
2. Initialize `yolox_s_crater6.py` from the promoted checkpoint and record this
   version/hash as the parent of every military run.
3. Preserve class- and viewpoint-level validation reporting during military
   fine-tuning, with special attention to `wheel` transfer.
4. Create a versioned Carparts23-v2 proposal that removes or relabels `object`;
   do not silently alter the current checkpoint taxonomy.

## Further questions

- Do the weak light/mirror/wheel classes reflect object size, occlusion, label
  consistency, or visual ambiguity?
- Does the civilian checkpoint improve six-class military convergence and AP
  versus official COCO initialization and scratch?
- Is a 22-class civilian retrain worthwhile after Stage 2 establishes which
  source classes actually contribute to military transfer?
