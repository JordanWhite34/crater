# Humvee source-six initialization comparison

Date: August 24, 2026

Status: exploratory comparison complete; no military checkpoint promoted.

## Answer

Initializing YOLOX-S from the promoted civilian Carparts23 checkpoint produced
substantially better Humvee validation performance than training from random
weights under the same schedule. Best validation COCO AP50:95 increased from
`0.1297` to `0.4784`, an absolute gain of `0.3487`.

This supports the staged transfer design, but it does not establish
release-quality military performance. The dataset is small and Humvee-only,
its source taxonomy differs from canonical CRATER, and authoritative provenance
and grouping metadata remain incomplete.

## Data and split

- Source export: 224 images, including 222 JPEG and 2 PNG files.
- Annotations: 2,226 COCO bounding boxes; 3 images have no boxes.
- Received classes: `wheel_tire`, `windshield`, `door`, `engine_bay`,
  `weapon_station`, and `comms_equipment`.
- Split seed: `42`.
- Interim grouping: 116 Commons upload/photo-series groups; largest group 18
  images; zero groups crossed splits.

| Split | Images | Boxes |
| --- | ---: | ---: |
| Train | 156 | 1,598 |
| Validation | 34 | 322 |
| Test | 34 | 306 |

The grouping combines numeric Commons upload sequences with three manually
linked visual series. It is more leakage-resistant than an image-random split,
but it is not a substitute for verified source, scene, and vehicle identity.

## Controlled experiment

Both YOLOX-S runs used the same generated annotations, seed `42`, batch size 8,
mixed precision, 100 epochs, 5 warmup epochs, 15 no-augmentation epochs, and
evaluation every 5 epochs. Training ran with PyTorch `2.11.0+cu128` on an NVIDIA
RTX 5000 Ada Generation.

The initialized run used civilian parent `carparts23-yolox-s-coco-v1`, verified
before training against SHA-256
`5ab8c99a5a67ca253aea6f1d4a6974f827b5e5bd16aca227306c129a7fd94d47`.
YOLOX loaded shape-compatible weights and replaced incompatible 23-class
prediction tensors with the six-output head.

## Validation results

| Run | Initialization | Best epoch | AP50:95 | Training time |
| --- | --- | ---: | ---: | ---: |
| `scratch_seed42` | Random | 92 | 0.1297 | 31.9 min |
| `carparts_initialized_seed42` | Carparts23 civilian checkpoint | 91 | **0.4784** | 33.6 min |

The initialized run was selected using validation AP50:95. After selection, it
produced 13, 7, and 9 detections at confidence `0.25` on three test-split images.
That was a qualitative inference smoke test only; it is not a full held-out
COCO evaluation.

## Interpretation boundary

- The result is evidence that civilian component pretraining transfers useful
  localization features to this small Humvee dataset.
- It is not evidence of performance on other military vehicle families,
  canonical CRATER classes, damage severity, or operational effects.
- The six received labels are not interchangeable with canonical CRATER. In
  particular, `door` and `engine_bay` must not be silently mapped to `hull`, and
  this Humvee dataset contains no `track` class.
- This is a one-seed comparison. It demonstrates a large observed difference,
  but it does not quantify run-to-run uncertainty.
- Three held-out images have been visually inspected. Any later aggregate test
  evaluation must be fixed in advance and reported without test-driven tuning.

## Artifact status

The executed notebook is the versioned experiment record. Generated annotations,
logs, TensorBoard data, and checkpoints were written under
`outputs/detection/humvee_source6/` and remain ignored local artifacts. The
selected military checkpoint is not promoted and its SHA-256 is not yet recorded
in the repository. A notebook output does not replace the binary or its lineage.

## Promotion gates

1. Restore or reproduce `carparts_initialized_seed42/best_ckpt.pth`; record its
   hash, parent, source commit, split hashes, and training configuration.
2. Complete authoritative URL, license, source, scene, vehicle, and
   near-duplicate metadata; review the three images without boxes.
3. Freeze the evaluation protocol and run one aggregate held-out COCO test with
   per-class and error-stratified results.
4. Keep this received-taxonomy model exploratory. Build the canonical CRATER
   six-class dataset as a separate, explicitly versioned annotation lineage.
5. Promote a military checkpoint only after the applicable data, evaluation,
   and reproducibility gates are satisfied.
