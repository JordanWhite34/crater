# Third-Party Notices

## car-seg / Carparts-Seg data

- Title: `car-seg Dataset`
- Creator: Gianmarco Russo
- Published: November 2023
- Source: <https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm/dataset/4>
- Distribution used: <https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip>
- License: Creative Commons Attribution 4.0 International
- License text: <https://creativecommons.org/licenses/by/4.0/>

Modifications made for this package: invalid detector negatives with empty label
files were excluded; visually related variants were grouped and resplit;
validation/test variants were reduced to one per source group; YOLO segmentation
polygons were converted to COCO segmentation and bounding-box annotations; and a
derived three-class CRATER transfer view was added.

Suggested citation:

```bibtex
@misc{car-seg-un1pm_dataset,
  title        = {car-seg Dataset},
  type         = {Open Source Dataset},
  author       = {Gianmarco Russo},
  howpublished = {Roboflow Universe},
  url          = {https://universe.roboflow.com/gianmarco-russo-vt9xr/car-seg-un1pm},
  year         = {2023},
  month        = {nov}
}
```

No Ultralytics model weights or Python source code are included or required.

## YOLOX

- Project: `YOLOX`
- Copyright: Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
- Source: <https://github.com/Megvii-BaseDetection/YOLOX>
- Included as: Git submodule at `third_party/YOLOX`
- Pinned revision: `6ddff4824372906469a7fae2dc3206c7aa4bbaee`
- License: Apache License 2.0
- License text: `third_party/YOLOX/LICENSE`
- License URL: <https://www.apache.org/licenses/LICENSE-2.0>

YOLOX is included as the third-party detection framework used to configure,
train, evaluate, and export CRATER object-detection models. CRATER-specific
dataset preparation, experiment configuration, and trained model artifacts are
separate from the upstream YOLOX source. The YOLOX submodule is retained at the
revision listed above so training behavior can be reproduced against a known
upstream version.
