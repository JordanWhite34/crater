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

