# CRATER fresh setup

All source images are in `images/`. All labels and source metadata are in
`labels.json`. The Python files and `requirements.txt` are intentionally empty;
choose your own dependencies and write the pipeline from scratch.

| File | Intended use |
| --- | --- |
| `dataset.py` | Read images and labels; choose transforms and splits |
| `model.py` | Define the model |
| `train.py` | Train and save checkpoints |
| `evaluate.py` | Compute evaluation metrics |
| `predict.py` | Run inference |
| `requirements.txt` | Record dependencies |

## Label document

The document uses COCO-style `images`, `annotations`, and `categories`, plus
`sources`. Bounding boxes use `[x, y, width, height]` in image pixels.
Segmentation geometry is preserved where supplied. Image filenames use their
full SHA-256 hash to avoid filename collisions. Exact byte duplicates share one
image record; resized or recompressed copies remain separate images.

Each image has a `provenance` list with source IDs, original filenames and image
records, domain (`real`, `synthetic`, or `unknown`), and available manifest
metadata. Each annotation has `provenance` identifying its source and original
record. Resolve `source_id` against `sources` for the original label file, its
hash, taxonomy, licenses, and other source metadata. License IDs in original
records are scoped to their source, not to the combined document.

Categories are scoped to each source label file. Original spellings remain
unchanged, including `tire`, `wheel_tire`, and `weapons_station`. Decide your
own mapping before training. Existing split and grouping values are retained
as provenance; this setup assigns no new train/validation/test split.

The inventory includes civilian CarParts23 labels, the alternative derived
CRATER transfer taxonomy, real Humvee component labels, real and synthetic CVAT
damage labels, and all three splits of the new Military Vehicle Components
COCO export. Several label sources describe the same images. Select the source
or label version you intend to use rather than training on every annotation
indiscriminately. Damage annotations expose `damage_level` and CVAT attributes;
their original normalized CSV records and CVAT boxes are also retained.

The new export's real/synthetic domain is recorded as `unknown` because the
export does not establish that distinction. Generation intent is provenance,
not a human damage label. Review/group related images before assigning splits;
keep synthetic data out of validation and test. CVAT default `intact` values
do not prove that each box was explicitly reviewed.

Only image files actually present in the supplied source folders are included.
Old README inventories are not used to infer missing data. Model outputs,
generated crops, checkpoints, and the old application code are excluded.

## Git

This is the independent `fresh-setup` branch, with a new root commit. It lives
in a linked checkout so the original checkout and its uncommitted work remain
intact. Images are tracked with Git LFS. Install Git LFS before cloning or
checking out this branch on another computer, and run `git lfs pull` if image
files contain pointer text. No training code or dependencies are provided.
