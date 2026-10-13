# CRATER

This branch contains the CRATER image data layout and package scaffold.

## Raw data

- data/civilian/raw/images/ contains 2,970 images linked to civilian source records (source IDs 1-6).
- data/civilian/raw/labels/labels.json contains their 21,877 annotations, 78 categories, 6 source records, and provenance.
- data/military/raw/images/ contains 543 images linked to military source records (source IDs 7-13).
- data/military/raw/labels/labels.json contains their 6,447 annotations, 60 categories, 12 source records, and provenance.

The two label files together preserve all records from the former combined inventory: 3,513 images, 28,324 annotations, 138 categories, and 18 source records. Image paths are repository-relative. Annotation, image, category, and source IDs and provenance are retained. Category IDs are source-scoped; use source_id when selecting labels.

## Layout

- configs/ - model and data configuration
- data/{civilian,military}/raw/{images,labels}/ - source images and labels grouped by provenance
- data/{civilian,military}/{processed,splits}/ - prepared data and split definitions
- src/crater/{data,models,training,evaluation,inference}/ - package code
- scripts/ - command-line entry points
- 	ests/ - automated checks
- outputs/ - generated results and checkpoints

Empty directories are local scaffolding until project files are added.
