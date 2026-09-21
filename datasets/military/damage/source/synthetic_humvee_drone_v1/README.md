# Synthetic Humvee drone-view expansion v1

Ten wide, high-oblique drone-view images generated to resemble the repository's
real non-synthetic Humvee photographs: field documentation, varied terrain and
weather, imperfect framing, and substantial environmental context.

The set emphasizes the weaker detector classes and rare damage combinations from
the held-out diagnostics. It is training-only and must remain excluded from
validation and test splits. Intended canonical labels are in `labels.csv`.

The labels are generation targets and still require human CVAT review. Generated
images do not provide reliable box coordinates, so this inventory intentionally
does not fabricate bounding boxes.
