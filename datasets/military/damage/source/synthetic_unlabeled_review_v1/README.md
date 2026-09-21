# Reviewed labels for synthetic unlabeled Humvee set v1

This is the reviewed annotation companion for
`datasets/military/humvee_unlabeled_v1/images`.

- 36 images reviewed
- 36 dominant-component boxes
- canonical labels in `damage_box_annotations.csv`
- CVAT 1.1 annotations in `annotations.xml`
- source hashes and dimensions in `source_manifest.csv`

The review intentionally records one dominant sampled component per image. It
does not claim exhaustive annotation of every visible wheel, door, windshield,
or other component in the scene. `possible_damage` is used for visually
ambiguous but inspectable damage, and `unobservable` is used when the requested
component cannot be reliably inspected. All splits remain blank pending dataset
partitioning; synthetic data must stay out of validation and test sets.
