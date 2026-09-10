# Labels and splits

Use the six received Humvee detector labels without renaming existing data or
checkpoints. They differ from the broader proposed detector classes in
`configs/taxonomy.json`.

| Part | Damage model |
| --- | --- |
| `wheel_tire` | `mobility` |
| `windshield`, `door`, `engine_bay` | `structure` |
| `weapon_station`, `comms_equipment` | `mission_equipment` |

CVAT's `weapons_station` spelling is normalized to `weapon_station` on import;
the original label is retained in the source record.

| CVAT value | Damage label | Meaning |
| --- | --- | --- |
| `intact` | `no_visible_damage` | Observable component with no visible damage cue. |
| `degraded` | `possible_damage` | A plausible damage cue with ambiguous extent or cause. |
| `destroyed` | `severe_visible_damage` | Clear major deformation, missing material, destruction or burning. |
| `unknown` | `unobservable` | Insufficient visibility to judge the component. |

Draw tight boxes around visible parts. Check part identity, box fit and damage
labels, including CVAT's default `intact`. For synthetic images, check generation
artifacts; prompts are provenance, not ground-truth labels. `unobservable` is a
separate visual category, not a higher severity.

Review the proposed source groups in the damage notebook's review CSV and mark
each image `accept` or `reject`. Merge photos of the same scene or vehicle into
one group. Automatic Commons-ID proximity groups are proposals requiring review.

Split real source groups before making crops. Every crop and augmentation keeps
its source split. Synthetic crops are training-only. Select epochs on real
validation data, then evaluate once on the held-out real test set. Missing
classes are coverage gaps; high accuracy on intact-heavy data is insufficient.

Train damage classifiers from human boxes. Detector-produced crops introduce
localization and missed-detection errors and need a separate pipeline evaluation.
