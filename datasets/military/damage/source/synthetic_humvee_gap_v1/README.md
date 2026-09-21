# Synthetic Humvee gap set v1

Six project-local images generated to add representation for the weakest detector
and damage combinations identified in `experiments/heldout_representation_diagnostics.ipynb`.

These are training-only candidates. They must remain excluded from validation and
test splits. The labels below are intended targets from the generation brief and
require visual QA before conversion into box-level CVAT annotations.

| file | component class | component group | damage level |
| --- | --- | --- | --- |
| `01_comms_possible.png` | `comms_equipment` | `mission_equipment` | `possible_damage` |
| `02_weapon_station_severe.png` | `weapon_station` | `mission_equipment` | `severe_visible_damage` |
| `03_windshield_severe.png` | `windshield` | `structure` | `severe_visible_damage` |
| `04_door_possible.png` | `door` | `structure` | `possible_damage` |
| `05_engine_bay_unobservable.png` | `engine_bay` | `structure` | `unobservable` |
| `06_wheel_tire_severe.png` | `wheel_tire` | `mobility` | `severe_visible_damage` |

The labels use the repository's canonical taxonomy. Because generative images do
not provide trustworthy object coordinates, no bounding boxes are asserted here;
annotate and review them in CVAT before importing into the canonical damage CSV.
