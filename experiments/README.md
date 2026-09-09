# Experiments

Experiments are organized by prediction task, then by transfer stage.

```text
experiments/
  detection/
    yolox_s_carparts23.py       civilian 23-class pretraining
    yolox_s_crater6.py          military 6-class fine-tuning
  damage/
    README.md                   crop-classification contract
    part_damage_classification.ipynb  real training, then synthetic fine-tuning
    training.py                 ResNet18 training and checkpoint evaluation
```

An experiment file defines model identity and stable training policy. Machine
choices such as GPU count, batch size, and mixed precision stay on the command
line. Generated runs go under `outputs/<task>/<domain>/<run>` and never beside
source configurations.

Rules:

- create a new experiment file for a changed class set or meaningfully changed
  training policy;
- initialize military models from a recorded civilian checkpoint, never by
  silently overwriting it;
- select models using validation data and reserve test data for the final
  estimate;
- report per-class metrics, not only an aggregate score; and
- keep damage-classification metrics separate from detector metrics.
