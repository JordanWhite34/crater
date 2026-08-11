# Component damage-crop annotation guide

Damage labels describe visible evidence in a component crop. They do not claim
vehicle-level operational effects.

| Label | Rule |
| --- | --- |
| `no_visible_damage` | The component is sufficiently observable and no damage cue is visible. |
| `possible_damage` | A plausible damage cue is present, but extent or cause remains ambiguous. |
| `severe_visible_damage` | Clear major deformation, missing material, destruction, or active burning affects the component. |
| `unobservable` | Resolution, crop error, occlusion, smoke, or lighting prevents a defensible visual judgment. |

Annotators may record the nonexclusive cues `deformation`, `missing_material`,
`scorch`, `smoke`, `flame`, and `debris`. A cue supports the damage label but
does not replace it.

## Crop policy

- Retain `source_image_id`, component instance ID, detector class, crop box, and
  source/license metadata for every crop.
- Generate training crops from reviewed ground-truth boxes.
- Keep predicted detector crops in a separate end-to-end evaluation view.
- Include modest context around a component, but do not let the full vehicle or
  a watermark become a shortcut for the label.
- Review ambiguous, tiny, heavily occluded, or truncated crops explicitly.
- Never relabel an absent component as `unobservable`; without a crop there is
  no classifier example.

## Split policy

Assign the source image group to train, validation, or test before cropping.
Every crop and augmentation inherits that split. Multiple parts from the same
vehicle image may not cross split boundaries.
