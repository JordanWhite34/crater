# Military component annotation guide

Use this guide for low-altitude oblique/aerial military vehicle images. Bounding
boxes are required; masks are optional.

| Class | Annotation rule |
| --- | --- |
| `windshield` | One tight box around the visible windshield assembly. Include split panes together when they form one assembly. |
| `wheel` | One tight box per visible wheel/tire assembly. Do not infer indistinguishable wheels. |
| `track` | One tight box per visible continuous track assembly. Left and right tracks are separate when both are visible. |
| `hull` | Box the visible main body or crew compartment. Exclude a separable turret, mounted weapon, and long antenna elements. |
| `comms_system` | Box each identifiable external communications assembly, such as an antenna cluster or dish. Do not infer hidden equipment. |
| `mounted_gun` | Box the visible weapon assembly, including the receiver or mount and visible barrel. Exclude a separable turret body. |

## Box policy

- Draw tight boxes around visible extent; do not estimate hidden geometry.
- Label a partially occluded component only when its identity remains clear.
- Annotate every confidently visible instance.
- Do not encode damage in detector class names or boxes.
- A component that cannot be observed is handled downstream, not added as a
  false detector instance.

## Required metadata

- source and license or approval status;
- vehicle family: `tracked_armored`, `wheeled_tactical`, or `logistics_truck`;
- view: `overhead`, `high_oblique`, `low_oblique`, or `ground_level`;
- source asset, sequence, scene, and vehicle identifiers when known; and
- synthetic versus real.

## Split policy

Split by source asset, sequence, scene, and vehicle identity—not by individual
frame. Near-adjacent frames and synthetic variants of one scene stay together.
