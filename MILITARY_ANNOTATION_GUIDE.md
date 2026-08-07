# CRATER Military Component Annotation Guide

Use this guide for the first 200–300 low-altitude oblique/aerial military
vehicle crops. Bounding boxes are required; masks are optional.

## Component classes

| Class | Annotation rule |
| --- | --- |
| `windshield` | One tight box around the visible windshield assembly. Include split panes in one box when they form one assembly. |
| `wheel` | One tight box per visible wheel/tire assembly. Do not box a wheel that cannot be distinguished confidently. |
| `track` | One tight box per visible continuous track assembly. Left and right tracks are separate instances when both are visible. |
| `hull` | One box around the visible main body/crew compartment. Exclude a separable turret, mounted weapon, and long antenna elements. |
| `comms_system` | Box each identifiable external communications assembly, such as an antenna cluster or dish. Do not infer equipment hidden inside the hull. |
| `mounted_gun` | One box around the visible weapon assembly, including the receiver/mount and visible barrel. Exclude the turret body when separable. |

## Box policy

- Draw tight boxes around the **visible** extent; do not estimate hidden geometry.
- Label partially occluded components when their identity is still clear.
- Do not box an uncertain speck merely because the vehicle type usually has that
  component. Unobservable components become `unknown/not_observable` later in
  the BBN rather than false detector labels.
- Annotate every confidently visible instance in a crop. Missing an obvious
  component creates a false negative during detector training.
- Keep the part ontology separate from damage level. Damage labels belong to the
  later component-crop classification dataset.

## Required image metadata

Record the following with each image:

- source and license/approval status;
- vehicle family: `tracked_armored`, `wheeled_tactical`, or `logistics_truck`;
- view: `overhead`, `high_oblique`, `low_oblique`, or `ground_level`;
- source asset or sequence ID;
- vehicle or scene ID when known;
- synthetic versus real.

## Split policy

Split by source asset, video/flight, scene, and vehicle identity—not by individual
frame. Near-adjacent frames and synthetic renders of the same scene must stay in
one split. This prevents the same vehicle/background from inflating validation
and test performance.

For the first bridge set, aim for all three vehicle families and deliberately
include examples where each target component is both visible and not observable.

