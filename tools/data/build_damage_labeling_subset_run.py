"""Build the fixed Commons subset crop-labeling run used by the notebook."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from damage_crop_annotation import (
    create_annotation_run,
    discover_source_images,
    find_repo_root,
    load_coco_class_names,
    run_yolox_inference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="commons_damage_subset16_source6_v1")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = find_repo_root()
    subset_root = root / "datasets/military/damage/source/commons_labeling_subset_v1"
    run_dir = root / "datasets/military/damage/annotation_runs/predicted_crops" / args.run_name
    if run_dir.exists():
        print(f"Annotation run already exists; leaving it unchanged: {run_dir}")
        return

    image_paths = discover_source_images(
        subset_root / "images",
        subset_root / "selection_manifest.csv",
        local_file_field="subset_local_file",
    )
    class_annotation_file = root / "datasets/military/components/annotations/humvee_source6_instances_test.json"
    candidates, inference_metadata = run_yolox_inference(
        repo_root=root,
        image_paths=image_paths,
        experiment_file=root / "experiments/detection/yolox_s_crater6.py",
        checkpoint_file=root / "outputs/detection/humvee_source6/carparts_initialized_seed42/best_ckpt.pth",
        class_names=load_coco_class_names(class_annotation_file),
        confidence=args.confidence,
        device_name=args.device,
    )
    if not candidates:
        raise RuntimeError("The detector produced no crop candidates.")
    create_annotation_run(
        candidates=candidates,
        run_dir=run_dir,
        repo_root=root,
        inference_metadata=inference_metadata,
        crop_padding_fraction=0.08,
    )
    print(f"Created {run_dir}")
    print(f"Crops by head: {dict(Counter(item['part_group'] for item in candidates))}")
    print(f"Crops by detector class: {dict(Counter(item['detector_class'] for item in candidates))}")


if __name__ == "__main__":
    main()
