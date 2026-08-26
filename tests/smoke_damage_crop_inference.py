"""One-image smoke test for the real Humvee checkpoint and YOLOX environment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data.damage_crop_annotation import (
    discover_source_images,
    find_repo_root,
    load_coco_class_names,
    run_yolox_inference,
)


def main() -> None:
    root = find_repo_root()
    source_root = root / "datasets/military/damage/source/wikimedia_commons_candidates"
    images = discover_source_images(source_root / "images", source_root / "curated_manifest.csv")[:1]
    annotations = root / "datasets/military/components/annotations/humvee_source6_instances_test.json"
    candidates, metadata = run_yolox_inference(
        repo_root=root,
        image_paths=images,
        experiment_file=root / "experiments/detection/yolox_s_crater6.py",
        checkpoint_file=root / "outputs/detection/humvee_source6/carparts_initialized_seed42/best_ckpt.pth",
        class_names=load_coco_class_names(annotations),
        confidence=0.25,
        device_name="cpu",
    )
    print(f"smoke_image={images[0].name}")
    print(f"detections={len(candidates)}")
    print(f"checkpoint_sha256={metadata['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
