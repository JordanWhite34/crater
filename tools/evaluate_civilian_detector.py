#!/usr/bin/env python3
"""Evaluate a civilian detector checkpoint with validation/test safeguards."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from train_civilian_detector import (
    ANNOTATION_FILE,
    CHECKPOINT_SCHEMA_VERSION,
    EXPECTED_CATEGORY_COUNT,
    MODEL_NAME,
    TRAINABLE_BACKBONE_LAYERS,
    CivilianPartsDataset,
    build_model,
    category_names,
    detection_collate,
    nonnegative_int,
    positive_int,
    resolve_device,
    sha256,
)


VALIDATION_ANNOTATION_FILE = Path("annotations/carparts23_instances_val.json")
TEST_ANNOTATION_FILE = Path("annotations/carparts23_instances_test.json")
EXPECTED_TEST_ANNOTATION_SHA256 = (
    "2187072b83f9ab5fa14825ef9e99ffbe6cea6945c5e1c1570337de8358045a71"
)
EXPECTED_TEST_IMAGES = 192


def sha256_argument(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("must be a 64-character SHA-256 digest")
    return normalized


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to evaluate. Required for a final-test run.",
    )
    parser.add_argument("--data-root", type=Path, default=project_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--final-test",
        type=sha256_argument,
        metavar="EXPECTED_CHECKPOINT_SHA256",
        help=(
            "Run the hard-coded held-out test split and require this frozen "
            "checkpoint digest. Omit for validation-only evaluation."
        ),
    )
    args = parser.parse_args()
    if args.final_test is not None:
        if args.checkpoint is None:
            parser.error("--final-test requires an explicit --checkpoint")
        if args.output_dir is None:
            parser.error("--final-test requires an explicit new --output-dir")
        if args.checkpoint.name.lower() != "checkpoint_best.pt":
            parser.error("--final-test requires the frozen checkpoint_best.pt")
    elif args.checkpoint is None:
        args.checkpoint = (
            project_root
            / "runs"
            / "civilian_detector_baseline"
            / "checkpoint_last.pt"
        )
    return args


def coco_predictions(
    model: torch.nn.Module,
    dataset: CivilianPartsDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
    )
    predictions: list[dict[str, Any]] = []
    processed_image_ids: list[int] = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = [
                image.to(device, non_blocking=device.type == "cuda")
                for image in images
            ]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                processed_image_ids.append(image_id)
                boxes = output["boxes"].detach().cpu()
                labels = output["labels"].detach().cpu()
                scores = output["scores"].detach().cpu()
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box.tolist()
                    category_id = int(label.item())
                    score_value = float(score.item())
                    values = (x1, y1, x2, y2, score_value)
                    if not all(math.isfinite(value) for value in values):
                        raise RuntimeError(f"Non-finite prediction for image {image_id}")
                    if x2 <= x1 or y2 <= y1:
                        raise RuntimeError(f"Invalid prediction box for image {image_id}")
                    if not 1 <= category_id <= EXPECTED_CATEGORY_COUNT:
                        raise RuntimeError(
                            f"Invalid predicted category {category_id} for image {image_id}"
                        )
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score_value,
                        }
                    )
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"{split_name} batch={batch_index}/{len(loader)}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if processed_image_ids != dataset.ids:
        raise RuntimeError(
            f"{split_name.title()} loader did not process every image exactly once"
        )
    elapsed_seconds = time.perf_counter() - started_at
    return predictions, {
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": len(dataset) / elapsed_seconds,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
    }


def mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def write_json_atomic(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    serialization_options = (
        {"separators": (",", ":")} if compact else {"indent": 2}
    )
    temporary_path.write_text(
        json.dumps(value, allow_nan=False, **serialization_options) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def evaluate_predictions(
    dataset: CivilianPartsDataset,
    predictions: list[dict[str, Any]],
    split_name: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not predictions:
        raise RuntimeError(f"The checkpoint produced no {split_name} detections")
    detections = dataset.coco.loadRes(predictions)
    evaluator = COCOeval(dataset.coco, detections, iouType="bbox")
    evaluator.params.imgIds = sorted(dataset.ids)
    evaluator.params.catIds = sorted(category["id"] for category in dataset.coco.dataset["categories"])
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    overall_names = (
        "ap",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar1",
        "ar10",
        "ar100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    overall = {
        name: float(value) for name, value in zip(overall_names, evaluator.stats)
    }

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    area_index = evaluator.params.areaRngLbl.index("all")
    max_detections_index = evaluator.params.maxDets.index(100)
    iou50_index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    sliced_overall = {
        "ap": mean_valid(precision[:, :, :, area_index, max_detections_index]),
        "ap50": mean_valid(
            precision[iou50_index, :, :, area_index, max_detections_index]
        ),
        "ar100": mean_valid(recall[:, :, area_index, max_detections_index]),
    }
    for name, value in sliced_overall.items():
        if value is None or not np.isclose(value, overall[name]):
            raise RuntimeError(f"COCO metric slicing mismatch for {name}")
    per_class = []
    prediction_counts = {category_id: 0 for category_id in evaluator.params.catIds}
    for prediction in predictions:
        prediction_counts[prediction["category_id"]] += 1
    categories = {
        category["id"]: category["name"]
        for category in dataset.coco.dataset["categories"]
    }
    for category_index, raw_category_id in enumerate(evaluator.params.catIds):
        category_id = int(raw_category_id)
        annotation_ids = dataset.coco.getAnnIds(
            imgIds=dataset.ids, catIds=[category_id]
        )
        annotations = dataset.coco.loadAnns(annotation_ids)
        per_class.append(
            {
                "category_id": category_id,
                "category_name": categories[category_id],
                "instances": len(annotations),
                "images": len({annotation["image_id"] for annotation in annotations}),
                "supported": bool(annotations),
                "low_support": len(annotations) < 10,
                "predictions": prediction_counts[category_id],
                "coco_ap": mean_valid(
                    precision[
                        :, :, category_index, area_index, max_detections_index
                    ]
                ),
                "coco_ap50": mean_valid(
                    precision[
                        iou50_index,
                        :,
                        category_index,
                        area_index,
                        max_detections_index,
                    ]
                ),
                "coco_ar100": mean_valid(
                    recall[:, category_index, area_index, max_detections_index]
                ),
            }
        )
    return overall, per_class


def main() -> None:
    args = parse_args()
    final_test = args.final_test is not None
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = sha256(checkpoint_path)
    if final_test and checkpoint_sha256 != args.final_test:
        raise ValueError(
            "Frozen checkpoint SHA-256 mismatch: "
            f"expected {args.final_test}, found {checkpoint_sha256}"
        )
    if final_test:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            raise FileExistsError(
                f"Final-test output directory must not already exist: {output_dir}"
            )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if checkpoint.get("completed_epochs", 0) < 1:
        raise ValueError("Checkpoint has no completed training epoch")
    if len(checkpoint.get("history", [])) != checkpoint["completed_epochs"]:
        raise ValueError("Checkpoint history does not match its completed epochs")

    contract = checkpoint["contract"]
    data_root = args.data_root.resolve()
    expected_contract = {
        "model": MODEL_NAME,
        "initial_weights": "coco",
        "num_classes_including_background": EXPECTED_CATEGORY_COUNT + 1,
        "trainable_backbone_layers": TRAINABLE_BACKBONE_LAYERS,
        "torch_version": str(torch.__version__),
    }
    for name, expected in expected_contract.items():
        if contract.get(name, expected if name == "trainable_backbone_layers" else None) != expected:
            raise ValueError(f"Checkpoint contract mismatch: {name}")
    train_annotation_path = data_root / ANNOTATION_FILE
    if sha256(train_annotation_path) != contract["annotation_sha256"]:
        raise ValueError("Training annotations differ from the checkpoint contract")

    split_name = "test" if final_test else "validation"
    annotation_path = data_root / (
        TEST_ANNOTATION_FILE if final_test else VALIDATION_ANNOTATION_FILE
    )
    annotation_sha256 = sha256(annotation_path)
    if final_test and annotation_sha256 != EXPECTED_TEST_ANNOTATION_SHA256:
        raise ValueError(
            "Held-out test annotations differ from the frozen evaluation contract"
        )
    dataset = CivilianPartsDataset(root=data_root, annFile=annotation_path)
    if final_test and len(dataset) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_IMAGES} held-out test images, found {len(dataset)}"
        )
    names = category_names(dataset)
    if names != contract["categories"]:
        raise ValueError(
            f"{split_name.title()} taxonomy differs from the checkpoint taxonomy"
        )

    device = resolve_device(args.device)
    model = build_model(
        contract["num_classes_including_background"],
        weights_name="none",
        image_size=contract["image_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    completed_epochs = checkpoint["completed_epochs"]
    global_step = checkpoint["global_step"]
    del checkpoint
    model.to(device)
    inference_settings = {
        "precision": "float32",
        "score_threshold": model.roi_heads.score_thresh,
        "nms_threshold": model.roi_heads.nms_thresh,
        "detections_per_image": model.roi_heads.detections_per_img,
    }
    predictions, inference = coco_predictions(
        model,
        dataset,
        device,
        args.batch_size,
        args.num_workers,
        split_name,
    )
    overall, per_class = evaluate_predictions(dataset, predictions, split_name)
    if sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("Checkpoint changed during evaluation")

    unsupported_categories = [
        item["category_name"] for item in per_class if not item["supported"]
    ]
    if final_test:
        object_metrics = next(
            item for item in per_class if item["category_name"] == "object"
        )
        if unsupported_categories != ["object"] or any(
            object_metrics[name] is not None
            for name in ("coco_ap", "coco_ap50", "coco_ar100")
        ):
            raise RuntimeError(
                "The zero-instance test class 'object' must have null COCO metrics"
            )

    result = {
        "status": "completed",
        "scope": (
            "final held-out test; configuration frozen before access; "
            "no training decisions permitted"
            if final_test
            else "validation split only; held-out test split not accessed"
        ),
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "annotation_file": str(annotation_path),
        "annotation_sha256": annotation_sha256,
        "images": len(dataset),
        "detections": len(predictions),
        "supported_category_count": sum(
            item["supported"] for item in per_class
        ),
        "unsupported_categories": unsupported_categories,
        "device": str(device),
        "runtime": {
            "torch_version": str(torch.__version__),
            "torchvision_version": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "inference_settings": inference_settings,
        "inference": inference,
        "overall": overall,
        "per_class": per_class,
    }
    if not final_test:
        selector_values = [
            item["coco_ap"]
            for item in per_class
            if item["category_name"] != "object" and item["coco_ap"] is not None
        ]
        result["selection_metric"] = {
            "name": "mean_per_class_coco_ap_excluding_object",
            "value": float(np.mean(selector_values)),
            "excluded_categories": ["object"],
            "minimum_improvement": 0.005,
        }

    if final_test:
        output_dir.mkdir(parents=True, exist_ok=False)
        metrics_path = output_dir / "test_metrics.json"
        predictions_path = output_dir / "test_predictions.json"
    else:
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else checkpoint_path.parent / f"validation_epoch_{completed_epochs:03d}"
        )
        metrics_path = output_dir / "validation_metrics.json"
        predictions_path = output_dir / "validation_predictions.json"

    write_json_atomic(predictions_path, predictions, compact=True)
    result["predictions_file"] = str(predictions_path)
    result["predictions_sha256"] = sha256(predictions_path)
    write_json_atomic(metrics_path, result)
    print(
        f"{split_name.title()} metrics saved: {metrics_path}"
    )


if __name__ == "__main__":
    main()
