#!/usr/bin/env python3
"""Smoke-test or train the civilian Faster R-CNN component detector.

With no ``--epochs`` argument, this performs the original one-image optimizer
smoke step. Epoch mode adds only a bounded pilot or an epoch-boundary resumable
checkpoint; validation, test evaluation, and overlays remain separate steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms.functional import pil_to_tensor


MODEL_NAME = "fasterrcnn_resnet50_fpn"
ANNOTATION_FILE = Path("annotations/carparts23_instances_train.json")
EXPECTED_CATEGORY_COUNT = 23
CHECKPOINT_SCHEMA_VERSION = 1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
TRAINABLE_BACKBONE_LAYERS = 3


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def smoke_image_size(value: str) -> int:
    parsed = int(value)
    if parsed < 32:
        raise argparse.ArgumentTypeError("must be at least 32 pixels")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root,
        help="Dataset root containing images/ and annotations/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a mode-specific directory beneath runs/.",
    )
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=positive_float, default=0.005)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--weights",
        choices=("none", "coco"),
        default="none",
        help=(
            "Use 'none' for an offline mechanics check. The eventual baseline "
            "must use 'coco' pretrained weights."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=smoke_image_size,
        default=320,
        help="Fixed detector input edge length.",
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        help="Total target epochs. Omit for the one-image smoke step.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--max-batches",
        type=positive_int,
        help="Run a disposable bounded pilot; no checkpoint is saved.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from an epoch-boundary checkpoint_last.pt.",
    )
    parser.add_argument("--log-interval", type=positive_int, default=10)

    args = parser.parse_args()
    if args.max_batches is not None and args.epochs is None:
        parser.error("--max-batches requires --epochs")
    if args.resume is not None and args.epochs is None:
        parser.error("--resume requires --epochs")
    if args.resume is not None and args.max_batches is not None:
        parser.error("a bounded pilot cannot be resumed")
    if args.epochs is not None and args.resume is None and args.weights != "coco":
        parser.error("new epoch training must start with --weights coco")
    if args.resume is not None and args.weights != "none":
        parser.error("omit --weights when resuming; the checkpoint supplies them")
    if args.output_dir is None:
        if args.resume is not None:
            args.output_dir = args.resume.parent
        else:
            run_name = (
                "civilian_detector_smoke"
                if args.epochs is None
                else (
                    "civilian_detector_pilot"
                    if args.max_batches is not None
                    else "civilian_detector_training"
                )
            )
            args.output_dir = project_root / "runs" / run_name
    return args


class CivilianPartsDataset(CocoDetection):
    """Adapt the prepared COCO boxes to TorchVision detector targets."""

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image, annotations = super().__getitem__(index)
        if not annotations:
            raise ValueError(f"Image index {index} has no annotations")

        boxes = torch.tensor(
            [annotation["bbox"] for annotation in annotations], dtype=torch.float32
        )
        boxes[:, 2:] += boxes[:, :2]  # COCO xywh -> Faster R-CNN xyxy.

        target = {
            "boxes": boxes,
            "labels": torch.tensor(
                [annotation["category_id"] for annotation in annotations],
                dtype=torch.int64,
            ),
            "image_id": torch.tensor([self.ids[index]], dtype=torch.int64),
            "area": torch.tensor(
                [annotation["area"] for annotation in annotations],
                dtype=torch.float32,
            ),
            "iscrowd": torch.tensor(
                [annotation.get("iscrowd", 0) for annotation in annotations],
                dtype=torch.int64,
            ),
        }
        image_tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255)
        return image_tensor, target


def category_names(dataset: CivilianPartsDataset) -> list[str]:
    categories = sorted(dataset.coco.dataset["categories"], key=lambda item: item["id"])
    ids = [category["id"] for category in categories]
    expected_ids = list(range(1, EXPECTED_CATEGORY_COUNT + 1))
    if ids != expected_ids:
        raise ValueError(f"Expected contiguous category IDs {expected_ids}; found {ids}")
    return [category["name"] for category in categories]


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot use CUDA")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def build_model(num_classes: int, weights_name: str, image_size: int) -> torch.nn.Module:
    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        if weights_name == "coco"
        else None
    )
    if weights is None:
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            norm_layer=FrozenBatchNorm2d,
            trainable_layers=TRAINABLE_BACKBONE_LAYERS,
        )
        return FasterRCNN(
            backbone,
            num_classes=num_classes,
            min_size=image_size,
            max_size=image_size,
        )

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        min_size=image_size,
        max_size=image_size,
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    return model


def to_device(target: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in target.items()
    }


def create_optimizer(
    model: torch.nn.Module, learning_rate: float
) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )


def train_one_step(
    model: torch.nn.Module,
    image: Tensor,
    target: dict[str, Tensor],
    device: torch.device,
    learning_rate: float,
) -> tuple[dict[str, float], float]:
    model.to(device).train()
    optimizer = create_optimizer(model, learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    losses = model([image.to(device)], [to_device(target, device)])
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Training produced a non-finite loss: {total_loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - started_at
    return {
        **{name: value.detach().item() for name, value in losses.items()},
        "total": total_loss.detach().item(),
    }, step_seconds


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detection_collate(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def training_contract(
    args: argparse.Namespace,
    dataset: CivilianPartsDataset,
    names: list[str],
    annotation_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "initial_weights": "coco",
        "num_classes_including_background": len(names) + 1,
        "categories": names,
        "annotation_file": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "dataset_images": len(dataset),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "trainable_backbone_layers": TRAINABLE_BACKBONE_LAYERS,
        "amp": device.type == "cuda",
        "augmentation": "none",
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def run_training_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dataset: CivilianPartsDataset,
    device: torch.device,
    args: argparse.Namespace,
    epoch_index: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(args.seed + epoch_index)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=detection_collate,
        generator=generator,
    )
    batches_to_run = min(len(loader), args.max_batches or len(loader))
    amp_enabled = device.type == "cuda"
    loss_sums: dict[str, float] = {}
    images_seen = 0

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    for batch_index, (images, targets) in enumerate(
        islice(loader, batches_to_run), start=1
    ):
        images = [
            image.to(device, non_blocking=device.type == "cuda") for image in images
        ]
        targets = [to_device(target, device) for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            losses = model(images, targets)
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Epoch {epoch_index + 1}, batch {batch_index} produced "
                f"non-finite loss {total_loss.item()}"
            )
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_losses = {
            **{name: value.detach().item() for name, value in losses.items()},
            "total": total_loss.detach().item(),
        }
        for name, value in batch_losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value
        images_seen += len(images)
        if batch_index % args.log_interval == 0 or batch_index == batches_to_run:
            print(
                f"epoch={epoch_index + 1} batch={batch_index}/{batches_to_run} "
                f"loss={batch_losses['total']:.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    if not loss_sums:
        raise RuntimeError("Training epoch completed without processing a batch")
    mean_losses = {
        name: value / batches_to_run for name, value in loss_sums.items()
    }
    return {
        "epoch": epoch_index + 1,
        "batches": batches_to_run,
        "full_epoch_batches": len(loader),
        "images": images_seen,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_batch": elapsed_seconds / batches_to_run,
        "projected_full_epoch_minutes": (
            elapsed_seconds / batches_to_run * len(loader) / 60
        ),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "losses": mean_losses,
    }


def validate_resume_contract(
    saved: dict[str, Any], current: dict[str, Any]
) -> None:
    saved = dict(saved)
    saved.setdefault("trainable_backbone_layers", TRAINABLE_BACKBONE_LAYERS)
    locked_fields = (
        "model",
        "initial_weights",
        "num_classes_including_background",
        "categories",
        "annotation_sha256",
        "dataset_images",
        "image_size",
        "batch_size",
        "seed",
        "learning_rate",
        "momentum",
        "weight_decay",
        "trainable_backbone_layers",
        "amp",
        "augmentation",
        "device",
        "torch_version",
        "torchvision_version",
        "cuda_runtime",
    )
    mismatches = [field for field in locked_fields if saved.get(field) != current.get(field)]
    if mismatches:
        raise ValueError(
            "Resume settings differ from the checkpoint: " + ", ".join(mismatches)
        )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    completed_epochs: int,
    history: list[dict[str, Any]],
    contract: dict[str, Any],
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed_epochs": completed_epochs,
        "global_step": sum(epoch["batches"] for epoch in history),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "history": history,
        "contract": contract,
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def run_smoke(
    args: argparse.Namespace,
    dataset: CivilianPartsDataset,
    names: list[str],
    annotation_path: Path,
    device: torch.device,
) -> None:
    if not 0 <= args.image_index < len(dataset):
        raise ValueError(
            f"image-index must be between 0 and {len(dataset) - 1}; "
            f"received {args.image_index}"
        )

    image, target = dataset[args.image_index]
    model = build_model(len(names) + 1, args.weights, args.image_size)
    losses, step_seconds = train_one_step(
        model, image, target, device, args.learning_rate
    )

    image_id = int(target["image_id"].item())
    image_record = dataset.coco.loadImgs(image_id)[0]
    result = {
        "status": "passed",
        "scope": "one optimizer-step smoke test; not a baseline checkpoint",
        "model": MODEL_NAME,
        "weights": args.weights,
        "num_classes_including_background": len(names) + 1,
        "categories": names,
        "annotation_file": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "image_index": args.image_index,
        "image_id": image_id,
        "image_file": image_record["file_name"],
        "box_count": int(target["boxes"].shape[0]),
        "image_size": args.image_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "step_seconds": step_seconds,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        ),
        "losses": losses,
    }
    result_path = args.output_dir.resolve() / "smoke_result.json"
    write_json(result_path, result)
    print(f"Civilian detector smoke step passed: {result_path}")


def run_epoch_training(
    args: argparse.Namespace,
    dataset: CivilianPartsDataset,
    names: list[str],
    annotation_path: Path,
    device: torch.device,
) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_last.pt"
    if args.resume is None and checkpoint_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint without --resume: "
            f"{checkpoint_path}"
        )
    if args.resume is not None and args.resume.resolve() != checkpoint_path:
        raise ValueError(
            "Resume checkpoint must be checkpoint_last.pt in --output-dir"
        )
    contract = training_contract(args, dataset, names, annotation_path, device)
    model_weights = "none" if args.resume is not None else args.weights
    model = build_model(len(names) + 1, model_weights, args.image_size).to(device)
    optimizer = create_optimizer(model, args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    completed_epochs = 0
    history: list[dict[str, Any]] = []

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume.resolve(), map_location="cpu", weights_only=True
        )
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema version")
        validate_resume_contract(checkpoint["contract"], contract)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        completed_epochs = checkpoint["completed_epochs"]
        history = checkpoint["history"]
        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if completed_epochs >= args.epochs:
            raise ValueError(
                f"Checkpoint already completed {completed_epochs} epochs; "
                f"target is {args.epochs}"
            )

    invocation = {
        "target_epochs": args.epochs,
        "max_batches": args.max_batches,
        "num_workers": args.num_workers,
        "resume": str(args.resume.resolve()) if args.resume is not None else None,
        "contract": contract,
    }
    write_json(output_dir / "training_config.json", invocation)

    if args.max_batches is not None:
        stats = run_training_epoch(
            model, optimizer, scaler, dataset, device, args, epoch_index=0
        )
        result = {
            "status": "passed",
            "scope": "bounded pilot; model discarded and no checkpoint saved",
            "stats": stats,
            "contract": contract,
        }
        result_path = output_dir / "pilot_result.json"
        write_json(result_path, result)
        print(f"Civilian detector pilot passed: {result_path}")
        return

    for epoch_index in range(completed_epochs, args.epochs):
        stats = run_training_epoch(
            model, optimizer, scaler, dataset, device, args, epoch_index
        )
        history.append(stats)
        completed_epochs = epoch_index + 1
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            completed_epochs,
            history,
            contract,
        )
        write_json(
            output_dir / "training_history.json",
            {
                "status": (
                    "completed" if completed_epochs == args.epochs else "in_progress"
                ),
                "completed_epochs": completed_epochs,
                "target_epochs": args.epochs,
                "history": history,
            },
        )
        print(f"Saved epoch {completed_epochs} checkpoint: {checkpoint_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = args.data_root.resolve()
    annotation_path = data_root / ANNOTATION_FILE
    dataset = CivilianPartsDataset(root=data_root, annFile=annotation_path)
    names = category_names(dataset)
    device = resolve_device(args.device)
    if args.epochs is None:
        run_smoke(args, dataset, names, annotation_path, device)
    else:
        run_epoch_training(args, dataset, names, annotation_path, device)


if __name__ == "__main__":
    main()
