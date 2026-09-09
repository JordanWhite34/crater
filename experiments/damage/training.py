"""Small, inspectable ResNet18 baseline for canonical part-damage labels."""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from tools.data.import_cvat_damage_annotations import sha256_file
from tools.data.prepare_damage_classification import read_csv, validate_splits, write_csv


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 1e-4
    seed: int = 42
    image_size: int = 224


class Letterbox:
    """Preserve all visible damage and aspect ratio instead of center-cropping."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR,
                            color=(124, 116, 104))


class DamageCrops(Dataset):
    def __init__(self, rows, crop_root, levels, size, training=False):
        self.rows, self.crop_root, self.levels = rows, crop_root, levels
        steps = [Letterbox(size)]
        if training:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend([transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.crop_root / row["crop_file"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.levels.index(row["damage_level"])


def build_model(class_count: int, initialization: str = "scratch"):
    if initialization not in {"scratch", "imagenet"}:
        raise ValueError("Initialization must be scratch or imagenet.")
    weights = ResNet18_Weights.IMAGENET1K_V1 if initialization == "imagenet" else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def classification_metrics(labels, probabilities, levels):
    """Use a fixed class order; absent recall is null, fixed-class F1 is zero."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if not len(labels):
        raise ValueError("Cannot evaluate an empty split.")
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((len(levels), len(levels)), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    f1 = np.divide(2 * true_positive, support + predicted,
                   out=np.zeros(len(levels)), where=(support + predicted) > 0)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for index in range(10):
        selected = (confidence >= index / 10) & (
            (confidence < (index + 1) / 10) if index < 9 else (confidence <= 1))
        if selected.any():
            ece += float(selected.mean() * abs(correct[selected].mean() - confidence[selected].mean()))
    return {
        "samples": len(labels), "accuracy": float(correct.mean()),
        "macro_f1": float(f1.mean()), "ece_10_bins": ece,
        "negative_log_likelihood": float(-np.log(np.clip(
            probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "brier_score": float(((probabilities - np.eye(len(levels))[labels]) ** 2).sum(axis=1).mean()),
        "class_order": list(levels), "confusion_matrix": confusion.tolist(),
        "per_class": {name: {"support": int(support[i]), "f1": float(f1[i]),
                              "recall": float(true_positive[i] / support[i]) if support[i] else None}
                      for i, name in enumerate(levels)},
    }


@torch.inference_mode()
def predict(model, loader, device, levels):
    model.eval()
    labels, probabilities = [], []
    for images, targets in loader:
        probabilities.extend(model(images.to(device)).softmax(dim=1).cpu().tolist())
        labels.extend(targets.tolist())
    return classification_metrics(labels, probabilities, levels), probabilities


def make_loader(rows, crop_root, levels, config, training=False):
    # num_workers=0 is intentional: reliable in Windows notebook kernels.
    return DataLoader(DamageCrops(rows, crop_root, levels, config.image_size, training),
                      batch_size=config.batch_size, shuffle=training, num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed))


def verify_crops(rows, crop_root):
    for row in rows:
        path = (crop_root / row["crop_file"]).resolve()
        if crop_root.resolve() not in path.parents or sha256_file(path) != row["crop_sha256"]:
            raise ValueError(f"Missing, changed or unsafe crop: {path}")


def verify_manifest(rows, crop_root):
    manifest = crop_root / "crops_manifest.csv"
    if read_csv(manifest) != [{k: str(v) for k, v in row.items()} for row in rows]:
        raise ValueError("In-memory crop labels/splits differ from the saved manifest.")
    return sha256_file(manifest)


def train_stage(*, rows, crop_root: Path, output_dir: Path, group: str, levels,
                domain: str, config: TrainingConfig, device: str,
                initialization: str = "imagenet", parent_checkpoint: Path | None = None):
    """Train on one domain; select epochs exclusively on the fixed real val set.

    Synthetic training requires the real-stage checkpoint from this same crop
    manifest. Epoch zero remains a valid fallback if synthetic training hurts.
    """
    validate_splits(rows)
    if domain not in {"real", "synthetic"} or (domain == "synthetic") != (parent_checkpoint is not None):
        raise ValueError("Real stage starts fresh; synthetic stage requires its real parent.")
    if (config.epochs < 1 or config.batch_size < 2 or config.image_size < 32
            or config.learning_rate <= 0 or config.weight_decay < 0):
        raise ValueError("Use positive epochs/lr, batch_size >= 2, image_size >= 32 and weight_decay >= 0.")
    train_rows = [r for r in rows if r["component_group"] == group and r["domain"] == domain and r["split"] == "train"]
    val_rows = [r for r in rows if r["component_group"] == group and r["domain"] == "real" and r["split"] == "val"]
    if len(train_rows) < 2 or not val_rows:
        raise ValueError(f"{group}/{domain}: need >=2 training crops and a nonempty real validation set.")
    verify_crops(train_rows + val_rows, crop_root)
    manifest_hash = verify_manifest(rows, crop_root)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    parent = None
    if parent_checkpoint is not None:
        parent = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
        if (parent["group"] != group or parent["levels"] != list(levels)
                or parent["domain"] != "real" or parent["manifest_sha256"] != manifest_hash
                or parent["config"]["image_size"] != config.image_size):
            raise ValueError("Parent checkpoint group, labels, real stage, data or preprocessing mismatch.")
    model = build_model(len(levels), "scratch" if parent else initialization)
    if parent:
        model.load_state_dict(parent["model"])
    model.to(device)
    counts = np.bincount([levels.index(r["damage_level"]) for r in train_rows], minlength=len(levels))
    missing = [levels[i] for i, count in enumerate(counts) if not count]
    if missing:
        warnings.warn(f"{group}/{domain}: no training examples for {missing}; those outputs are unsupported.")
    if np.count_nonzero(counts) < 2:
        raise ValueError(f"{group}/{domain}: only one damage class in training; collect/review more labeled groups.")
    # Gentle inverse-frequency correction, computed from this stage's training rows only.
    weights = torch.tensor(1 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(train_rows, crop_root, levels, config, training=True)
    val_loader = make_loader(val_rows, crop_root, levels, config)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "architecture": "resnet18", "group": group, "levels": list(levels), "domain": domain,
        "config": asdict(config), "manifest_sha256": manifest_hash,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint) if parent else None,
        "initialization": "real_stage_checkpoint" if parent else initialization,
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "device": str(device), "training_class_counts": counts.tolist(),
        "unsupported_training_labels": missing,
        "preprocessing": "RGB; whole-box letterbox; ImageNet normalization; train horizontal flip",
        "selection": "highest real validation macro F1, ties keep earlier epoch",
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def save(name, epoch, metrics):
        torch.save({**metadata, "epoch": epoch, "validation": metrics,
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, output_dir / name)

    history, best_score = [], -1.0
    if parent:
        baseline, _ = predict(model, val_loader, device, levels)
        best_score = baseline["macro_f1"]
        save("best.pt", 0, baseline)
        history.append({"epoch": 0, "train_loss": None, "validation": baseline})
    for epoch in range(1, config.epochs + 1):
        model.train()
        # Freeze running BN statistics: small per-group batches otherwise drift,
        # especially during synthetic-only adaptation. Affine weights still learn.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        loss_total, loss_weight_total = 0.0, 0.0
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            targets = targets.to(device)
            loss = loss_function(model(images.to(device)), targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss in {group}/{domain}, epoch {epoch}.")
            loss.backward()
            optimizer.step()
            batch_weight = float(weights[targets].sum())
            loss_total += float(loss.detach()) * batch_weight
            loss_weight_total += batch_weight
        metrics, _ = predict(model, val_loader, device, levels)
        history.append({"epoch": epoch, "train_loss": loss_total / loss_weight_total, "validation": metrics})
        save("last.pt", epoch, metrics)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            save("best.pt", epoch, metrics)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"{group}/{domain} epoch {epoch}/{config.epochs}: "
              f"train loss {history[-1]['train_loss']:.4f}; real val macro F1 {metrics['macro_f1']:.4f}", flush=True)
    return output_dir / "best.pt"


def evaluate_checkpoint(checkpoint_path, rows, crop_root, device, split="test"):
    """Evaluate real crops only; preserve per-crop predictions for error review."""
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test.")
    validate_splits(rows)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if verify_manifest(rows, crop_root) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation data manifest differs from training.")
    selected = [r for r in rows if r["domain"] == "real" and r["split"] == split
                and r["component_group"] == checkpoint["group"]]
    if not selected:
        raise ValueError(f"No real {split} crops for {checkpoint['group']}")
    verify_crops(selected, crop_root)
    levels = checkpoint["levels"]
    model = build_model(len(levels))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    loader = make_loader(selected, crop_root, levels, TrainingConfig(**checkpoint["config"]))
    metrics, probabilities = predict(model, loader, device, levels)
    predictions = [{"annotation_id": row["annotation_id"], "source_image": row["source_image"],
                    "component_class": row["component_class"], "damage_level": row["damage_level"],
                    "prediction": levels[int(np.argmax(probs))],
                    **{f"p_{level}": float(prob) for level, prob in zip(levels, probs)}}
                   for row, probs in zip(selected, probabilities)]
    return {"checkpoint_sha256": sha256_file(checkpoint_path), "split": split,
            "group": checkpoint["group"], "epoch": checkpoint["epoch"], **metrics}, predictions


def final_comparison(checkpoints, rows, crop_root, output_dir, device):
    """Persist one paired test pass after checkpoint selection is frozen."""
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, checkpoint in checkpoints.items():
        if Path(name).name != name:
            raise ValueError("Comparison names must be plain filenames.")
        metrics, predictions = evaluate_checkpoint(checkpoint, rows, crop_root, device)
        results[name] = metrics
        write_csv(output_dir / f"{name}_predictions.csv", predictions)
        (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
