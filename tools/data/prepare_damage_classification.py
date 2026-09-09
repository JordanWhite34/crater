"""Prepare reviewed CVAT part crops for real-then-synthetic damage training."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from tools.data.import_cvat_damage_annotations import PART_GROUPS, sha256_file


SPLITS = ("train", "val", "test")
REVIEW_FIELDS = (
    "source_image", "source_image_sha256", "domain", "group_id", "review_status",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inventory(root: Path, include_synthetic: bool = True) -> list[dict]:
    """Join labels to image provenance and verify source hashes before use."""
    damage_root = root / "datasets/military/damage/source"
    sources = [("real", damage_root / "humvee_cvat_damage_v1",
                root / "datasets/military/components/source/humvee")]
    if include_synthetic:
        sources.append(("synthetic", damage_root / "synthetic_humvee_damage_v1",
                        damage_root / "synthetic_humvee_damage_v1/images"))
    levels = json.loads((root / "configs/taxonomy.json").read_text())["damage"]["levels"]
    inventory, annotation_ids = [], set()
    for domain, source_dir, image_dir in sources:
        label_file = source_dir / "damage_box_annotations.csv"
        manifest_file = (source_dir if domain == "synthetic" else image_dir) / "source_manifest.csv"
        manifest_rows = read_csv(manifest_file)
        manifest = {r["file_name"]: r for r in manifest_rows}
        if len(manifest) != len(manifest_rows):
            raise ValueError(f"Duplicate image names in {manifest_file}")
        label_hash, manifest_hash = sha256_file(label_file), sha256_file(manifest_file)
        verified = set()
        for row in read_csv(label_file):
            name = row["source_image"]
            if name not in manifest or Path(name).name != name:
                raise ValueError(f"Unknown or unsafe source filename: {name}")
            provenance = manifest[name]
            expected = row["source_image_sha256"].lower()
            if expected != provenance["sha256"].lower():
                raise ValueError(f"Annotation/manifest hash mismatch: {name}")
            image_path = image_dir / name
            if name not in verified:
                if sha256_file(image_path) != expected:
                    raise ValueError(f"Source image hash mismatch: {image_path}")
                with Image.open(image_path) as image:
                    if image.size != (int(provenance["width"]), int(provenance["height"])):
                        raise ValueError(f"Image dimensions mismatch: {name}")
                    image.verify()
                verified.add(name)
            annotation_id = row["annotation_id"]
            if annotation_id in annotation_ids or not re.fullmatch(r"[A-Za-z0-9_-]+", annotation_id):
                raise ValueError(f"Duplicate or unsafe annotation ID: {annotation_id}")
            annotation_ids.add(annotation_id)
            if row["damage_level"] not in levels or row["annotation_source"] != "manual":
                raise ValueError(f"Invalid damage label or nonmanual box: {annotation_id}")
            if PART_GROUPS.get(row["component_class"]) != row["component_group"]:
                raise ValueError(f"Unexpected component grouping: {annotation_id}")
            box = tuple(float(row[f"box_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))
            x1, y1, x2, y2 = box
            if not (all(math.isfinite(x) for x in box)
                    and 0 <= x1 < x2 <= int(provenance["width"])
                    and 0 <= y1 < y2 <= int(provenance["height"])):
                raise ValueError(f"Invalid box: {annotation_id}")
            inventory.append({
                **row, "source_image_sha256": expected, "domain": domain,
                "image_path": image_path.relative_to(root).as_posix(),
                "source_labels_sha256": label_hash,
                "source_manifest_sha256": manifest_hash,
                "generation_group_id": provenance.get("generation_group_id", ""),
            })
    return inventory


def proposed_review_rows(inventory: list[dict]) -> list[dict]:
    """Seed review with the existing detector pilot's *interim* photo groups.

    Commons ID proximity is only a review aid, not evidence of scene identity.
    No row is accepted automatically, including synthetic annotations.
    """
    images = {(r["domain"], r["source_image"]): r for r in inventory}
    real = sorted((r for r in images.values() if r["domain"] == "real"),
                  key=lambda r: int(re.search(r"\d+", r["source_image"])[0]))
    groups = []
    for row in real:
        image_id = int(re.search(r"\d+", row["source_image"])[0])
        if not groups or image_id - groups[-1][-1][0] > 5000:
            groups.append([])
        groups[-1].append((image_id, row["source_image"]))
    for linked in (
        {"commons_72095749.jpg", "commons_73531531.jpg"},
        {"commons_52358471.jpg", "commons_53183995.jpg", "commons_53326023.jpg"},
        {"commons_40286022.jpg", "commons_56467136.jpg"},
    ):
        related = [g for g in groups if linked.intersection(name for _, name in g)]
        if related:
            groups = [g for g in groups if g not in related] + [sum(related, [])]
    by_name = {name: f"real_{min(i for i, _ in group)}"
               for group in groups for _, name in group}
    return [dict(source_image=name, source_image_sha256=row["source_image_sha256"],
                 domain=domain, group_id=(by_name[name] if domain == "real"
                                         else row["generation_group_id"]),
                 review_status="pending")
            for (domain, name), row in sorted(images.items())]


def apply_review(inventory: list[dict], review_path: Path, seed: int = 42) -> list[dict]:
    """Require completed QA, then split real source groups before making crops."""
    reviews = read_csv(review_path)
    by_key = {(r["domain"], r["source_image"]): r for r in reviews}
    expected = {(r["domain"], r["source_image"]) for r in inventory}
    if len(by_key) != len(reviews) or set(by_key) != expected:
        raise ValueError("Review CSV must contain exactly one row per inventory image.")
    pending = [r["source_image"] for r in reviews if r["review_status"] not in {"accept", "reject"}]
    if pending:
        raise ValueError(f"Review {len(pending)} pending images in {review_path}; "
                         "verify groups, boxes and labels, then mark accept or reject.")
    accepted = []
    hash_groups, group_domains = {}, {}
    for row in inventory:
        review = by_key[(row["domain"], row["source_image"])]
        if review["source_image_sha256"].lower() != row["source_image_sha256"]:
            raise ValueError(f"Review hash mismatch: {row['source_image']}")
        if review["review_status"] == "reject":
            continue
        group = review["group_id"].strip()
        if not group:
            raise ValueError(f"Missing reviewed group: {row['source_image']}")
        if hash_groups.setdefault(row["source_image_sha256"], group) != group:
            raise ValueError("Identical image content assigned to different groups.")
        if group_domains.setdefault(group, row["domain"]) != row["domain"]:
            raise ValueError("A source group crosses real and synthetic domains.")
        accepted.append({**row, "group_id": group})
    group_images = defaultdict(set)
    for row in accepted:
        if row["domain"] == "real":
            group_images[row["group_id"]].add(row["source_image_sha256"])
    if len(group_images) < 3:
        raise ValueError("At least three accepted real source groups are required.")
    groups = sorted(group_images)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda g: len(group_images[g]), reverse=True)
    total = sum(map(len, group_images.values()))
    targets = dict(zip(SPLITS, (0.70 * total, 0.15 * total, 0.15 * total)))
    counts, assignments = Counter(), {}
    for group in groups:
        split = max(SPLITS, key=lambda s: (targets[s] - counts[s]) / targets[s])
        assignments[group] = split
        counts[split] += len(group_images[group])
    for row in accepted:
        row["split"] = "train" if row["domain"] == "synthetic" else assignments[row["group_id"]]
    validate_splits(accepted)
    return accepted


def validate_splits(rows: list[dict]) -> None:
    for key in ("source_image_sha256", "group_id"):
        seen = {}
        for row in rows:
            if row["split"] not in SPLITS:
                raise ValueError(f"Invalid split: {row['split']}")
            if row["domain"] not in {"real", "synthetic"}:
                raise ValueError(f"Invalid domain: {row['domain']}")
            if row["domain"] == "synthetic" and row["split"] != "train":
                raise ValueError("Synthetic images must remain training-only.")
            if seen.setdefault(row[key], row["split"]) != row["split"]:
                raise ValueError(f"{key} crosses dataset splits: {row[key]}")
    if {r["split"] for r in rows if r["domain"] == "real"} != set(SPLITS):
        raise ValueError("Real train, validation and test splits must all be nonempty.")


def materialize_crops(root: Path, rows: list[dict], destination: Path) -> list[dict]:
    """Keep entire annotated boxes; save hashes and lineage beside PNG crops."""
    validate_splits(rows)
    destination.mkdir(parents=True, exist_ok=False)
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_path"]].append(row)
    result = []
    for image_path, annotations in sorted(by_image.items()):
        path = root / image_path
        if sha256_file(path) != annotations[0]["source_image_sha256"]:
            raise ValueError(f"Image changed since inventory load: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            for row in annotations:
                box = tuple((math.floor if i < 2 else math.ceil)(float(row[f"box_{axis}"]))
                            for i, axis in enumerate(("x1", "y1", "x2", "y2")))
                crop_file = f"crops/{row['domain']}/{row['annotation_id']}.png"
                output = destination / crop_file
                output.parent.mkdir(parents=True, exist_ok=True)
                image.crop(box).save(output)
                result.append({**row, "crop_file": crop_file,
                               "crop_sha256": sha256_file(output)})
    write_csv(destination / "crops_manifest.csv", result)
    return result
