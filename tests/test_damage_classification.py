"""Data leakage, metric and real-to-synthetic checkpoint contract tests."""

import copy
import numpy as np
import pytest
from PIL import Image

from tools.data.import_cvat_damage_annotations import sha256_file
from tools.data.prepare_damage_classification import (
    REVIEW_FIELDS, apply_review, materialize_crops, validate_splits, write_csv,
)


LEVELS = ["no_visible_damage", "possible_damage", "severe_visible_damage", "unobservable"]


def inventory_fixture(root):
    rows = []
    for index in range(12):
        domain = "real" if index < 9 else "synthetic"
        path = root / f"commons_{index}.png"
        Image.new("RGB", (40, 30), (index * 20, 50, 80)).save(path)
        for label in range(4):
            rows.append({
                "annotation_id": f"box_{index}_{label}", "source_image": path.name,
                "source_image_sha256": sha256_file(path), "domain": domain,
                "image_path": path.name, "component_group": "mobility",
                "component_class": "wheel_tire", "damage_level": LEVELS[label],
                "box_x1": "0.2", "box_y1": "1.5", "box_x2": "30.3", "box_y2": "25.1",
                "generation_group_id": f"synthetic_{index}" if domain == "synthetic" else "",
            })
    reviews = [{"source_image": r["source_image"], "source_image_sha256": r["source_image_sha256"],
                "domain": r["domain"], "group_id": f"group_{i}", "review_status": "accept"}
               for i, r in enumerate(rows[::4])]
    review_path = root / "review.csv"
    write_csv(review_path, reviews, REVIEW_FIELDS)
    return rows, reviews, review_path


def test_group_split_is_deterministic_and_synthetic_is_training_only(tmp_path):
    rows, reviews, path = inventory_fixture(tmp_path)
    reviews[1]["group_id"] = reviews[0]["group_id"]
    write_csv(path, reviews, REVIEW_FIELDS)
    first = apply_review(rows, path)
    second = apply_review(list(reversed(rows)), path)
    assignments = lambda items: {r["annotation_id"]: r["split"] for r in items}
    assert assignments(first) == assignments(second)
    validate_splits(first)
    linked_names = {reviews[0]["source_image"], reviews[1]["source_image"]}
    linked = [r["split"] for r in first if r["source_image"] in linked_names]
    assert len(set(linked)) == 1
    assert {r["split"] for r in first if r["domain"] == "synthetic"} == {"train"}


def test_pending_review_and_changed_hash_fail(tmp_path):
    rows, reviews, path = inventory_fixture(tmp_path)
    reviews[0]["review_status"] = "pending"
    write_csv(path, reviews, REVIEW_FIELDS)
    with pytest.raises(ValueError, match="pending"):
        apply_review(rows, path)
    reviews[0]["review_status"] = "accept"
    reviews[0]["source_image_sha256"] = "changed"
    write_csv(path, reviews, REVIEW_FIELDS)
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_review(rows, path)


def test_leakage_rejected_by_group_hash_and_domain(tmp_path):
    rows, _, path = inventory_fixture(tmp_path)
    accepted = apply_review(rows, path)
    for key in ("group_id", "source_image_sha256"):
        broken = copy.deepcopy(accepted)
        train = next(r for r in broken if r["split"] == "train")
        val = next(r for r in broken if r["split"] == "val")
        val[key] = train[key]
        with pytest.raises(ValueError, match="crosses dataset splits"):
            validate_splits(broken)
    broken = copy.deepcopy(accepted)
    next(r for r in broken if r["domain"] == "synthetic")["split"] = "test"
    with pytest.raises(ValueError, match="training-only"):
        validate_splits(broken)


def test_crop_geometry_lineage_and_no_overwrite(tmp_path):
    rows, _, path = inventory_fixture(tmp_path)
    prepared = materialize_crops(tmp_path, apply_review(rows, path), tmp_path / "prepared")
    assert len(prepared) == 48
    crop = tmp_path / "prepared" / prepared[0]["crop_file"]
    with Image.open(crop) as image:
        assert image.size == (31, 25)  # floor starts, ceil ends; preserve whole box
    assert prepared[0]["crop_sha256"] == sha256_file(crop)
    assert prepared[0]["source_image_sha256"] == rows[0]["source_image_sha256"]
    with pytest.raises(FileExistsError):
        materialize_crops(tmp_path, apply_review(rows, path), tmp_path / "prepared")


def test_metrics_keep_absent_classes_explicit():
    from experiments.damage.training import classification_metrics
    metrics = classification_metrics([0, 0, 1], np.eye(4)[[0, 1, 1]], LEVELS)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["severe_visible_damage"]["recall"] is None
    assert metrics["confusion_matrix"] == [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_real_then_synthetic_training_and_checkpoint_reload(tmp_path):
    import torch
    from experiments.damage.training import (
        TrainingConfig, evaluate_checkpoint, final_comparison, load_completed_stage, train_stage,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        rows, _, path = inventory_fixture(tmp_path)
        crop_root = tmp_path / "prepared"
        prepared = materialize_crops(tmp_path, apply_review(rows, path), crop_root)
        config = TrainingConfig(epochs=1, batch_size=16, image_size=32)
        common = dict(rows=prepared, crop_root=crop_root, group="mobility", levels=LEVELS,
                      config=config, device="cpu", initialization="scratch")
        load_common = {key: value for key, value in common.items() if key != "device"}
        baseline = train_stage(**common, output_dir=tmp_path / "real", domain="real")
        assert load_completed_stage(
            **load_common, output_dir=tmp_path / "real", domain="real"
        ) == baseline
        original_hash = sha256_file(baseline)
        adapted = train_stage(**common, output_dir=tmp_path / "synthetic", domain="synthetic",
                              parent_checkpoint=baseline)
        assert load_completed_stage(
            **load_common, output_dir=tmp_path / "synthetic", domain="synthetic",
            parent_checkpoint=baseline,
        ) == adapted
        assert sha256_file(baseline) == original_hash
        payload = torch.load(adapted, weights_only=True)
        assert payload["parent_checkpoint_sha256"] == original_hash
        assert payload["levels"] == LEVELS
        assert payload["validation"]["macro_f1"] >= torch.load(baseline, weights_only=True)["validation"]["macro_f1"]
        result = final_comparison({"real": baseline, "synthetic": adapted}, prepared,
                                  crop_root, tmp_path / "test", "cpu")
        assert result["real"]["samples"] == result["synthetic"]["samples"]
        assert result["real"]["class_order"] == LEVELS
        with pytest.raises(FileExistsError):
            final_comparison({"real": baseline}, prepared, crop_root, tmp_path / "test", "cpu")
        with pytest.raises(ValueError, match="real parent"):
            train_stage(**common, output_dir=tmp_path / "bad", domain="synthetic")
        with pytest.raises(ValueError, match="current.*config"):
            load_completed_stage(
                **{**load_common, "config": TrainingConfig(epochs=2, batch_size=16, image_size=32)},
                output_dir=tmp_path / "real", domain="real",
            )
        changed = copy.deepcopy(prepared)
        changed[0]["damage_level"] = LEVELS[1]
        with pytest.raises(ValueError, match="differ from the saved manifest"):
            evaluate_checkpoint(baseline, changed, crop_root, "cpu")
        (crop_root / prepared[0]["crop_file"]).write_bytes(b"changed image")
        with pytest.raises(ValueError, match="changed or unsafe crop"):
            train_stage(**common, output_dir=tmp_path / "changed", domain="real")
    finally:
        torch.set_num_threads(previous_threads)
