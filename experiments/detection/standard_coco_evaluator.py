"""Portable COCO evaluation for CRATER YOLOX experiments.

YOLOX's default evaluator prefers a JIT-compiled C++ implementation. This
subclass keeps the same YOLOX evaluation pipeline while using pycocotools'
standard Python extension, so evaluation does not require an MSVC toolchain.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

from loguru import logger
from pycocotools.cocoeval import COCOeval
from yolox.evaluators.coco_evaluator import (
    COCOEvaluator,
    per_class_AP_table,
    per_class_AR_table,
)
from yolox.utils import is_main_process


class StandardCOCOEvaluator(COCOEvaluator):
    """Evaluate detections with the standard pycocotools COCO evaluator."""

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluating with standard pycocotools COCOeval.")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()
        batch_size = self.dataloader.batch_size

        average_inference_time = 1000 * inference_time / (n_samples * batch_size)
        average_nms_time = 1000 * nms_time / (n_samples * batch_size)
        time_info = (
            "Average forward time: {:.2f} ms, Average NMS time: {:.2f} ms, "
            "Average inference time: {:.2f} ms\n"
        ).format(
            average_inference_time,
            average_nms_time,
            average_inference_time + average_nms_time,
        )

        if not data_dict:
            return 0, 0, time_info

        coco_ground_truth = self.dataloader.dataset.coco
        results_path = self._write_results_file(data_dict)
        try:
            coco_detections = coco_ground_truth.loadRes(str(results_path))
        finally:
            results_path.unlink(missing_ok=True)

        coco_eval = COCOeval(coco_ground_truth, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        summary = io.StringIO()
        with contextlib.redirect_stdout(summary):
            coco_eval.summarize()

        info = time_info + summary.getvalue()
        category_ids = sorted(coco_ground_truth.cats)
        category_names = [
            coco_ground_truth.cats[category_id]["name"]
            for category_id in category_ids
        ]

        if self.per_class_AP:
            info += "per class AP:\n" + per_class_AP_table(coco_eval, category_names)
        if self.per_class_AR:
            info += "per class AR:\n" + per_class_AR_table(coco_eval, category_names)

        return coco_eval.stats[0], coco_eval.stats[1], info

    @staticmethod
    def _write_results_file(data_dict) -> Path:
        """Write detections to a closed temporary file for Windows compatibility."""

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as results_file:
            json.dump(data_dict, results_file)
            return Path(results_file.name)
