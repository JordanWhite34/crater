"""YOLOX-S fine-tuning experiment for the six CRATER military components."""

from pathlib import Path

from yolox.data import COCODataset, TrainTransform, ValTransform
from yolox.exp import Exp as YOLOXExp

from standard_coco_evaluator import StandardCOCOEvaluator


class Exp(YOLOXExp):
    """Fine-tune a civilian-initialized YOLOX-S detector on military data."""

    def __init__(self):
        super().__init__()

        project_root = Path(__file__).resolve().parents[2]

        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 6

        self.data_dir = str(
            project_root / "datasets" / "military" / "components"
        )
        self.train_ann = "crater6_instances_train.json"
        self.val_ann = "crater6_instances_val.json"
        self.test_ann = "crater6_instances_test.json"

        self.exp_name = Path(__file__).stem
        self.output_dir = str(
            project_root / "outputs" / "detection" / "military"
        )

    def get_dataset(self, cache=False, cache_type="ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        annotation_file = (
            self.test_ann if kwargs.get("testdev", False) else self.val_ann
        )
        return COCODataset(
            data_dir=self.data_dir,
            json_file=annotation_file,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def get_evaluator(
        self,
        batch_size,
        is_distributed,
        testdev=False,
        legacy=False,
    ):
        return StandardCOCOEvaluator(
            dataloader=self.get_eval_loader(
                batch_size,
                is_distributed,
                testdev=testdev,
                legacy=legacy,
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
