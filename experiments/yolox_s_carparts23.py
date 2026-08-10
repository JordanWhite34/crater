from pathlib import Path

from yolox.exp import Exp as YOLOXExp
from yolox.data import COCODataset, TrainTransform, ValTransform


class Exp(YOLOXExp):
    def __init__(self):
        super().__init__()

        project_root = Path(__file__).resolve().parents[1]

        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 23

        self.data_dir = str(project_root)
        self.train_ann = "carparts23_instances_train.json"
        self.val_ann = "carparts23_instances_val.json"
        self.test_ann = "carparts23_instances_test.json"

        self.exp_name = Path(__file__).stem
        self.output_dir = str(project_root / "outputs" / "yolox")

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
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name="",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )