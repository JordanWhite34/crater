"""
ingests batch of images of vehicles, uses model and writes output file per image in outputs/.

this expects a vehicle crop of 1 single image.
"""
from tqdm import tqdm
import argparse

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}

def parse_args():
    # parses CLI args, configurable are:
        # model checkpoint
        # path to input images
        # output path for component bounding boxes
        # confidence threshold
        # inference device

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, help="path to model checkpoint")
    parser.add_argument("-i", "--input_path", type=str, help="path to folder of input images")
    parser.add_argument("-o", "--output_path", type=str, help="path to destination for model outputs")
    parser.add_argument("-c", "--confidence_threshold", type=float, help="inference confidence threshold")
    parser.add_argument("-d", "--device", type=str, help="device used for inference")

    return parser.parse_args()
