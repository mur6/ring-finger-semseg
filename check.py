import argparse
import random
import time
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np

base_data_dir = Path("../blender-for-finger-segmentation/")
# print(base_dir)

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
# print(feature_extractor)


from dataset import ImageSegmentationDataset

train_dataset = ImageSegmentationDataset(
    root_dir=base_data_dir / "training", feature_extractor=feature_extractor, transforms=None
)
valid_dataset = ImageSegmentationDataset(
    root_dir=base_data_dir / "validation", feature_extractor=feature_extractor, transforms=None, train=False
)
