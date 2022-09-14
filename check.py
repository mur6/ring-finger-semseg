import argparse
import random
import time
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np

base_dir = Path("./blender-for-finger-segmentation/data")
print(base_dir)

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
print(feature_extractor)
