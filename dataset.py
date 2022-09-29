# import os

import albumentations as aug
import cv2
import numpy as np

# from transformers import AdamW
import torch
from PIL import Image

# from sklearn.metrics import accuracy_score
# from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

# import pandas as pd


class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, transforms=None, train=True):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        self.img_path = self.root_dir / "images"
        self.mask_path = self.root_dir / "masks"
        self.transforms = transforms

        # read images
        image_path_iter = self.img_path.glob("*.jpg")
        self.images = sorted(image_path_iter, key=lambda p: p.name)
        # read annotations
        mask_path_iter = self.mask_path.glob("*.png")
        # for root, dirs, files in os.walk(self.ann_dir):
        self.masks = sorted(mask_path_iter, key=lambda p: p.name)
        print(f"images count: {len(self.images)}")
        print(f"masks count: {len(self.masks)}")
        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation_map = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        # segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        # segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)
        #         image = Image.open()
        #         segmentation_map = Image.open()

        if self.transforms is not None:
            norm_image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
            norm_image = self.transforms(norm_image)
            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.feature_extractor(norm_image, segmentation_map, return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


if __name__ == "__main__":
    from pathlib import Path

    from transformers import SegformerFeatureExtractor

    base_data_dir = Path("../blender-for-finger-segmentation/data2/")

    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

    train_dataset = ImageSegmentationDataset(
        base_data_dir / "training",
        # "data/datasets/contour_checked_numbers_training.json",
        feature_extractor=feature_extractor_inference,
    )
    valid_dataset = ImageSegmentationDataset(
        base_data_dir / "validation",
        # "data/datasets/contour_checked_numbers_validation.json",
        feature_extractor=feature_extractor_inference,
    )
    t = train_dataset[0]
    pixel_values = t["pixel_values"]
    labels = t["labels"]
    # points = t["points"]
    print(pixel_values.shape, labels.shape)
