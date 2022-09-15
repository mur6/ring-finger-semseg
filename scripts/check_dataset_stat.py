from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from dataset import ImageSegmentationDataset


def check_dataset():
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    train_dataset = ImageSegmentationDataset(
        root_dir=Path("data/outputs/training/"), feature_extractor=feature_extractor, transforms=None
    )

    loader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=0)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for encoded_inputs in loader:
        images = encoded_inputs["pixel_values"]
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        # print(batch_samples)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    print(len(loader))
    print(f"mean={mean} std={std}")


if __name__ == "__main__":
    check_dataset()
