import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from src.model.orig_model import OrigSegformerForSemanticSegmentation, get_model


def get_images(samples_dir):
    samples_dir = Path(samples_dir)

    def _iter_pil_images():
        sample_images = sorted(list(samples_dir.glob("*.jpg")))
        for p in sample_images:
            image = Image.open(p)
            yield image

    return tuple(_iter_pil_images())


images = get_images("data/samples")

model_dir = "models/custom_segsem_02/"
model = OrigSegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval()

feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
inputs = feature_extractor_inference(images=images, return_tensors="pt")

with torch.no_grad():
    _, points = model(**inputs)
    # # shape (batch_size, num_labels, height, width)
    print(points)
    points = points.numpy()
    print(points)

    # logits = outputs.logits
    # print(f"outputs: {type(outputs)}")
    # print(f"outputs.logis: {logits.shape}")

    # masks = data[1]
    # masks[masks==2] = -1
    # points = data[2].numpy()
    # outputs = model(masks=masks.unsqueeze(0))
    points = points * 112 + 112
    print(points)

    # #, points)
    # print(masks.shape, points.shape)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(data[0].numpy().transpose(1, 2, 0))
    # ax[1].imshow(masks)

    # # draw_dot(ax[1], points[:2])
    # # draw_dot(ax[1], points[2:])
    # plt.show()
