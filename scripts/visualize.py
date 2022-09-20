import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

softmax = torch.nn.Softmax()


def get_images(samples_dir):
    samples_dir = Path(samples_dir)

    def _iter_pil_images():
        sample_images = sorted(list(samples_dir.glob("*.jpg")))
        for p in sample_images:
            image = Image.open(p)
            yield image

    return tuple(_iter_pil_images())


def main(pt_file, output_image_file):
    logits = torch.load(output_pt_file)
    converted = (softmax(logits) > 0.95).type(torch.uint8)
    # converted = softmax(logits)
    images = get_images("data/samples")
    fig, axes = plt.subplots(len(images), 2, figsize=(9, 10))
    for ax, orig_image, result_image in zip(axes, images, converted):
        # permute to match the desired memory format
        # result_image = result_image.permute(1, 2, 0).detach().numpy()
        result_image = result_image.detach().numpy()
        save(ax, orig_image, result_image)

    plt.tight_layout()
    plt.savefig(output_image_file, bbox_inches="tight")


def save(ax, orig_image, result_image):
    ax[0].set_title("Original image")
    ax[0].imshow(orig_image)
    ax[1].set_title("hand / ring-finger")
    ax[1].imshow(result_image[1] + result_image[2] * 2, interpolation="none")
    # plt.savefig(f"output{idx}.png")


if __name__ == "__main__":
    output_pt_file = sys.argv[1]
    output_image_file = sys.argv[2]
    main(output_pt_file, output_image_file)
