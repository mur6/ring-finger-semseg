import argparse
import random
import time
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

noise_t = A.Compose(
    [
        A.OneOf(
            [
                A.Cutout(num_holes=40, max_h_size=4, max_w_size=4),
                A.Cutout(num_holes=35, max_h_size=8, max_w_size=8),
                A.Cutout(num_holes=20, max_h_size=16, max_w_size=16),
            ],
            p=0.8,
        ),
        A.GaussNoise(),
        A.Blur(blur_limit=3),
        # A.OpticalDistortion(),
        # A.GridDistortion(),
    ]
)


# def get_mat_image(*, imsize):
#     image_file = Path("data/image.jpeg")
#     img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, imsize)
#     return img


IMAGE_SIZE = (224, 224)


def iter_background_images(path):
    bg_transform = A.Compose(
        [
            A.RandomCrop(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1]),
            # A.HorizontalFlip(p=0.5),
            A.RandomRotate90(),
            A.Flip(),
        ]
    )

    def _iter():
        for image_file in Path(path).glob("*.jpg"):
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] >= IMAGE_SIZE[0] and image.shape[1] >= IMAGE_SIZE[1]:
                yield image

    background_images = list(_iter())
    while True:
        image = random.choice(background_images)
        augmented_image = bg_transform(image=image)["image"]
        yield augmented_image


def composite_image(*, fg, bg, segmentation_map):
    image_mask = np.where(segmentation_map != 0, 1, 0)
    im_c = fg * np.stack([image_mask] * 3, axis=2)
    im_g = bg * np.stack([1 - image_mask] * 3, axis=2)
    return im_c + im_g


def iter_image_and_segmentation_map(root_dir):
    def _load(root_dir):
        image_path_iter = (root_dir / "images").glob("*.jpg")
        images = sorted(image_path_iter, key=lambda p: p.name)
        mask_path_iter = (root_dir / "masks").glob("*.png")
        masks = sorted(mask_path_iter, key=lambda p: p.name)
        print(f"images count: {len(images)}")
        print(f"masks count: {len(masks)}")
        assert len(images) == len(masks), "There must be as many images as there are segmentation maps"
        return images, masks

    images, masks = _load(root_dir)
    for image, mask in zip(images, masks):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation_map = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        yield image, segmentation_map


def main(args):
    output_dir = Path("data/outputs/")
    output_dir.mkdir(exist_ok=True)

    background_image_iter = iter_background_images(args.background_image_path)

    base_data_dir = Path("../blender-for-finger-segmentation/")
    it = iter_image_and_segmentation_map(base_data_dir / "training")

    for image, segmentation_map in it:
        image = composite_image(fg=image, bg=next(background_image_iter), segmentation_map=segmentation_map)
        image = noise_t(image=image)["image"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    coco_data_path = "/Users/taichi.muraki/workspace/Python/fiftyone/coco2017/validation/data"
    # "/content/data"
    parser.add_argument("--background_image_path", type=Path, default=coco_data_path)

    args = parser.parse_args()
    main(args)
