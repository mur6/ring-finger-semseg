import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

noise_t = A.Compose(
    [
        A.OneOf(
            [
                A.Cutout(num_holes=50, max_h_size=8, max_w_size=8),
                A.Cutout(num_holes=40, max_h_size=12, max_w_size=12),
                A.Cutout(num_holes=30, max_h_size=16, max_w_size=16),
            ],
            p=0.9,
        ),
        A.GaussNoise(),
        A.Blur(blur_limit=3),
        A.RandomBrightness(limit=0.4),
        A.RandomScale(0.25),
        A.Rotate(border_mode=cv2.BORDER_CONSTANT),
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
    # print(image_mask)
    # k = np.stack([image_mask] * 3, axis=2)
    # print("stack.shape:", k.shape)
    # print(k)
    im_fg = fg * np.stack([image_mask] * 3, axis=2)
    im_bg = bg * np.stack([1 - image_mask] * 3, axis=2)
    return (im_fg + im_bg).astype(np.uint8)


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


def generate_and_save_images(
    blender_image_path, *, output_image_dir, output_mask_dir, background_image_iter, output_visualize_data=False
):
    it = iter_image_and_segmentation_map(blender_image_path)
    # it = list(it)[:5]
    for idx, (image, segmentation_map) in enumerate(it):
        image = composite_image(fg=image, bg=next(background_image_iter), segmentation_map=segmentation_map)
        # print("composite type:", image.dtype)
        transformed = noise_t(image=image, mask=segmentation_map)
        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]
        # print(type(image), image.shape)
        cv2.imwrite(str(output_image_dir / f"image_{idx:06}.jpg"), transformed_image[..., ::-1])
        cv2.imwrite(str(output_mask_dir / f"image_{idx:06}.png"), transformed_mask)
        if output_visualize_data:
            plt.imshow(transformed_mask)
            plt.savefig(str(output_mask_dir / "../visualized" / f"image_{idx:06}.jpg"))


def main(args):
    background_image_iter = iter_background_images(args.background_image_path)
    sub_dir = args.target
    output_dir = args.output_base_path / sub_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)
    (output_dir / "visualized").mkdir(exist_ok=True)
    blender_image_path = args.blender_image_base_path / sub_dir
    generate_and_save_images(
        blender_image_path,
        output_image_dir=output_dir / "images",
        output_mask_dir=output_dir / "masks",
        background_image_iter=background_image_iter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # coco_data_path = "/Users/taichi.muraki/workspace/Python/fiftyone/coco2017/validation/data"
    parser.add_argument("--background_image_path", type=Path, default="/content/data")
    parser.add_argument("--blender_image_base_path", type=Path, required=True)
    parser.add_argument("--output_base_path", type=Path, default="data/outputs")
    parser.add_argument("--target", default="training")

    args = parser.parse_args()
    main(args)
