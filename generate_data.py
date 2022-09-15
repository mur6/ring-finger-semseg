import argparse
import random
import time
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
from PIL import Image

noise_t = A.Compose(
    [
        A.GaussNoise(),
        A.Blur(blur_limit=3),
        # A.OpticalDistortion(),
        # A.GridDistortion(),
        A.OneOf(
            [
                A.Cutout(num_holes=40, max_h_size=4, max_w_size=4),
                A.Cutout(num_holes=35, max_h_size=8, max_w_size=8),
                A.Cutout(num_holes=20, max_h_size=16, max_w_size=16),
            ],
            p=0.8,
        ),
    ]
)


def get_mat_image(*, imsize):
    image_file = Path("data/images/mat_high.jpeg")
    # img = cv2.imread(str(image_file), 0)
    img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, imsize)
    return img


def make_random_four_points(*, rho, imsize):
    x = random.randint(0, rho)
    y = random.randint(0, rho)
    yield x, y
    x = imsize - random.randint(0, rho)
    y = random.randint(0, rho)
    yield x, y
    x = imsize - random.randint(0, rho)
    y = imsize - random.randint(0, rho)
    yield x, y
    x = random.randint(0, rho)
    y = imsize - random.randint(0, rho)
    yield x, y


# class PreProcess:
#     imsize = (IM_SIZE, IM_SIZE)
#     original_four_points = np.float32([(0, 0), (IM_SIZE, 0), (IM_SIZE, IM_SIZE), (0, IM_SIZE)])
#     original_four_points_numpy = np.array(original_four_points)
#     mat_image = get_mat_image(imsize=imsize)

#     def __init__(
#         self, new_path: Path, data_size: int, background_images: List[np.ndarray], generate_image: bool
#     ) -> None:
#         if not new_path.exists():
#             new_path.mkdir(parents=True)
#         self.new_path = new_path
#         self.data_size = data_size
#         self.rho = RHO
#         self.background_images = background_images
#         self.generate_image = generate_image

#     def make_warped_image(self):
#         # img = self.mat_image.copy()
#         perturbed_four_points = list(make_random_four_points(rho=self.rho, imsize=self.imsize[0]))
#         H_inverse = cv2.getPerspectiveTransform(self.original_four_points, np.float32(perturbed_four_points))
#         dst = get_random_background(self.background_images)
#         warped_image = cv2.warpPerspective(
#             self.mat_image, H_inverse, self.imsize, borderMode=cv2.BORDER_TRANSPARENT, dst=dst
#         )
#         warped_image = noise_t(image=warped_image)["image"]
#         return warped_image, perturbed_four_points

#     def get_datum(self, warped_image, perturbed_four_points):
#         # training_image = np.dstack((self.mat_image, warped_image))
#         training_image = warped_image
#         # print(training_image.shape)
#         perturbed_four_points_numpy = np.array(perturbed_four_points)
#         H_four_points = np.subtract(perturbed_four_points_numpy, self.original_four_points_numpy)
#         return (training_image, perturbed_four_points_numpy, H_four_points)

#     def __call__(self):
#         # save .npy files
#         # print("Generate {} {} files from {} raw data...".format(data_size, new_path, len(filenames)))
#         for i in range(self.data_size):
#             warped_image, perturbed_four_points = self.make_warped_image()
#             d = self.get_datum(warped_image, perturbed_four_points)
#             np.save(self.new_path / f"{i:06d}", d)
#             if self.generate_image:
#                 cv2.imwrite(str(self.new_path / f"{i:06d}.jpeg"), warped_image)


def init_background_images(path):
    image_size = (224, 224)

    def _iter():
        for image_file in Path(path).glob("*.jpg"):
            image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] >= image_size[0] and image.shape[1] >= image_size[1]:
                yield image

    return list(_iter())


t_for_background = A.Compose(
    [
        A.RandomCrop(width=224, height=224),
        # A.HorizontalFlip(p=0.5),
        A.RandomRotate90(),
        A.Flip(),
    ]
)


def get_random_background(background_images):
    image = random.choice(background_images)
    augmented_image = t_for_background(image=image)["image"]
    return augmented_image


import matplotlib.pyplot as plt


def composite_image(*, fg, bg, segmentation_map):
    image_mask = np.where(segmentation_map != 0, 1, 0)
    im_c = fg * np.stack([image_mask] * 3, axis=2)
    im_g = bg * np.stack([1 - image_mask] * 3, axis=2)
    return im_c + im_g


def main(args):
    start = time.time()

    output_dir = Path("data/outputs/")
    background_images = init_background_images(args.background_image_path)
    bg = get_random_background(background_images)
    # b = background_images[0]
    # mask = encoded_inputs["labels"].numpy()
    base_data_dir = Path("../blender-for-finger-segmentation/")
    from dataset import ImageSegmentationDataset

    train_dataset = ImageSegmentationDataset(root_dir=base_data_dir / "training")
    image, segmentation_map = train_dataset[0]

    im_out = composite_image(fg=image, bg=bg, segmentation_map=segmentation_map)

    plt.imshow(im_out)
    plt.show()
    # kwargs = dict(
    #     , generate_image=args.generate_image
    # )

    # PreProcess(output_dir / "training/", data_size=args.training_data_size, **kwargs)()
    # PreProcess(output_dir / "validation/", data_size=args.validation_data_size, **kwargs)()
    # PreProcess(output_dir / "testing/", data_size=args.testing_data_size, **kwargs)()

    elapsed_time = time.time() - start
    print(
        "Generate dataset in {:.0f}h {:.0f}m {:.0f}s.".format(
            elapsed_time // 3600,
            (elapsed_time % 3600) // 60,
            (elapsed_time % 3600) % 60,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    coco_data_path = "/Users/taichi.muraki/workspace/Python/fiftyone/coco2017/validation/data"
    # "/content/data"
    parser.add_argument("--background_image_path", type=Path, default=coco_data_path)
    # parser.add_argument("--training_data_size", type=int, default=100, help="training data_size")
    # parser.add_argument("--validation_data_size", type=int, default=10, help="validation data_size")
    # parser.add_argument("--testing_data_size", type=int, default=10, help="testing data_size")
    # parser.add_argument("--generate_image", type=bool, default=False, help="Generate jpeg image or not")
    args = parser.parse_args()
    main(args)
