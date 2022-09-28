import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage import measure
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as transforms
from torchvision.transforms import Resize, ToTensor

albumentations_transform = A.Compose(
    [
        A.Resize(224, 224),
    ]
)


to_tensor = ToTensorV2()


def solve_points(mask):
    contours = measure.find_contours(mask, 1.5, mask=(mask > 0))
    if len(contours) != 1:
        raise ValueError("Contours length is not one line.")
    # print("length: ", len(contours))
    contour = contours[0][:, ::-1]
    point1, point2 = (contour[0], contour[-1])
    # print(point1, point2)
    return np.array([point1, point2])


def load_json(filename):
    content = Path(filename).read_text()
    return json.loads(content)


class RingFingerDataset(Dataset):
    def iter_file_path(self, base_path, *, extension):
        for p in base_path.glob("*." + extension):
            num = int(p.stem.split("_")[1])
            if num in self.contour_ok_number_set:
                yield p

    def __init__(self, root_dir, contour_ok_number_file, feature_extractor, transform=None):
        self.root_dir = root_dir
        self.img_path = self.root_dir / "images"
        self.mask_path = self.root_dir / "masks"
        self.contour_ok_number_set = set(load_json(contour_ok_number_file))
        self.feature_extractor = feature_extractor
        self.transform = transform

        # read images
        image_path_iter = self.iter_file_path(self.img_path, extension="jpg")
        self.images = sorted(image_path_iter, key=lambda p: p.name)
        # read annotations
        mask_path_iter = self.iter_file_path(self.mask_path, extension="png")
        self.masks = sorted(mask_path_iter, key=lambda p: p.name)
        print(f"images count: {len(self.images)}")
        print(f"masks count: {len(self.masks)}")
        assert len(self.images) == len(self.masks), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = Image.fromarray(aug["image"])
            mask = aug["mask"]

        # if self.transform is None:
        #    image = Image.fromarray(image)
        aug = albumentations_transform(image=image, mask=mask)
        image = aug["image"]
        mask = aug["mask"]
        image = to_tensor(image=image)["image"]
        points = torch.from_numpy(solve_points(mask))
        mask = torch.from_numpy(mask).long()

        # if self.transforms is not None:
        #     norm_image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        #     norm_image = self.transforms(norm_image)
        #     # randomly crop + pad both image and segmentation map to same size
        #     encoded_inputs = self.feature_extractor(norm_image, segmentation_map, return_tensors="pt")
        # else:
        #     encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        # for k, v in encoded_inputs.items():
        #     encoded_inputs[k].squeeze_()  # remove batch dimension

        return image, mask, points
