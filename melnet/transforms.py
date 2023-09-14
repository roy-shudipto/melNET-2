import albumentations as A
import cv2
import numpy as np
import random
from albumentations.pytorch import ToTensorV2
from loguru import logger

from melnet.defaults import IMAGE_SET_LIMIT_FOR_MEAN_STD_CALC
from melnet.utils import get_RGB_image


class Transforms:
    def __init__(self, image_paths, input_size) -> None:
        self.image_paths = image_paths
        self.input_size = input_size
        self.mean = None
        self.std = None

        self.calc_mean_std()

    def calc_mean_std(self) -> None:
        # calculate over a subset of images if the dataset length crosses a limit
        if len(self.image_paths) > IMAGE_SET_LIMIT_FOR_MEAN_STD_CALC:
            sub_set = random.sample(self.image_paths, IMAGE_SET_LIMIT_FOR_MEAN_STD_CALC)
        else:
            sub_set = self.image_paths

        # calculate mean, std
        rgb_values = (
            np.concatenate(
                [
                    cv2.resize(
                        get_RGB_image(image_path, color_flag=1),
                        (self.input_size, self.input_size),
                    )
                    for image_path in sub_set
                ],
                axis=0,
            )
            / 255.0
        )
        rgb_values = np.reshape(rgb_values, (-1, 3))

        self.mean = np.mean(rgb_values, axis=0)
        self.std = np.std(rgb_values, axis=0)

        logger.info(f"Calculated Mean: {self.mean}")
        logger.info(f"Calculated STD: {self.std}")

    def get_train_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(width=self.input_size, height=self.input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RGBShift(p=0.5),
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
                ToTensorV2(),
            ]
        )

    def get_val_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(width=self.input_size, height=self.input_size),
                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
                ToTensorV2(),
            ]
        )
