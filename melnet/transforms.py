import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

from melnet.utils import get_RGB_image


class Transforms:
    def __init__(self, image_paths, input_size):
        self.image_paths = image_paths
        self.input_size = input_size
        self.mean = None
        self.std = None

        self.calc_mean_std()

    def calc_mean_std(self):
        rgb_values = (
            np.concatenate(
                [
                    cv2.resize(
                        get_RGB_image(image_path, color_flag=1),
                        (self.input_size, self.input_size),
                    )
                    for image_path in self.image_paths
                ],
                axis=0,
            )
            / 255.0
        )
        rgb_values = np.reshape(rgb_values, (-1, 3))

        self.mean = np.mean(rgb_values, axis=0)
        self.std = np.std(rgb_values, axis=0)

    def get_train_transform(self):
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

    def get_val_transform(self):
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
