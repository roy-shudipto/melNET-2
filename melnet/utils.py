import cv2
import pathlib
import numpy as np


def get_RGB_image(filepath: pathlib.Path, color_flag: int = 1):
    if color_flag not in [0, 1]:
        exit("Expected color-flag: [0, 1]")
    try:
        image = cv2.imread(filepath.as_posix(), color_flag)
    except AssertionError:
        exit(f"Unable to read image from: [{filepath}].")

    if image is None:
        exit(f"[{filepath}] returns a None object.")

    return np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
