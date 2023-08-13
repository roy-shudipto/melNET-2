import cv2
import numpy as np
import pathlib
from loguru import logger
from sklearn.model_selection import StratifiedKFold


def get_RGB_image(filepath: pathlib.Path, color_flag: int = 1) -> np.ndarray:
    if color_flag not in [0, 1]:
        exit("Expected color-flag: [0, 1]")
    try:
        image = cv2.imread(filepath.as_posix(), color_flag)
    except AssertionError:
        exit(f"Unable to read image from: [{filepath}].")

    if image is None:
        exit(f"[{filepath}] returns a None object.")

    return np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def init_counter(d_src: dict) -> dict:
    d_dst = {}
    for key in d_src.keys():
        d_dst[key] = 0
    return d_dst


def valid_counter(d: dict) -> bool:
    for _, val in d.items():
        if val < 1:
            return False
    return True


def get_skf(folds: int, single_fold_split: float) -> StratifiedKFold:
    if folds > 1:
        return StratifiedKFold(n_splits=folds, shuffle=False, random_state=None)
    elif folds == 1:
        # for a single fold, use single_fold_split (train/val split)
        # we can use StratifiedKFold for this purpose: n_splits = 1/(1-train_split)
        logger.info(
            f"Single fold detected. Using train/val split of {single_fold_split:.2f}/{1-single_fold_split:.2f}"
        )
        return StratifiedKFold(
            n_splits=round(1 / (1 - single_fold_split)),
            shuffle=False,
            random_state=None,
        )
