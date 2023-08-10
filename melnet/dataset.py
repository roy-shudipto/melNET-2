import albumentations as A
import cv2
import pathlib
from albumentations.pytorch import ToTensorV2
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from typing import List


# compose transform
train_transform = A.Compose(
    [
        A.Resize(width=400, height=400),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RGBShift(p=0.5),
        A.Normalize(
            mean=(0.63918286, 0.43600938, 0.32808192),
            std=(0.14957589, 0.13829103, 0.11973361),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
val_transform = A.Compose(
    [
        A.Resize(width=400, height=400),
        A.Normalize(
            mean=(0.63918286, 0.43600938, 0.32808192),
            std=(0.14957589, 0.13829103, 0.11973361),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


def get_image(filepath: pathlib.Path, color_flag: int = 1):
    if color_flag not in [0, 1]:
        exit("Expected color-flag: [0, 1]")
    try:
        image = cv2.imread(filepath.as_posix(), color_flag)
    except AssertionError:
        exit(f"Unable to read image from: [{filepath}].")

    if image is None:
        exit(f"[{filepath}] returns a None object.")

    return image


class TrainingDataset(Dataset):
    def __init__(self, img_paths, class_ids, transform):
        # class variables
        self.img_list = img_paths
        self.cls_list = class_ids
        self.transform = transform

    def __getitem__(self, index):
        # read the image
        image = get_image(pathlib.Path(self.img_list[index]), color_flag=1)

        # read class-id
        class_id = self.cls_list[index]

        # return the package
        return self.transform(image=image)["image"], class_id

    def __len__(self):
        return len(self.img_list)


class ClassificationDatasetFolds:
    def __init__(
        self, dataset_root: pathlib.Path, folds: int, single_fold_split: float
    ):
        self.folds = folds
        self.dataset_root = dataset_root
        self.single_fold_split = single_fold_split
        self.image_paths: List = []
        self.class_ids: List = []
        self.class_map: dict = {}
        self.number_of_classes = None

        # for a single fold, use single_fold_split (train/val split)
        # we can use StratifiedKFold for this purpose: n_splits = 1/(1-train_split)
        if self.folds == 1:
            self.skf = StratifiedKFold(
                n_splits=round(1 / (1 - self.single_fold_split)),
                shuffle=False,
                random_state=None,
            )
            logger.info(
                f"Single fold detected. Using train/val split of {single_fold_split:.2f}/{1-single_fold_split:.2f}"
            )
        else:
            self.skf = StratifiedKFold(
                n_splits=self.folds, shuffle=False, random_state=None
            )

        self._get_data()

    def _get_data(self):
        for idx, dataset_dir in enumerate(self.dataset_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            self.class_map[idx] = dataset_dir.name
            for dataset_file in dataset_dir.iterdir():
                if not dataset_file.is_file():
                    continue
                self.image_paths.append(dataset_file)
                self.class_ids.append(idx)

        self.number_of_classes = len(self.class_map)

    def get_datasets(
        self, *, fold_index: int, batch_size: int, num_worker: int
    ) -> dict:
        # check: valid fold-index
        if fold_index < 0 or fold_index >= self.folds:
            logger.error(f"Received invalid fold index: {fold_index}")
            logger.error(f"Fold index needs be within: 0 to {self.folds-1}")
            exit(1)

        # return dataloader-dictionary for the input fold-index
        for idx, (train_index, val_index) in enumerate(
            self.skf.split(self.image_paths, self.class_ids)
        ):
            if idx != fold_index:
                continue

            # get train-dataloader
            train_dataset = TrainingDataset(
                [self.image_paths[i] for i in train_index],
                [self.class_ids[i] for i in train_index],
                train_transform,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_worker,
            )

            # get val-dataloader
            val_dataset = TrainingDataset(
                [self.image_paths[i] for i in val_index],
                [self.class_ids[i] for i in val_index],
                val_transform,
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker
            )

            return {"train": train_dataloader, "val": val_dataloader}
