import pathlib
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

from melnet.defaults import CLASS_MAP
from melnet.transforms import Transforms
from melnet.utils import get_RGB_image, get_skf, init_counter, valid_counter


class TrainingDataset(Dataset):
    def __init__(self, img_paths, class_ids, transform) -> None:
        # class variables
        self.img_list = img_paths
        self.cls_list = class_ids
        self.transform = transform

    def __getitem__(self, index) -> Tuple:
        # read the image
        image = get_RGB_image(pathlib.Path(self.img_list[index]), color_flag=1)

        # read class-id
        class_id = self.cls_list[index]

        # return the package
        return self.transform(image=image)["image"], class_id

    def __len__(self) -> int:
        return len(self.img_list)


class ClassificationDatasetFolds:
    def __init__(
        self,
        dataset_root: pathlib.Path,
        input_size: int,
        folds: int,
        single_fold_split: float,
    ) -> None:
        self.dataset_root = dataset_root
        self.input_size = input_size
        self.folds = folds
        self.skf = get_skf(folds, single_fold_split)
        self.image_paths: List = []
        self.class_map: dict = CLASS_MAP
        self.number_of_classes = len(CLASS_MAP.keys())
        self.class_count = init_counter(CLASS_MAP)
        self.class_ids: List = []

        self._init_dataset()

        logger.info(f"Number of classes: {self.number_of_classes}")
        logger.info(f"Class Map: {self.class_map}")

        if valid_counter(self.class_count) is False:
            logger.error(f"Class Count: {self.class_count} is not valid.")
            exit(1)

        else:
            logger.info(f"Class Count: {self.class_count}")

        self.transform = Transforms(self.image_paths, self.input_size)

    def _init_dataset(self) -> None:
        for dataset_dir in self.dataset_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            if dataset_dir.name not in self.class_map.keys():
                continue
            for dataset_file in dataset_dir.iterdir():
                if not dataset_file.is_file():
                    continue
                self.image_paths.append(dataset_file)
                self.class_ids.append(self.class_map[dataset_dir.name])
                self.class_count[dataset_dir.name] += 1

    def get_datasets(
        self, *, fold_index: int, batch_size: int, num_worker: int
    ) -> dict:
        # check: valid fold-index
        if fold_index < 0 or fold_index >= self.folds:
            logger.error(f"Number of folds: {self.folds}")
            logger.error(f"Fold index needs be within: 0 to {self.folds-1}")
            logger.error(f"Received invalid fold index: {fold_index}")
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
                self.transform.get_train_transform(),
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
                self.transform.get_val_transform(),
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker
            )

            return {"train": train_dataloader, "val": val_dataloader}
