import pathlib
import yaml
from datetime import datetime
from loguru import logger
from typing import List, Optional

from melnet.defaults import (
    CHECKPOINT_EXTENSION,
    CONFIG_EXTENSION,
    LOG_EXTENSION,
    MODEL_LIST,
    OPTIMIZER_LIST,
)


class TrainingConfig:
    def __init__(self, config_path: pathlib.Path) -> None:
        logger.info(f"Reading config-file: {config_path}")

        # check config-file extension
        if config_path.suffix.lower() != CONFIG_EXTENSION.lower():
            logger.error(
                f"Config file: {config_path} needs to have {CONFIG_EXTENSION} extension."
            )
            exit(1)

        # read config
        try:
            self.config = yaml.safe_load(open(config_path.as_posix()))
        except FileNotFoundError:
            logger.error(f"Config file: {config_path} is not found.")
            exit(1)

        # read parameters
        self.dataset_root = pathlib.Path(self._read_param(["DATASET_ROOT"], str))
        self.checkpoint_root = pathlib.Path(self._read_param(["CHECKPOINT_ROOT"], str))
        self.model_architecture = self._read_param(["MODEL_ARCHITECTURE"], str)
        self.load_checkpoint = self._read_param(["LOAD_CHECKPOINT"], str)
        self.fine_tune_fc = self._read_param(["FINE_TUNE_FC"], bool)
        self.optimizer = self._read_param(["OPTIMIZER"], str)
        self.learning_rate = self._read_param(["LEARNING_RATE"], float)
        self.schedular_step = self._read_param(["SCHEDULER_STEP"], float)
        self.schedular_gamma = self._read_param(["SCHEDULER_GAMMA"], float)
        self.momentum = self._read_param(["MOMENTUM"], float)
        self.num_workers = self._read_param(["NUM_WORKER"], int)
        self.batch_size = self._read_param(["BATCH_SIZE"], int)
        self.epochs = self._read_param(["EPOCHS"], int)
        self.input_size = self._read_param(["INPUT_SIZE"], int)
        self.cross_validation_fold = self._read_param(["CROSS_VALIDATION_FOLDS"], int)
        self.single_fold_split = self._read_param(["SINGLE_FOLD_SPLIT"], float)

        # check: model_architecture is supported
        if self.model_architecture not in MODEL_LIST:
            logger.error(
                f"Model Architecture: {self.model_architecture} is not supported."
            )
            logger.error(f"Supported Model Architectures: {MODEL_LIST}.")
            exit(1)

        # check: optimizer is supported
        if self.optimizer not in OPTIMIZER_LIST:
            logger.error(f"Optimizer: {self.optimizer} is not supported.")
            logger.error(f"Supported Optimizers: {OPTIMIZER_LIST}.")
            exit(1)

        # check: cross_validation_fold is valid
        if self.cross_validation_fold <= 0:
            logger.error(
                f"Cross Valiadation Fold: {self.cross_validation_fold} is not supported."
            )
            logger.error("Supported Cross Valiadation Fold: any integer bigger than 0.")
            exit(1)

        # get checkpoint-directory
        current = datetime.now()

        checkpoint_name = (
            "checkpoint_"
            + str(current.year)
            + str(current.month).zfill(2)
            + str(current.day).zfill(2)
            + str(current.hour).zfill(2)
            + str(current.minute).zfill(2)
            + str(current.second).zfill(2)
        )

        self.checkpoint_directory = self.checkpoint_root / pathlib.Path(checkpoint_name)

        # generate path for copying config-file
        self.config_dst = self.checkpoint_directory / config_path.name

    def _read_param(self, keys: List, data_type: type) -> Optional[any]:
        # get the parameter value
        try:
            obj = self.config
            for key in keys:
                obj = obj[key]

            # convert value to the expected datatype
            try:
                logger.debug(f"Config {keys} = {obj}")
                if obj is not None:
                    return data_type(obj)
                else:
                    return None
            except ValueError:
                logger.error(f"Failed to convert {obj} to {data_type}.")
                exit(1)

        except KeyError:
            logger.error(f"Key: {keys} is not found in the config.")
            exit(1)

    def get_log_path(self, fold: int) -> pathlib.Path:
        filename = f"log_fold{fold}{LOG_EXTENSION}"
        return self.checkpoint_directory / pathlib.Path(filename)

    def get_checkpoint_path(self, fold: int) -> pathlib.Path:
        filename = f"checkpoint_fold{fold}{CHECKPOINT_EXTENSION}"
        return self.checkpoint_directory / pathlib.Path(filename)
