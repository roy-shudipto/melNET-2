import pathlib
import yaml
from datetime import datetime
from itertools import product
from loguru import logger
from typing import List, Optional

from melnet.defaults import (
    CHECKPOINT_EXTENSION,
    CONFIG_EXTENSION,
    CONFIG_OUTPUT_NAME,
    LOG_EXTENSION,
    MODEL_LIST,
    OPTIMIZER_LIST,
)


class TrainingConfig:
    def __init__(self, config: dict) -> None:
        self.config = config

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
        self.config_dst = self.checkpoint_directory / CONFIG_OUTPUT_NAME

    def _read_param(self, keys: List, data_type: type) -> Optional[any]:
        # get the parameter value
        try:
            obj = self.config
            for key in keys:
                obj = obj[key]
        except KeyError:
            logger.error(f"Key: {keys} is not found in the config.")
            exit(1)

        # boolean
        if data_type is bool:
            if obj in [True, False]:
                logger.debug(f"Config {keys} = {data_type(obj)}")
                return obj
            else:
                logger.error(f"Key: {keys} needs to be a boolean [true, false].")
                exit(1)

        # convert value to the expected datatype
        try:
            if obj is None:
                logger.debug(f"Config {keys} = None")
                return None
            elif str(obj).upper() in ["NONE", "NULL"]:
                logger.debug(f"Config {keys} = None")
                return None
            else:
                logger.debug(f"Config {keys} = {data_type(obj)}")
                return data_type(obj)
        except ValueError:
            logger.error(f"Failed to convert {obj} to {data_type}.")
            exit(1)

    def get_log_path(self, fold: int) -> pathlib.Path:
        filename = f"log_fold{fold}{LOG_EXTENSION}"
        return self.checkpoint_directory / pathlib.Path(filename)

    def get_checkpoint_path(self, fold: int) -> pathlib.Path:
        filename = f"checkpoint_fold{fold}{CHECKPOINT_EXTENSION}"
        return self.checkpoint_directory / pathlib.Path(filename)

    def save(self) -> None:
        with open(self.config_dst, "w") as file:
            yaml.dump(self.config, file, sort_keys=False)


def parse_config(config_path: pathlib.Path) -> List[dict]:
    logger.info(f"Reading config-file: {config_path}")

    # check config-file extension
    if config_path.suffix.lower() != CONFIG_EXTENSION.lower():
        logger.error(
            f"Config file: {config_path} needs to have {CONFIG_EXTENSION} extension."
        )
        exit(1)

    # read config
    try:
        config = yaml.safe_load(open(config_path.as_posix()))
    except FileNotFoundError:
        logger.error(f"Config file: {config_path} is not found.")
        exit(1)

    # generate config-variations
    val_list = []
    for _, val in config.items():
        if isinstance(val, list):
            val_list.append(val)
        else:
            val_list.append([val])

    combinations = list(product(*val_list))

    config_variations = []
    for combination in combinations:
        config_dict = {}
        for idx, key in enumerate(config.keys()):
            config_dict[key] = combination[idx]

        config_variations.append(config_dict)

    return config_variations
