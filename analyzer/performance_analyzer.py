import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from loguru import logger
from typing import List

from analyzer.defaults import (
    REFERENCE_DICT,
    X_AXIS_HEADER,
    TRAINING_LOG_HEADERS,
    TRAINING_LOG_EXTENSION,
)


class PerformaceAnalyzer:
    def __init__(self):
        pass

    @classmethod
    def training_log_is_valid(cls, training_log: pathlib.Path) -> bool:
        # check: file exists
        if not training_log.exists():
            logger.error(f"Training-log [{training_log}] does not exist.")
            return False

        # check: extension is valid
        if training_log.suffix.lower() != TRAINING_LOG_EXTENSION.lower():
            logger.error(
                f"Training-log [{training_log}] needs to have [{TRAINING_LOG_EXTENSION}] extension."
            )
            return False

        # check: training-log contains required header
        log_df = pd.read_csv(training_log)

        for header in TRAINING_LOG_HEADERS:
            if header in list(log_df.columns):
                continue
            logger.error(
                f"Header: [{header}] is missing in training-log: [{training_log}]"
            )
            return False

        return True

    def plot_training_performance(
        self, training_log: pathlib.Path, reference: List[str]
    ) -> None:
        # check: training-log is valid
        if self.training_log_is_valid(training_log) is False:
            logger.error(f"Invalid training-log: {training_log}")
            exit(1)

        # read training-log
        log_df = pd.read_csv(training_log)

        # plot training performance
        plt.title = plt.figure("Training-log Plot")
        plt.xlabel(f"X: {X_AXIS_HEADER}")
        plt.ylabel("Y: METRIC")
        x_vals = np.array(log_df[X_AXIS_HEADER].to_list())
        for y_axis_header in reference:
            y_vals = np.array(log_df[y_axis_header].to_list())
            plt.plot(x_vals, y_vals, REFERENCE_DICT[y_axis_header]["plt_style"])
        plt.legend(reference)
        plt.show()

    def find_best_epoch(self, training_log: pathlib.Path, reference: str) -> dict:
        # check: training-log is valid
        if self.training_log_is_valid(training_log) is False:
            logger.error(f"Invalid training-log: {training_log}")
            exit(1)

        # read training-log
        log_df = pd.read_csv(training_log)

        # find best epoch
        for filter_ref, filter_func_name in REFERENCE_DICT[reference][
            "filter_by"
        ].items():
            # check: filter_ref is valid
            if filter_ref not in TRAINING_LOG_HEADERS:
                logger.error(
                    f"Invalid filter-reference [{filter_ref}] in REFERENCE_DICT[{reference}][filter_by]."
                )
                logger.error(
                    f"Expected filter-reference names: {TRAINING_LOG_HEADERS}."
                )
                exit(1)

            # check: filter_func_name is valid
            if filter_func_name == "max":
                filter_func = max
            elif filter_func_name == "min":
                filter_func = min
            else:
                logger.error(
                    f"Invalid filter-function name [{filter_func_name}] in REFERENCE_DICT[{reference}][filter_by]."
                )
                logger.error("Expected filter-function names: [min, max].")
                exit(1)

            # filter by reference
            log_df = log_df[
                log_df[filter_ref] == filter_func(log_df[filter_ref].to_list())
            ]

            # return when only one row is left
            if len(log_df.index) == 1:
                log_dict = log_df.to_dict("records")[0]
                log_dict["TRAINING LOG"] = training_log.as_posix()
                return log_dict

        logger.error("Unable to filter-down to a single row dataframe:")
        logger.error(f"\n{log_df}")
        exit(1)

    def calc_cross_validation_performance(self):
        pass

    def find_best_cross_validation(self):
        pass
