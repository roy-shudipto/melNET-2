import click
import pathlib
from loguru import logger

from analyzer.defaults import REFERENCE_DICT
from analyzer.performance_analyzer import PerformaceAnalyzer


# COMMAND: plot-training-performance
@click.command()
@click.option(
    "--training_log",
    type=str,
    required=True,
    help="Path to a training-log [.csv].",
)
@click.option(
    "--reference",
    type=click.Choice(list(REFERENCE_DICT.keys())),
    multiple=True,
    help="Metric-reference for plotting. Multiple choice is allowed.",
)
def plot_training_performance(training_log, reference):
    logger.info("melNET analyzer: Plot performance of a training-run.")
    PerformaceAnalyzer().plot_training_performance(
        pathlib.Path(training_log), list(reference)
    )


# COMMAND: find-best-epoch
@click.command()
@click.option(
    "--training_log",
    type=str,
    required=True,
    help="Path to a training-log [.csv].",
)
@click.option(
    "--reference",
    type=click.Choice(REFERENCE_DICT.keys()),
    multiple=False,
    help="Metric-reference for evaluation. Multiple choice is NOT allowed.",
)
def find_best_epoch(training_log, reference):
    logger.info("melNET analyzer: Find the best epoch of a training-run.")
    best_epoch_dict = PerformaceAnalyzer().find_best_epoch(
        pathlib.Path(training_log), reference
    )

    for header, val in best_epoch_dict.items():
        logger.info(f"{header}: {val}")


# COMMAND: calc-cross-validation-performance
@click.command()
@click.option(
    "--root",
    type=str,
    required=True,
    help="Path to a training-run directory.",
)
@click.option(
    "--reference",
    type=click.Choice(REFERENCE_DICT.keys()),
    multiple=False,
    help="Metric-reference for evaluation. Multiple choice is NOT allowed.",
)
def calc_cross_validation_performance():
    logger.info("melNET analyzer: Calculate cross-validation performace.")
    PerformaceAnalyzer().calc_cross_validation_performance()


# COMMAND: find-best-cross-validation
@click.command()
@click.option(
    "--root",
    type=str,
    required=True,
    help="Path to a root-directory with multiple training-runs.",
)
@click.option(
    "--reference",
    type=click.Choice(REFERENCE_DICT.keys()),
    multiple=False,
    help="Metric-reference for evaluation. Multiple choice is NOT allowed.",
)
def find_best_cross_validation():
    logger.info("melNET analyzer: Find the best performing cross-validation.")
    PerformaceAnalyzer().find_best_cross_validation()
