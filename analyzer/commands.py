import click
from loguru import logger

from analyzer.performance_analyzer import PerformaceAnalyzer


@click.command()
def plot_training_performance():
    logger.info("melNET analyzer: Plot performance of a training-run.")
    PerformaceAnalyzer.plot_training_performance()


@click.command()
def find_best_epoch():
    logger.info("melNET analyzer: Find the best epoch of a training-run.")
    PerformaceAnalyzer.find_best_epoch()


@click.command()
def calc_cross_validation_performance():
    logger.info("melNET analyzer: Calculate cross-validation performace.")
    PerformaceAnalyzer.calc_cross_validation_performance()


@click.command()
def find_best_cross_validation():
    logger.info("melNET analyzer: Find the best performing cross-validation.")
    PerformaceAnalyzer.find_best_cross_validation()
