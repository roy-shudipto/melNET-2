import click
from loguru import logger

from analyzer.commands import (
    plot_training_performance,
    find_best_epoch,
    calc_cross_validation_performance,
    find_best_cross_validation,
)


@click.group()
def melnet_analyzer():
    logger.info("Running melNET-analyzer.")


# plot performance of a training-run; given: training-log, List[reference]
melnet_analyzer.add_command(plot_training_performance)

# find the best epoch of a training-run; given: training-log, reference
melnet_analyzer.add_command(find_best_epoch)

# calculate cross-validation performace; given: List[training-log], reference
melnet_analyzer.add_command(calc_cross_validation_performance)

# find the best performing cross-validation; given: List[List[training-log]], reference
melnet_analyzer.add_command(find_best_cross_validation)

if __name__ == "__main__":
    melnet_analyzer()
