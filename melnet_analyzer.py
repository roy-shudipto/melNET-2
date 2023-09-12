import pandas as pd
import pathlib
from argparse import ArgumentParser
from loguru import logger


if __name__ == "__main__":
    logger.info("Analyzing melNET performance.")

    parser = ArgumentParser(description=("Analyzes melNET performace."))

    parser.add_argument(
        "--checkpoint_dir",
        nargs="+",
        required=True,
        help="Path(s) to checkpoint directory for analyzing performace.",
    )

    parser.add_argument(
        "--reference",
        required=True,
        choices=["ACCURACY", "PRECISION", "RECALL", "SPECIFICITY", "F1_SCORE"],
        help="Reference for analysis.",
    )

    args = parser.parse_args()

    # analyze performance on Val-scores
    ref = f"VAL {args.reference}"

    # looping over checkpoint direcotries
    for checkpoint_path in args.checkpoint_dir:
        # convert to pathlib.Path
        checkpoint_path = pathlib.Path(pathlib.Path(checkpoint_path))

        # check: checkpoint-path is a directory
        if not checkpoint_path.is_dir():
            logger.debug(f"Skipping path: {checkpoint_path}. Path is not a directory.")
            continue

        # check: checkpoint-path contains [checkpoint] in name
        if checkpoint_path.name.find("checkpoint") < 0:
            logger.debug(
                f"Skipping path: {checkpoint_path}. Path does not contain [checkpoint] in name."
            )
            continue

        # looping over log files of each checkpoint directory
        for log_file in checkpoint_path.iterdir():
            # check: log_file has .csv suffix
            if log_file.suffix.lower() != ".csv":
                continue

            # check: log_file has [log] in name
            if log_file.name.find("log") < 0:
                continue

            # read log
            log_df = pd.read_csv(log_file)
            print(log_file)
            print(log_df.columns)
            print(log_df[ref])

            exit()
