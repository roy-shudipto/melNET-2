import pathlib
import shutil
from argparse import ArgumentParser
from loguru import logger

# GLOBAL VARIABLES FOR MEDNODE
MEDNODE_IMAGE_EXTENSION = ".jpg"
MEDNODE_CLASS_MAP = {
    "melanoma": "melanoma",
    "naevus": "not_melanoma",
}

if __name__ == "__main__":
    logger.info("Starting to generate classification-dataset from MEDNODE.")

    parser = ArgumentParser(
        description=("Generates classification-dataset from MEDNODE.")
    )

    parser.add_argument(
        "--mednode_root",
        type=str,
        required=True,
        help="Path to MEDNODE dataset. Downloaded zipped folder needs to be un-zipped."
        + "\nDownload link: https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root to classification dataset.",
    )

    args = parser.parse_args()
    mednode_root = pathlib.Path(args.mednode_root)
    output_root = pathlib.Path(args.output_root)

    # check: MEDNODE subdirectories exist
    for mednode_class in MEDNODE_CLASS_MAP.keys():
        if (mednode_root / mednode_class).exists() and (
            mednode_root / mednode_class
        ).is_dir():
            continue
        logger.error(f"Unable to find sub-directory: {(mednode_root / mednode_class)}")
        exit(1)

    # remove existing root
    if output_root.exists():
        logger.debug(f"Removing existing root: [{output_root}]")
        shutil.rmtree(output_root)

    # create dataset-directories and initiate a class-counter
    counter = {}
    for mednode_class in MEDNODE_CLASS_MAP.keys():
        dataset_dir = output_root / MEDNODE_CLASS_MAP[mednode_class]

        if dataset_dir.exists():
            continue

        # create dataset-directories
        dataset_dir.mkdir(parents=True)
        logger.info(f"Created dataset-directory: {dataset_dir}")

        # initiate class-counter
        counter[MEDNODE_CLASS_MAP[mednode_class]] = 0

    # create classification-dataset
    logger.info("Generating classification-dataset ...")
    for mednode_class in MEDNODE_CLASS_MAP.keys():
        for read_path in (mednode_root / mednode_class).iterdir():
            if not read_path.with_suffix(MEDNODE_IMAGE_EXTENSION):
                continue

            # get write-path
            write_path = output_root / MEDNODE_CLASS_MAP[mednode_class] / read_path.name

            # copy image from MEDNODE to classification-dataset
            shutil.copy(read_path, write_path)

            # update counter
            counter[MEDNODE_CLASS_MAP[mednode_class]] += 1

    # display class-count
    logger.info("Classification-dataset counter:")
    for key, val in counter.items():
        logger.info(f"{key}: {val}")
