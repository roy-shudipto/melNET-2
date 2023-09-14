import pandas as pd
import pathlib
import shutil
from argparse import ArgumentParser
from loguru import logger

# GLOBAL VARIABLES FOR HAM10000
HAM_SUBDIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
HAM_METADATA = "HAM10000_metadata"
HAM_IMAGE_EXTENSION = ".jpg"
HAM_CLASS_MAP = {
    "mel": "melanoma",
    "nv": "not_melanoma",
}

if __name__ == "__main__":
    logger.info("Starting to generate classification-dataset from HAM10000.")

    parser = ArgumentParser(
        description=("Generates classification-dataset from HAM10000.")
    )

    parser.add_argument(
        "--ham_root",
        type=str,
        required=True,
        help="Path to HAM10000 dataset. All zipped folders need to be un-zipped."
        + "\nDownload link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root to classification dataset.",
    )

    args = parser.parse_args()
    ham_root = pathlib.Path(args.ham_root)
    output_root = pathlib.Path(args.output_root)

    # check: HAM10000 subdirectories exist
    for sub_dir in HAM_SUBDIRS:
        if (ham_root / sub_dir).exists() and (ham_root / sub_dir).is_dir():
            continue
        logger.error(f"Unable to find sub-directory: {(ham_root / sub_dir)}")
        exit(1)

    # check: HAM10000 metadata exists
    if (
        not (ham_root / HAM_METADATA).exists()
        or not (ham_root / HAM_METADATA).is_file()
    ):
        logger.error(f"Unable to find metadata-file: {(ham_root / HAM_METADATA)}")
        exit(1)

    # remove existing root
    if output_root.exists():
        logger.debug(f"Removing existing root: [{output_root}]")
        shutil.rmtree(output_root)

    # create dataset-directories and initiate a class-counter
    counter = {}
    for ham_class in HAM_CLASS_MAP.keys():
        dataset_dir = output_root / HAM_CLASS_MAP[ham_class]

        if dataset_dir.exists():
            continue

        # create dataset-directories
        dataset_dir.mkdir(parents=True)
        logger.info(f"Created dataset-directory: {dataset_dir}")

        # initiate class-counter
        counter[HAM_CLASS_MAP[ham_class]] = 0

    # read HAM10000 metadata
    try:
        ham_df = pd.read_csv(ham_root / HAM_METADATA)
    except RuntimeError:
        logger.error(f"Unable to read metadata-file: {(ham_root / HAM_METADATA)}")
        exit(1)

    # create classification-dataset
    logger.info("Generating classification-dataset ...")
    for _, row in ham_df.iterrows():
        # get write-path
        if row["dx"] in HAM_CLASS_MAP.keys():
            write_path = (
                output_root / HAM_CLASS_MAP[row["dx"]] / row["image_id"]
            ).with_suffix(HAM_IMAGE_EXTENSION)
        else:
            continue

        # get read-path
        read_path = None
        for sub_dir in HAM_SUBDIRS:
            image_path = (ham_root / sub_dir / row["image_id"]).with_suffix(
                HAM_IMAGE_EXTENSION
            )
            if image_path.exists():
                read_path = image_path
                break
        if not read_path:
            logger.error(f"Unable to find image with image-id: {row['image_id']}")
            exit(1)

        # copy image from HAM10000 to classification-dataset
        shutil.copy(read_path, write_path)

        # update counter
        counter[HAM_CLASS_MAP[row["dx"]]] += 1

    # display class-count
    logger.info("Classification-dataset counter:")
    for key, val in counter.items():
        logger.info(f"{key}: {val}")
