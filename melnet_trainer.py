import click
import pathlib
from loguru import logger

from melnet.config import TrainingConfig
from melnet.dataset import ClassificationDatasetFolds
from melnet.model import (
    get_model,
    get_device,
    get_optimizer,
    get_scheduler,
    get_loss_function,
)
from melnet.trainer import Trainer


def train(training_config: dict) -> None:
    # create a directory for checkpoint (if needed)
    if not training_config.checkpoint_directory.exists():
        training_config.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Successfully created checkpoint directory: {training_config.checkpoint_directory}"
        )
    else:
        logger.info(
            f"Checkpoint directory already exists: {training_config.checkpoint_directory}"
        )

    # save training-config to the checkpoint-directory
    training_config.save()
    logger.info(f"Training-config is saved as: {training_config.config_dst}")

    # get multi-fold datasets
    dataset_folds = ClassificationDatasetFolds(
        dataset_root=training_config.dataset_root,
        input_size=training_config.input_size,
        folds=training_config.cross_validation_fold,
        single_fold_split=training_config.single_fold_split,
    )

    # get model definition
    model = get_model(
        model_arch=training_config.model_architecture,
        classes=dataset_folds.number_of_classes,
        load_checkpoint=training_config.load_checkpoint,
        fine_tune_fc=training_config.fine_tune_fc,
    )
    device = get_device()
    criterion = get_loss_function()
    optimizer = get_optimizer(
        training_config.optimizer,
        model,
        training_config.learning_rate,
        training_config.momentum,
    )
    scheduler = get_scheduler(
        optimizer, training_config.schedular_step, training_config.schedular_gamma
    )
    logger.info(f"Model is initialized to get trained on: {device}")

    # run training for each fold
    for fold in range(training_config.cross_validation_fold):
        # get dataloader-dictionary
        dataloader_dict = dataset_folds.get_datasets(
            fold_index=fold,
            batch_size=training_config.batch_size,
            num_worker=training_config.num_workers,
        )

        # get trainer
        trainer = Trainer(
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=training_config.epochs,
            train_dataloader=dataloader_dict["train"],
            val_dataloader=dataloader_dict["val"],
            checkpoint_path=training_config.get_checkpoint_path(fold + 1),
            log_path=training_config.get_log_path(fold + 1),
        )

        # run trainer
        logger.info(
            f"Starting training for fold: {fold + 1}/{training_config.cross_validation_fold}"
        )
        trainer.run()

    logger.info("Successfully completed the training.")


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="Path to the training-config [.yaml] file.",
)
def run_training(config) -> None:
    # convert config from str to pathlib.Path
    config = pathlib.Path(config)

    # read config file
    training_config = TrainingConfig(config)
    logger.info(f"Successfully read config-file: {config}")

    # run
    train(training_config)


if __name__ == "__main__":
    logger.info("Running model-training pipeline.")
    run_training()
