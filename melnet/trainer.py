import copy
import pandas as pd
import torch
from loguru import logger

from melnet.defaults import LOG_HEADERS
from melnet.metric import Metric
from melnet.utils import gpu2cpu


class Trainer:
    def __init__(
        self,
        *,
        model,
        device,
        criterion,
        optimizer,
        scheduler,
        epochs,
        train_dataloader,
        val_dataloader,
        checkpoint_path,
        log_path,
    ) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path

    def run(self) -> None:
        # load model to the device
        model = self.model.to(self.device)

        # create a dataframe for logging
        df_log = pd.DataFrame(columns=LOG_HEADERS)

        # variables to find the best checkpoints (in validation)
        best_acc = 0.0
        best_loss = None
        best_epoch = None
        best_model_state_dict = None

        # iterate over each epoch
        for epoch_idx in range(self.epochs):
            # count epoch from 1
            epoch = epoch_idx + 1

            # track epoch performance
            metric = Metric()

            # each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # set model to training mode
                else:
                    model.eval()  # set model to evaluate mode

                # iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # forward (track history if only in train)
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    # update metric
                    metric.update(
                        phase=phase,
                        loss=loss.item() * inputs.size(0),
                        y_true=gpu2cpu(labels),
                        y_pred=gpu2cpu(preds),
                    )

                # epoch statistics
                metric.calc_score(phase=phase)

                if phase == "train":
                    # update scheduler
                    self.scheduler.step()

                else:
                    # deep copy the model if epoch accuracy (in validation) improves
                    if metric.accuracy[phase] >= best_acc:
                        best_acc = metric.accuracy[phase]
                        best_loss = metric.loss[phase]
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = epoch

            # update log
            df_log = pd.concat(
                [
                    df_log,
                    pd.DataFrame(
                        {
                            "EPOCH": [epoch],
                            "TRAIN LOSS": [metric.loss["train"]],
                            "TRAIN ACCURACY": [metric.accuracy["train"]],
                            "TRAIN PRECISION": [metric.precision["train"]],
                            "TRAIN SENSITIVITY": [metric.sensitivity["train"]],
                            "TRAIN SPECIFICITY": [metric.specificity["train"]],
                            "TRAIN CM": [metric.confusion_mat["train"]],
                            "VAL LOSS": [metric.loss["val"]],
                            "VAL ACCURACY": [metric.accuracy["val"]],
                            "VAL PRECISION": [metric.precision["val"]],
                            "VAL SENSITIVITY": [metric.sensitivity["val"]],
                            "VAL SPECIFICITY": [metric.specificity["val"]],
                            "VAL CM": [metric.confusion_mat["val"]],
                        }
                    ),
                ],
                axis=0,
            )

            # display epoch performance
            logger.info(
                f"Epoch {str(epoch).zfill(len(str(self.epochs)))}/{self.epochs}: "
                + f"Train Loss={metric.loss['train']:.2f}, "
                + f"Train Accuracy={metric.accuracy['train']:.2f}, "
                + f"Val Loss={metric.loss['val']:.2f}, "
                + f"Val Accuracy={metric.accuracy['val']:.2f}"
            )

        # best performance
        logger.info(
            f"Best Performace: Epoch={best_epoch}, Loss={best_loss:.2f}, Accuracy={best_acc:.2f}"
        )

        # save the best checkpoint
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": best_loss,
            },
            self.checkpoint_path,
        )
        logger.info(f"Checkpoint is saved as: {self.checkpoint_path}")

        # save log
        df_log.to_csv(self.log_path, index=False)
        logger.info(f"Log is saved as: {self.log_path}")
