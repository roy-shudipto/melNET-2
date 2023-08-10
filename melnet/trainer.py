import copy
import pandas as pd
import torch
from loguru import logger

from melnet.defaults import LOG_HEADERS
from melnet.metric import Metric


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
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.dataloaders = {"train": train_dataloader, "val": val_dataloader}
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path

    def run(self):
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
                running_loss = 0.0
                running_corrects = 0
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

                    # iteration statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # epoch statistics
                epoch_loss = running_loss / len(self.dataloaders[phase])
                epoch_acc = (
                    running_corrects.double() / len(self.dataloaders[phase])
                ).item()

                if phase == "train":
                    # update scheduler
                    self.scheduler.step()

                    # store result in metric
                    metric.train_loss = epoch_loss
                    metric.train_acc = epoch_acc
                else:
                    # deep copy the model if epoch accuracy improves
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_loss = epoch_loss
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = epoch

                    # store result in metric
                    metric.val_loss = epoch_loss
                    metric.val_acc = epoch_acc

            # update log
            df_log = pd.concat(
                [
                    df_log,
                    pd.DataFrame(
                        {
                            "EPOCH": [epoch],
                            "TRAIN LOSS": [metric.train_loss],
                            "TRAIN ACCURACY": [metric.train_acc],
                            "VAL LOSS": [metric.val_loss],
                            "VAL ACCURACY": [metric.val_acc],
                        }
                    ),
                ],
                axis=0,
            )

            logger.info(
                f"Epoch {epoch}/{self.epochs}: "
                + f"Train Loss={metric.train_loss:.2f}, "
                + f"Train Accuracy={metric.train_acc:.2f}, "
                + f"Val Loss={metric.val_loss:.2f}, "
                + f"Val Accuracy={metric.val_acc:.2f}"
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
