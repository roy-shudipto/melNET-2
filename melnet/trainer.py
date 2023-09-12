import copy
import torch
from loguru import logger

from melnet.log import TrainingLog
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
        write_checkpoint,
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
        self.write_checkpoint = write_checkpoint

    def run(self) -> None:
        # initiate training-logging
        training_log = TrainingLog()

        # load model to the device
        model = self.model.to(self.device)

        # variables to find the best checkpoint (in validation)
        best_epoch = None
        best_metric = None
        best_model_state_dict = None

        # iterate over each epoch
        for epoch_idx in range(self.epochs):
            # count epoch from 1
            epoch = epoch_idx + 1

            # run epoch
            metric, model_state_dict = self._run_epoch(model, epoch)

            # deep copy the model if epoch accuracy (in validation) improves
            if (
                best_metric is None
                or metric.accuracy["val"] >= best_metric.accuracy["val"]
            ):
                best_metric = metric
                best_model_state_dict = copy.deepcopy(model_state_dict)
                best_epoch = epoch

            # update training-log
            training_log.update(epoch=epoch, metric=metric)

        # log best performance
        logger.info(
            f"Best Performace: Epoch={best_epoch}, "
            + f"Loss={best_metric.loss['val']:.2f}, "
            + f"Accuracy={best_metric.accuracy['val']:.2f}"
        )

        # save the best checkpoint
        if self.write_checkpoint:
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_model_state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metric": best_metric,
                },
                self.checkpoint_path,
            )
            logger.info(f"Checkpoint is saved as: {self.checkpoint_path}")

        # save training-log
        training_log.save(path=self.log_path)
        logger.info(f"Training-log is saved as: {self.log_path}")

    def _run_epoch(self, model, epoch) -> (Metric, dict):
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

            # update scheduler
            if phase == "train":
                self.scheduler.step()

            # calculate epoch perfomance for the current phase
            metric.calc_score(phase=phase)

        # # log epoch performance
        # logger.info(
        #     f"Epoch {str(epoch).zfill(len(str(self.epochs)))}/{self.epochs}: "
        #     + f"Train Loss={metric.loss['train']:.2f}, "
        #     + f"Train Accuracy={metric.accuracy['train']:.2f}, "
        #     + f"Val Loss={metric.loss['val']:.2f}, "
        #     + f"Val Accuracy={metric.accuracy['val']:.2f}"
        # )

        return metric, model.state_dict()
