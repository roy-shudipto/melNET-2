import numpy as np
from loguru import logger
from sklearn.metrics import confusion_matrix


class Metric:
    def __init__(self) -> None:
        self.total_loss = {"train": 0, "val": 0}
        self.confusion_mat = {
            "train": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "val": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        }

        self.loss = {"train": None, "val": None}
        self.accuracy = {"train": None, "val": None}
        self.precision = {"train": None, "val": None}
        self.sensitivity = {"train": None, "val": None}
        self.specificity = {"train": None, "val": None}

    @classmethod
    def calc_confusion_matrix(cls, y_true, y_pred):
        # returning order: tn, fp, fn, tp
        # sklearn.metrics->confusion_matrix seems to have a bug.
        # it returns [[batch_size]] in the case of all TNs and all TPs.

        # calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred).ravel().tolist()

        # case: tn, fp, fn, tp are available
        if len(cm) == 4:
            cm = [int(i) for i in cm]
            return tuple(cm)

        # case: either tn or tp is available
        elif len(cm) == 1:
            # case: all true-negatives
            if np.sum(y_true) == np.sum(y_pred) == 0:
                return len(y_true), 0, 0, 0

            # case: all true-positives
            elif np.sum(y_true) == np.sum(y_pred) == len(y_true):
                return 0, 0, 0, len(y_true)

        logger.error(f"Invalid confusion matrix for y_true: {y_true}, y_pred: {y_pred}")
        logger.error(f"Calculated confusion matrix: {cm}")
        exit(1)

    def update(self, *, phase, loss, y_true, y_pred):
        # update loss
        self.total_loss[phase] += loss

        # calculate confusion matrix
        tn, fp, fn, tp = self.calc_confusion_matrix(y_true, y_pred)

        # update confusion matrix
        self.confusion_mat[phase]["tp"] += tp
        self.confusion_mat[phase]["tn"] += tn
        self.confusion_mat[phase]["fp"] += fp
        self.confusion_mat[phase]["fn"] += fn

    def calc_score(self, phase):
        # calculate average-loss
        self.loss[phase] = self.total_loss[phase] / (
            self.confusion_mat[phase]["tp"]
            + self.confusion_mat[phase]["tn"]
            + self.confusion_mat[phase]["fp"]
            + self.confusion_mat[phase]["fn"]
        )
        self.loss[phase] = round(self.loss[phase], 2)

        # calculate accuracy
        self.accuracy[phase] = (
            self.confusion_mat[phase]["tp"] + self.confusion_mat[phase]["tn"]
        ) / (
            self.confusion_mat[phase]["tp"]
            + self.confusion_mat[phase]["tn"]
            + self.confusion_mat[phase]["fp"]
            + self.confusion_mat[phase]["fn"]
        )
        self.accuracy[phase] = round(self.accuracy[phase] * 100.0, 2)

        # calculate precision
        try:
            self.precision[phase] = self.confusion_mat[phase]["tp"] / (
                self.confusion_mat[phase]["tp"] + self.confusion_mat[phase]["fp"]
            )
            self.precision[phase] = round(self.precision[phase], 2)
        except ZeroDivisionError:
            self.precision[phase] = 0.0

        # calculate sensitivity or, recall
        try:
            self.sensitivity[phase] = self.confusion_mat[phase]["tp"] / (
                self.confusion_mat[phase]["tp"] + self.confusion_mat[phase]["fn"]
            )
            self.sensitivity[phase] = round(self.sensitivity[phase], 2)
        except ZeroDivisionError:
            self.sensitivity[phase] = 0.0

        # calculate specificity
        try:
            self.specificity[phase] = self.confusion_mat[phase]["tn"] / (
                +self.confusion_mat[phase]["tn"] + self.confusion_mat[phase]["fp"]
            )
            self.specificity[phase] = round(self.specificity[phase], 2)
        except ZeroDivisionError:
            self.specificity[phase] = 0.0
