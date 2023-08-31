import pandas as pd

from melnet.defaults import LOG_HEADERS


class TrainingLog:
    def __init__(self):
        self.log = pd.DataFrame(columns=LOG_HEADERS)

    def update(self, *, epoch, metric):
        self.log = pd.concat(
            [
                self.log,
                pd.DataFrame(
                    {
                        "EPOCH": [epoch],
                        "TRAIN LOSS": [metric.loss["train"]],
                        "TRAIN ACCURACY": [metric.accuracy["train"]],
                        "TRAIN PRECISION": [metric.precision["train"]],
                        "TRAIN SENSITIVITY": [metric.sensitivity["train"]],
                        "TRAIN SPECIFICITY": [metric.specificity["train"]],
                        "TRAIN CONFUSION MAT": [metric.confusion_mat["train"]],
                        "VAL LOSS": [metric.loss["val"]],
                        "VAL ACCURACY": [metric.accuracy["val"]],
                        "VAL PRECISION": [metric.precision["val"]],
                        "VAL SENSITIVITY": [metric.sensitivity["val"]],
                        "VAL SPECIFICITY": [metric.specificity["val"]],
                        "VAL CONFUSION MAT": [metric.confusion_mat["val"]],
                    }
                ),
            ],
            axis=0,
        )

    def save(self, path):
        self.log.to_csv(path, index=False)
