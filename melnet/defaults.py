MODEL_LIST = ["RESNET18", "RESNET50", "RESNET101", "RESNET152"]
OPTIMIZER_LIST = ["ADAM", "SGD"]
CHECKPOINT_EXTENSION = ".pt"
CONFIG_EXTENSION = ".yaml"
CONFIG_OUTPUT_NAME = "training_config.yaml"
LOG_EXTENSION = ".csv"
LOG_HEADERS = [
    "EPOCH",
    "TRAIN LOSS",
    "TRAIN ACCURACY",
    "TRAIN PRECISION",
    "TRAIN SENSITIVITY",
    "TRAIN SPECIFICITY",
    "TRAIN CONFUSION MAT",
    "VAL LOSS",
    "VAL ACCURACY",
    "VAL PRECISION",
    "VAL SENSITIVITY",
    "VAL SPECIFICITY",
    "VAL CONFUSION MAT",
]
CLASS_MAP = {
    "not_melanoma": 0,
    "melanoma": 1,
}
