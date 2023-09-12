MODEL_LIST = ["RESNET18", "RESNET50", "RESNET101", "RESNET152"]
OPTIMIZER_LIST = ["ADAM", "SGD"]
FINE_TUNE_OPTIONS = ["all_layers", "fc_layers", "last_layer"]
CHECKPOINT_EXTENSION = ".pt"
CONFIG_EXTENSION = ".yaml"
CONFIG_OUTPUT_NAME = "training_config.yaml"
LOG_EXTENSION = ".csv"
LOG_HEADERS = [
    "EPOCH",
    "TRAIN LOSS",
    "TRAIN ACCURACY",
    "TRAIN PRECISION",
    "TRAIN RECALL",
    "TRAIN SPECIFICITY",
    "TRAIN F1_SCORE",
    "TRAIN CONFUSION MAT",
    "VAL LOSS",
    "VAL ACCURACY",
    "VAL PRECISION",
    "VAL RECALL",
    "VAL SPECIFICITY",
    "VAL F1_SCORE",
    "VAL CONFUSION MAT",
]
CLASS_MAP = {
    "not_melanoma": 0,
    "melanoma": 1,
}
