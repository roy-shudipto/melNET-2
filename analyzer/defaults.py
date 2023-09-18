REFERENCE_DICT = {
    "TRAIN LOSS": {
        "plt_style": "ro--",
        "filter_by": {"TRAIN LOSS": "min", "EPOCH": "max"},
    },
    "TRAIN ACCURACY": {
        "plt_style": "go--",
        "filter_by": {"TRAIN ACCURACY": "max", "EPOCH": "max"},
    },
    "TRAIN PRECISION": {
        "plt_style": "co--",
        "filter_by": {"TRAIN PRECISION": "max", "EPOCH": "max"},
    },
    "TRAIN RECALL": {
        "plt_style": "mo--",
        "filter_by": {"TRAIN RECALL": "max", "EPOCH": "max"},
    },
    "TRAIN SPECIFICITY": {
        "plt_style": "yo--",
        "filter_by": {"TRAIN SPECIFICITY": "max", "EPOCH": "max"},
    },
    "TRAIN F1_SCORE": {
        "plt_style": "bo--",
        "filter_by": {"TRAIN F1_SCORE": "max", "EPOCH": "max"},
    },
    "VAL LOSS": {
        "plt_style": "ro-",
        "filter_by": {"VAL LOSS": "min", "EPOCH": "max"},
    },
    "VAL ACCURACY": {
        "plt_style": "go-",
        "filter_by": {"VAL ACCURACY": "max", "EPOCH": "max"},
    },
    "VAL PRECISION": {
        "plt_style": "co-",
        "filter_by": {"VAL PRECISION": "max", "EPOCH": "max"},
    },
    "VAL RECALL": {
        "plt_style": "mo-",
        "filter_by": {"VAL RECALL": "max", "EPOCH": "max"},
    },
    "VAL SPECIFICITY": {
        "plt_style": "yo-",
        "filter_by": {"VAL SPECIFICITY": "max", "EPOCH": "max"},
    },
    "VAL F1_SCORE": {
        "plt_style": "bo-",
        "filter_by": {"VAL F1_SCORE": "max", "EPOCH": "max"},
    },
}
X_AXIS_HEADER = "EPOCH"
TRAINING_LOG_HEADERS = [X_AXIS_HEADER] + list(REFERENCE_DICT.keys())
TRAINING_LOG_EXTENSION = ".csv"
