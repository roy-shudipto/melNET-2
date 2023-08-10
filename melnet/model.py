import sys
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim import lr_scheduler
from torchvision import models


# model
def get_model(model_arch, classes, load_checkpoint, fine_tune_fc):
    # load a pretrained model
    if model_arch.lower() == "resnet152":
        model = models.resnet152(weights="ResNet152_Weights.DEFAULT")
    elif model_arch.lower() == "resnet101":
        model = models.resnet101(weights="ResNet101_Weights.DEFAULT")
    elif model_arch.lower() == "resnet50":
        model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    else:
        sys.exit(f"[MODEL: {model_arch}] is not supported by PyTorch.")

    # reset final layer as the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)

    # load checkpoint
    if load_checkpoint is not None:
        try:
            checkpoint = torch.load(load_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Successfully loaded checkpoint from: {load_checkpoint}")
        except:
            logger.error(f"Unable to load checkpoint from: {load_checkpoint}")
            exit(1)

    # fine-tune FC-layers only
    if fine_tune_fc is True:
        for name, para in model.named_parameters():
            if name.find("fc") < 0:
                para.requires_grad = False
        logger.info("Fine-tuning only fully-connected layers.")

    return model


# device
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loss-function
def get_loss_function():
    return nn.CrossEntropyLoss()


# optimizer
def get_optimizer(optimizer_name, model, lr, momentum=None):
    # define optimizer
    if optimizer_name.upper() == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name.upper() == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        sys.exit(f"[OPTIMIZER: {optimizer_name}] is not supported by PyTorch.")

    return optimizer


# lr-scheduler
def get_scheduler(optimizer, step, gamma):
    return lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
