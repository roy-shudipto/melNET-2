import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models


# model
def get_model(model_arch, classes):
    # load a pretrained model
    if model_arch.lower() == "resnet152":
        model = models.resnet152(pretrained=True)
    elif model_arch.lower() == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_arch.lower() == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        sys.exit(f"[MODEL: {model_arch}] is not supported by PyTorch.")

    # reset final layer as the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)

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
