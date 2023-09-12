import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from loguru import logger
from torchvision import models


# model
def get_model(*, model_arch, classes, load_checkpoint, fine_tune):
    # load a pretrained model
    if model_arch.lower() == "resnet152":
        model = models.resnet152(weights="ResNet152_Weights.DEFAULT")
    elif model_arch.lower() == "resnet101":
        model = models.resnet101(weights="ResNet101_Weights.DEFAULT")
    elif model_arch.lower() == "resnet50":
        model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    elif model_arch.lower() == "resnet18":
        model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    else:
        logger.error(f"[MODEL: {model_arch}] is not supported by PyTorch.")
        exit(1)

    # load checkpoint
    if load_checkpoint is not None:
        try:
            checkpoint = torch.load(load_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Successfully loaded checkpoint from: {load_checkpoint}")
        except:
            logger.error(f"Unable to load checkpoint from: {load_checkpoint}")
            exit(1)

    # fine-tune customization
    if fine_tune == "all_layers":
        logger.info("Fine-tune: all layers.")

    elif fine_tune == "fc_layers":
        for name, param in model.named_parameters():
            if name.find("fc") < 0:
                param.requires_grad = False
        logger.info("Fine-tune: fully-connected layers.")

    elif fine_tune == "last_layer":
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Fine-tune: last layer.")

    else:
        logger.error(f"Unable to customize fine-tuning using [fine_tune={fine_tune}].")
        exit(1)

    # reset final layer as the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)

    return model


# device
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# loss-function
def get_loss_function() -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss()


# optimizer
def get_optimizer(optimizer_name, model, lr, momentum=None) -> optim.Optimizer:
    # define optimizer
    if optimizer_name.upper() == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name.upper() == "ADAM":
        return optim.Adam(model.parameters(), lr=lr)
    else:
        logger.error(f"[OPTIMIZER: {optimizer_name}] is not supported by PyTorch.")
        exit(1)


# lr-scheduler
def get_scheduler(optimizer, step, gamma) -> lr_scheduler.StepLR:
    return lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
