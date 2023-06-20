import torch
from dataclasses import dataclass
import os


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class Config(object):
    dataset_type = "udacity"
    batch_size = 128
    num_workers = 8
    shuffle = True
    train_split_size = 0.80
    test_split_size = 0.20
    resize = (66, 200)
    epochs_count = 60
    learning_rate = 1e-4
    weight_decay = 1e-5
    momentum = 0.9
    log_interval = 10
    save_model = True
    root_path = "./"
    model_path = "model.pt"
    optimizer_path = "optimizer.pt"
    loss_path = "loss.pt"
    device = get_device()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    epsilon = 0.001
    early_stopping_patience = 15
    early_stopping_min_delta = 0.005
    
    scheduler_milestones = [40, 60]  # Decrease the learning rate every 30 epochs
    scheduler_step_size = 70        # Decrease the learning rate every 30 epochs
    scheduler_gamma = 0.1     # Multiply the learning rate by 0.1 when decreasing

    # loggers
    is_loss_logging_enabled = True
    is_image_logging_enabled = False
    is_learning_rate_logging_enabled = False
    is_grad_avg_logging_enabled = False

config = Config()
