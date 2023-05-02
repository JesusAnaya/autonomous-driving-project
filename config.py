import torch
from dataclasses import dataclass
import os


@dataclass
class Config(object):
    dataset_type = "udacity"
    batch_size = 32
    num_workers = os.cpu_count() - 1
    shuffle = True
    train_split_size = 0.7
    test_split_size = 0.3
    resize = (66, 200)
    epochs_count = 30
    learning_rate = 1e-3
    weight_decay = 1e-4
    momentum = 0.9
    log_interval = 10
    save_model = True
    root_path = "./"
    model_path = "model.pt"
    optimizer_path = "optimizer.pt"
    loss_path = "loss.pt"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    test_interval = 1
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    scheduler_step_size = 10  # Decrease the learning rate every 10 epochs
    scheduler_gamma = 0.1     # Multiply the learning rate by 0.1 when decreasing


config = Config()
