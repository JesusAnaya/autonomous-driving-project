import torch
from dataclasses import dataclass


@dataclass
class Config(object):
    batch_size = 100
    num_workers = 8
    shuffle = True
    train_split_size = 0.75
    test_split_size = 0.25
    resize = (66, 200)
    mean = [0.4322, 0.4853, 0.4998]
    std = [0.2499, 0.2696, 0.2984]
    epochs_count = 50
    learning_rate = 0.001
    momentum = 0.9
    log_interval = 10
    save_model = True
    root_path = "./"
    model_path = "model.pt"
    optimizer_path = "optimizer.pt"
    loss_path = "loss.pt"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    test_interval = 1


config = Config()
