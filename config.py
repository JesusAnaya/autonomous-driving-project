import torch
from dataclasses import dataclass


@dataclass
class Config(object):
    dataset_type = "udacity"
    batch_size = 50
    num_workers = 8
    shuffle = True
    train_split_size = 0.8
    test_split_size = 0.2
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
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    test_interval = 1

    scheduler_step_size = 10  # Decrease the learning rate every 10 epochs
    scheduler_gamma = 0.1     # Multiply the learning rate by 0.1 when decreasing


config = Config()
