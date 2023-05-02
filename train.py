import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet34
from torch.utils.tensorboard import SummaryWriter
import dataset_loader
import numpy as np
import pandas as pd
from model import NvidiaModel, NvidiaModelTransferLearning
from config import config
import argparse

parser = argparse.ArgumentParser(description="Compare loss values from two CSV files.")
parser.add_argument("--dataset_type", type=str, help="Dataset type", choices=['sully', 'udacity'], default='sully')
parser.add_argument("--batch_size", type=int, help="Batch size", default=50)
parser.add_argument("--epochs_count", type=int, help="Epochs count", default=60)


def save_model(model, log_dir="./save"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_path = os.path.join(log_dir, config.model_path)
    if config.device == 'cuda':
        model.to('cpu')
    torch.save(model.state_dict(), checkpoint_path)

    if config.device == 'cuda':
        model.to('cuda')
    print("Model saved in file: %s" % checkpoint_path)
    


def validation(model, val_subset_loader, loss_function):
    # Load model
    model.eval()
    batch_loss = np.array([])

    for data_val, target_val in val_subset_loader:
        # send data to device (its is medatory if GPU has to be used)
        data_val = data_val.to(config.device)

        # send target to device
        target_val = target_val.to(config.device)
        
        with torch.no_grad():
            # forward pass to the model
            y_pred_val = model(data_val)

            # cross entropy loss
            loss = loss_function(y_pred_val.float(), target_val.float())

        # Capture log
        batch_loss = np.append(batch_loss, [loss.item()])

    epoch_loss = batch_loss.mean()    
    print(f'Validation Loss: {epoch_loss:.9f}')
    return epoch_loss

            
def main():
    args = parser.parse_args()
    model = NvidiaModel()
    model.to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    loss_function = nn.MSELoss()

    batch_loss = np.array([])
    batch_loss_mean = np.array([])
    batch_val_loss = np.array([])
    
    start_time = time.time()

    datasets_types = ["sully"]

    train_subset_loader, val_subset_loader = dataset_loader.get_data_subsets_loaders(
        dataset_types=datasets_types,
        batch_size=args.batch_size
    )

    print("Training with a train set of size: ", len(train_subset_loader.dataset))
    print("Training with a validation set of size: ", len(val_subset_loader.dataset))

    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir='./logs/tensorboard/')

    for epoch in range(args.epochs_count):
        model.train()

        for batch_idx, (data, target) in enumerate(train_subset_loader):
            data = data.to(config.device)
            target = target.to(config.device)

            optimizer.zero_grad()

            y_pred = model(data)
            
            loss = loss_function(y_pred.float(), target.float())
        
            loss.backward()
            optimizer.step()

            batch_loss = np.append(batch_loss, [loss.item()])

            if batch_idx % 10 == 0 and batch_idx > 0:
                epoch_loss = batch_loss.mean()
                batch_loss_mean = np.append(batch_loss_mean, [epoch_loss])
                print(f'Epoch: {epoch+1}/{args.epochs_count} Batch {batch_idx} \nTrain Loss: {epoch_loss:.9f}')

                # Log the training loss to TensorBoard
                writer.add_scalar('Loss/train', epoch_loss, epoch * len(train_subset_loader) + batch_idx)

            batch_idx += 1

        scheduler.step()

        # Log learning rate
        #lr = optimizer.param_groups[0]['lr']
        #writer.add_scalar('learning_rate', lr, epoch * len(train_subset_loader))


        val_loss_mean = validation(model, val_subset_loader, loss_function)
        batch_val_loss = np.append(batch_val_loss, [val_loss_mean.item()])
        
        # Log the validation loss to TensorBoard
        writer.add_scalar('Loss/validation', val_loss_mean, epoch)

        save_model(model)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds")
    print("training finished")

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
