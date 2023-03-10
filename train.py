import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dataset_loader
import numpy as np
import pandas as pd
from model import NvidiaModel
from config import config


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
    print(f'Validation Loss: {epoch_loss:.6f}')
    return epoch_loss

            
def main():
    # train over the dataset about 30 times
    train_subset_loader, val_subset_loader = dataset_loader.get_data_subsets_loaders()
    test_loader = iter(val_subset_loader)
    num_images = len(train_subset_loader.dataset) + len(val_subset_loader.dataset)
    
    model = NvidiaModel()
    model.to(config.device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Loss function using MSE
    loss_function = nn.MSELoss()

    # to get batch loss
    batch_loss = np.array([])
    batch_loss_mean = np.array([])
    batch_val_loss = np.array([])
    
    for epoch in range(config.epochs_count):
        # change model in training mood
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_subset_loader):
            # send data to device (its is medatory if GPU has to be used)
            data = data.to(config.device)
            
            # send target to device
            target = target.to(config.device)

            # reset parameters gradient to zero
            optimizer.zero_grad()

            # forward pass to the model
            y_pred = model(data)
            
            # cross entropy loss
            loss = loss_function(y_pred.float(), target.float())
        
            loss.backward()
            optimizer.step()

            batch_loss = np.append(batch_loss, [loss.item()])

            if batch_idx % 10 == 0:
                epoch_loss = batch_loss.mean()
                batch_loss_mean = np.append(batch_loss_mean, [epoch_loss])
                print(f'Epoch: {epoch+1}/{config.epochs_count} Batch {batch_idx} \nTrain Loss: {epoch_loss:.6f}')
        
        val_loss_mean = validation(model, val_subset_loader, loss_function)
        batch_val_loss = np.append(batch_val_loss, [val_loss_mean.item()])
        save_model(model)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    pd.DataFrame({"loss": batch_loss_mean}).to_csv("logs/loss_acc_results.csv", index=None)
    pd.DataFrame({"val_loss": batch_val_loss}).to_csv("logs/loss_acc_validation.csv", index=None)
    print("loss_acc_results.csv saved!")


if __name__ == '__main__':
    main()
