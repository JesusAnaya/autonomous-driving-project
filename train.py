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

LOGDIR = './save'


def save_model(model):
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    checkpoint_path = os.path.join(LOGDIR, config.model_path)
    if config.device == 'cuda':
        model.to('cpu')
    torch.save(model.state_dict(), checkpoint_path)

    if config.device == 'cuda':
        model.to('cuda')
    print("Model saved in file: %s" % checkpoint_path)
    

def validation():
    try:
        xs, ys = next(test_loader)
    except:
        test_loader = iter(val_subset_loader)
        xs, ys = next(test_loader)

    xs = xs.to(config.device)
    ys = ys.to(config.device)

    with torch.no_grad():
        loss_value = loss_function(model(xs), ys).item()
    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * config.batch_size + batch_idx, loss_value))


def main():
    # train over the dataset about 30 times
    train_subset_loader, val_subset_loader = dataset_loader.get_data_subsets_loaders()
    test_loader = iter(val_subset_loader)
    num_images = len(train_subset_loader.dataset) + len(val_subset_loader.dataset)
    
    model = NvidiaModel()
    model.to(config.device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Loss function using MSE
    loss_function = nn.MSELoss()

    # change model in training mood
    model.train()

    # to get batch loss
    batch_loss = np.array([])
    batch_loss_mean = np.array([])
    
    for epoch in range(config.epochs_count):
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
            loss = loss_function(y_pred, target)
        
            loss.backward()
            optimizer.step()

            batch_loss = np.append(batch_loss, [loss.item()])

            if batch_idx % 10 == 0:
                epoch_loss = batch_loss.mean()
                batch_loss_mean = np.append(batch_loss_mean, [epoch_loss])
                print(f'Epoch: {epoch} Batch {batch_idx} \nTrain Loss: {epoch_loss:.6f}')

        save_model(model)

    loss_acc_df = pd.DataFrame({"loss": batch_loss_mean})
    loss_acc_df.to_csv("loss_acc_results.csv", index=None)
    print("loss_acc_results.csv saved!")


if __name__ == '__main__':
    main()
