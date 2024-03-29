import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import make_grid
from sklearn.model_selection import KFold
import dataset_loader as dataset_loader_module
import numpy as np
from tqdm import tqdm
import argparse
from model import NvidiaModel, activation
from config import config
from utils import EarlyStopping


parser = argparse.ArgumentParser(description="Compare loss values from two CSV files.")
parser.add_argument("--dataset_type", type=str, help="Dataset type", choices=['sully', 'udacity', 'udacity_sim_1'], default='sully')
parser.add_argument("--batch_size", type=int, help="Batch size", default=config.batch_size)
parser.add_argument("--epochs_count", type=int, help="Epochs count", default=config.epochs_count)
parser.add_argument("--tensorboard_run_name", type=str, help="Tensorboard run name", default='tensorboard')
parser.add_argument("--device", type=str, help="GPU device", default=None)


def save_model(model, log_dir="./save"):
    if not config.is_saving_enabled:
        return
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    checkpoint_path = os.path.join(log_dir, config.model_path)
    if config.device == 'cuda':
        model.to('cpu')
    torch.save(model.state_dict(), checkpoint_path)

    if config.device == 'cuda':
        model.to('cuda')


def train(desc_message, model, train_subset_loader, loss_function, optimizer):
    model.train()
    batch_loss = np.array([])

    for data, target in tqdm(train_subset_loader, desc=desc_message, ascii=' ='):

        data = data.to(config.device)
        target = target.to(config.device)

        optimizer.zero_grad()

        y_pred = model(data)

        loss = loss_function(y_pred.float(), target.float())

        loss.backward()
        optimizer.step()

        batch_loss = np.append(batch_loss, [loss.item()])

        epoch_loss = batch_loss.mean()

    return epoch_loss


def validation(desc_message, model, val_subset_loader, loss_function):
    # Load model
    model.eval()
    batch_loss = np.array([])

    with torch.no_grad():
        for data_val, target_val in tqdm(val_subset_loader, desc=desc_message, ascii=' ='):
            # send data to device (its is medatory if GPU has to be used)
            data_val = data_val.to(config.device)

            # send target to device
            target_val = target_val.to(config.device)
            
            # forward pass to the model
            y_pred_val = model(data_val)

            # cross entropy loss
            loss = loss_function(y_pred_val.float(), target_val.float())

            # Capture log
            batch_loss = np.append(batch_loss, [loss.item()])
                
    epoch_loss = batch_loss.mean()

    return epoch_loss


def add_grad_average_to_tensorboard(writer, model, train_subset_loader, epoch, fold):
    # Log the gradient norms to TensorBoard
    avg_grads = {name: 0 for name, param in model.named_parameters() if param.requires_grad}
    for name, param in model.named_parameters():
        if param.requires_grad:
            avg_grads[name] += param.grad.abs().mean().item()

    # Average over batches and write to tensorboard
    for name, grad_sum in avg_grads.items():
        avg_grad = grad_sum / len(train_subset_loader)
        writer.add_scalar(f'Grad Avg/{name}_fold{fold}', avg_grad, epoch)


def add_learning_rate_to_tensorboard(writer, optimizer, epoch, fold):
    # Log the learning rate to TensorBoard
    for param_group in optimizer.param_groups:
        writer.add_scalar(f'Learning_rate/lr_fold{fold}', param_group['lr'], epoch)


def add_images_to_tensorboard(writer, epoch, fold):
    # Normalize the activations from the 'first_conv_layer'
    images1 = activation['first_conv_layer'][0]

    # Normalize the images to [0,1] range
    images1 = (images1 - images1.min()) / (images1.max() - images1.min())

    # Visualize the first 16 feature maps
    grid1 = make_grid(images1[:16].unsqueeze(1), nrow=4, normalize=False)

    # Resize the grid using interpolation
    grid1 = F.interpolate(grid1.unsqueeze(0), scale_factor=2, mode='nearest').squeeze(0)

    writer.add_image(f'Images/First_layer_fold_{fold}', grid1, epoch)

    # Repeat the same process for the 'second_conv_layer'
    images2 = activation['second_conv_layer'][0]

    # Normalize the images to [0,1] range
    images2 = (images2 - images2.min()) / (images2.max() - images2.min())

    # Visualize the first 16 feature maps
    grid2 = make_grid(images2[:16].unsqueeze(1), nrow=4, normalize=False)

    # Resize the grid using interpolation
    grid2 = F.interpolate(grid2.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0)

    writer.add_image(f'Images/Second_layer_fold_{fold}', grid2, epoch)


def run_epoch(model, train_subset_loader, val_subset_loader, loss_function, optimizer, epoch, writer, fold, header):
    # Train the model
    epoch_loss = train(f"{header}, Training", model, train_subset_loader, loss_function, optimizer)

    # Validate the model
    val_epoch_loss = validation(f"{header}, Validation", model, val_subset_loader, loss_function)
    
    print(f'{header}, Train Loss: {epoch_loss:.9f}')
    print(f"{header}, Validation Loss: {val_epoch_loss:.9f}")
    
    if config.is_loss_logging_enabled:
        # Log the train/val loss to TensorBoard
        writer.add_scalars(f'Loss/Fold_{fold}', {'train': epoch_loss, 'val': val_epoch_loss}, epoch)

    if config.is_learning_rate_logging_enabled:
        # Log the learning rate to TensorBoard
        add_learning_rate_to_tensorboard(writer, optimizer, epoch, fold)
    
    if config.is_grad_avg_logging_enabled:
        # Log the average gradient to TensorBoard
        add_grad_average_to_tensorboard(writer, model, train_subset_loader, epoch, fold)

    if config.is_image_logging_enabled:
        # Log the feature maps to TensorBoard
        add_images_to_tensorboard(writer, epoch, fold)

    return epoch_loss, val_epoch_loss
    


def main():
    args = parser.parse_args()

    start_time = time.time()

    print(f"Starting Cross-Validation for:")
    print(f"    Folds: {config.cross_validation_folds}")
    print(f"    TensorBoard Run Name: {args.tensorboard_run_name}")
    print(f"    Number of Epochs: {args.epochs_count}")
    print(f"    Batch Size: {config.batch_size}")
    print(f"    Learning Rate: {config.learning_rate}")
    print(f"    Weight Decay: {config.weight_decay}")
    print(f"    Optimizer: {config.optimizer}")
    print(f"    Number of workers: {config.num_workers}")
    print(f"    Scheduler: {config.scheduler_type}")

    if args.device is not None:
        config.device = args.device
    
    # Initialize the TensorBoard writer
    writer = SummaryWriter(log_dir=f'./logs/{args.tensorboard_run_name}/')

    # Lists to store the average loss for each fold
    train_losses = []
    validate_losses = []

    print("Loading datasets concatenated...")

    dataset_types = [
        "udacity_sim_2",
        "carla_001",
        "carla_002",
        "carla_003"
    ]
    dataset = dataset_loader_module.get_datasets(dataset_types=dataset_types)

    print("Total data size: ", len(dataset))

    # Define cross-validator
    kfold = KFold(n_splits=config.cross_validation_folds, shuffle=True)

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(dataset), start=1):
        print(f"\nStarting fold {fold}...\n")

        # Reset the model, optimizer, and scheduler at the start of each fold
        model = NvidiaModel()
        model.to(config.device)

        if config.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
        elif config.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {config.optimizer}")
        
        if config.scheduler_type == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
        elif config.scheduler_type == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.scheduler_multistep_milestones, gamma=config.scheduler_gamma)
        elif config.scheduler_type == 'nonscheduler':
            scheduler = None
        else:
            raise ValueError(f"Invalid scheduler type: {config.scheduler_type}")

        loss_function = nn.MSELoss()

        # SubsetRandomSampler generates indices for train/validation samples
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(valid_ids)

        # Create the data loaders
        train_subset_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers
        )

        val_subset_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers
        )

        # Initialize the early stopping object
        early_stopping_val = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta)
        early_stopping_train = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.early_stopping_min_delta)

        # Lists to store the loss for each epoch in this fold
        fold_train_losses = []
        fold_validate_losses = []

        # Start epochs
        for epoch in range(1, args.epochs_count + 1):
            header = f"Fold: {fold}, Epoch: {epoch}/{args.epochs_count}"

            # Run epoch
            epoch_train_loss, epoch_validate_loss = run_epoch(
                model,
                train_subset_loader,
                val_subset_loader,
                loss_function,
                optimizer,
                epoch,
                writer,
                fold,
                header
            )

            # Save the losses for this epoch
            fold_train_losses.append(epoch_train_loss)
            fold_validate_losses.append(epoch_validate_loss)
            
            # Update the learning rate
            if scheduler is not None:
                scheduler.step()

            # early stopping
            early_stopping_train(epoch_train_loss)
            if early_stopping_train.early_stop:
                print(f"Early stopping triggered after {config.early_stopping_patience} epochs without improvement in training loss")
                break
            
            early_stopping_val(epoch_validate_loss)
            if early_stopping_val.early_stop:
                print(f"Early stopping triggered after {config.early_stopping_patience} epochs without improvement in validation loss")
                break

            # Save the final model
            save_model(model)

        # Calculate and save the average loss for this fold
        train_losses.append(sum(fold_train_losses) / len(fold_train_losses))
        validate_losses.append(sum(fold_validate_losses) / len(fold_validate_losses))

    average_train_loss = sum(train_losses) / len(train_losses)
    average_validate_loss = sum(validate_losses) / len(validate_losses)

    # Write to tensorboard
    writer.add_scalar('Cross Validation/Average training loss', average_train_loss)
    writer.add_scalar('Cross Validation/Average validation loss', average_validate_loss)

    # Print the cross validation scores
    print('Cross validation scores:')
    print('Average training loss: ', average_train_loss)
    print('Average validation loss: ', average_validate_loss)

    # Save the final model
    save_model(model)

    # Close the TensorBoard writer
    writer.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds")
    print("Training finished")


if __name__ == '__main__':
    main()
