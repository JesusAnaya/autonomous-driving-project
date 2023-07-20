from typing import Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from config import config
import pandas as pd
import numpy as np
import os
import torch
import cv2


def add_random_shadow_bgr(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a random shadow intensity and region
    intensity = 0.4
    x1, x2 = np.random.randint(0, img.shape[1], size=2)

    if x1 > x2:
        x1, x2 = x2, x1

    # Apply the shadow
    hsv[:, x1:x2, 2] = hsv[:, x1:x2, 2] * intensity

    # Convert the image back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_random_brightness_bgr(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate a random brightness offset
    offset = np.random.randint(-50, 50)

    # Add the offset to the V channel
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + offset, 0, 255)

    # Convert the image back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def convert_opencv_image_to_torch(image):
    # Convert the image from OpenCV numpy format to PyTorch format
    # From (H x W x C) to (C x H x W) and convert to float
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float()

    # Normalize the image
    image = (image / 127.5) - 1.0

    return image


class UdacitySimulatorDataset(Dataset):
    def __init__(self, csv_file="driving_log.csv", root_dir="datasets/udacity_sim_data_2"):
        self.dataset_folder = root_dir
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))

    def __len__(self):
        return len(self.data) * 3 * 3  # Each row contains three images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_idx = idx // 9  # Get the corresponding row in the CSV file
        img_type = (idx // 3) % 3  # Get the type of image (0=center, 1=left, 2=right)
        augmentation_type = idx % 3  # Get the type of augmentation (0=none, 1=flip, 2=brightness)

        if img_type == 0:
            img_file = self.data.iloc[img_idx]['center']
            angle = round(float(self.data.iloc[img_idx]['steering']), 4)
        elif img_type == 1:
            img_file = self.data.iloc[img_idx]['left']
            angle = round(float(self.data.iloc[img_idx]['steering']) + 0.20, 4)  # Adjust steering angle
        else:  # img_type == 2
            img_file = self.data.iloc[img_idx]['right']
            angle = round(float(self.data.iloc[img_idx]['steering']) - 0.20, 4)  # Adjust steering angle

        img_name = os.path.join(self.dataset_folder, 'IMG', os.path.basename(img_file))
        image = cv2.imread(img_name)

        # Augmentation. Apply selected augmentation
        if augmentation_type == 1:
            image = add_random_brightness_bgr(image)
        elif augmentation_type == 2:
            image = cv2.flip(image, 1)  # 1 for horizontal flipping
            angle = angle * -1.0
        else:
            pass  # No augmentation

        # Convert the image to YUV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # resize image to 200x66
        image = cv2.resize(image, (200, 66))

        torch_image = convert_opencv_image_to_torch(image)

        return torch_image, angle


class CarlaSimulatorDataset(Dataset):
    def __init__(self, csv_file="steering_data.csv", root_dir="dataset/dataset_carla_001_town04"):
        self.dataset_folder = root_dir
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))

    def __len__(self):
        return len(self.data) * 3  # Triple the dataset size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_idx = idx // 3  # Corrected from 9 to 3

        img_name = os.path.join(os.path.join(self.dataset_folder, 'images'), self.data.iloc[img_idx]['frame_name'])
        image = cv2.imread(img_name)

        angle = round(float(self.data.iloc[img_idx]['steering_angle']), 4)

        # Get the type of augmentation (0=none, 1=flip, 2=shadows)
        augmentation_type = idx % 3

        # Apply selected augmentation
        if augmentation_type == 1:  # Changed from 0 to 1
            image = add_random_brightness_bgr(image)
        elif augmentation_type == 2:  # Changed from 1 to 2
            image = cv2.flip(image, 1)  # 1 for horizontal flipping
            angle = angle * -1.0
        else:
            pass  # No augmentation

        # Convert the image to YUV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # resize image to 200x66
        image = cv2.resize(image, (200, 66))

        torch_image = convert_opencv_image_to_torch(image)

        return torch_image, angle


def get_inference_dataset(dataset_type='carla_001'):
    if dataset_type == 'carla_001':
        return CarlaSimulatorDataset(
            root_dir="datasets/dataset_carla_001_town04"
        )
    elif dataset_type == 'carla_002':
        return CarlaSimulatorDataset(
            root_dir="datasets/dataset_carla_002_town02_small"
        )
    elif dataset_type == 'carla_003':
        return CarlaSimulatorDataset(
            root_dir="datasets/dataset_carla_003_town01_small"
        )
    elif dataset_type == 'carla_004':
        return CarlaSimulatorDataset(
            root_dir="datasets/dataset_carla_004_town04_2"
        )
    elif dataset_type == 'udacity_sim_1':
        return UdacitySimulatorDataset(
            root_dir="datasets/udacity_sim_data_1"
        )
    elif dataset_type == 'udacity_sim_2':
        return UdacitySimulatorDataset(
            root_dir="datasets/udacity_sim_data_2"
        )
    else:
        raise ValueError("Invalid dataset type")
    

def get_datasets(dataset_types=['carla_001']) -> Dataset:
    datasets_list = []
    for dataset_type in dataset_types:
        dataset = get_inference_dataset(dataset_type)
        datasets_list.append(dataset)

    dataset_concatenated = ConcatDataset(datasets_list)

    return dataset_concatenated


def get_data_subsets_loaders(dataset_types=['udacity_sim_2'], batch_size=config.batch_size) -> Tuple[DataLoader, DataLoader]:
    loades_datasets = []

    for dataset_type in dataset_types:
        dataset = get_inference_dataset(dataset_type)
        loades_datasets.append(dataset)

    merged_dataset = ConcatDataset(loades_datasets)

    train_set, val_set = random_split(merged_dataset, [config.train_split_size, config.test_split_size])

    train_subset_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    val_subset_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    return train_subset_loader, val_subset_loader


def get_full_dataset_loader(dataset_type='carla_001') -> DataLoader:
    dataset = get_inference_dataset(dataset_type)

    full_dataset_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    return full_dataset_loader
