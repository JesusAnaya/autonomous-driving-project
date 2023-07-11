from typing import Tuple
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from config import config
import torch
import pandas as pd
import numpy as np
import os
import torch
import random
import cv2

transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(config.resize, antialias=True),
])


def normalize_angle(angle, max_angle_degrees=180):
    normalized_angle = angle / max_angle_degrees
    normalized_angle = max(min(normalized_angle, 1.0), -1.0)
    return normalized_angle


def unwrap_angle(angle):
    if angle > 180:
        angle -= 360
    elif angle < -180:
        angle += 360
    return angle


def random_horizontal_flip(image, angle, p=0.5):
    if random.random() < p:
        image = transforms.functional.hflip(image)
        angle = -angle
    return image, angle


def augment_image(image, angle, flip_p=0.5):
    image, angle = random_horizontal_flip(image, angle, p=flip_p)
    if random.random() < 0.5:
        image = transforms.functional.adjust_brightness(image, random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        image = transforms.functional.adjust_contrast(image, random.uniform(0.8, 1.2))
    return image, angle


def add_random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.25 + np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


class AugmentedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, angle):
        image, angle = augment_image(image, angle)
        image = self.transform(image)
        return image, angle
    

class SullyChenDataset(Dataset):
    def __init__(self, csv_file="data.txt", root_dir="dataset", transform=None):
        file_path = os.path.join(root_dir, csv_file)
        self.dataframe = pd.read_csv(
            file_path,
            sep=r"\s|,",
            header=None,
            engine="python"
        )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(os.path.join(self.root_dir, "data"), self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        width, height = image.size
        area = (0, 90, width, height)
        cropped_img = image.crop(area)

        y = unwrap_angle(float(self.dataframe.iloc[idx, 1]))
        y = normalize_angle(y, 180.0)

        if self.transform:
            cropped_img, y = self.transform(cropped_img, y)
                    
        return cropped_img, float(y)
    
    def set_transform(self, transform):
        self.transform = transform


class UdacityDataset(Dataset):
    def __init__(self, csv_file="interpolated.csv", root_dir="dataset2", transform=None):
        self.transform = transform
        self.dataset_folder = root_dir
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))

        # Filter for center_camera images only
        self.data = self.data[self.data['frame_id'] == 'center_camera']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.dataset_folder, self.data.iloc[idx]['filename'])
        image = Image.open(img_name)
        width, height = image.size
        area = (0, 180, width, height)
        cropped_img = image.crop(area)

        angle = np.radians(self.data.iloc[idx]['angle'])

        if self.transform:
            image = self.transform(cropped_img)

        return image, float(angle)


class UdacitySimulator1Dataset(Dataset):
    def __init__(self, csv_file="driving_log.csv", root_dir="datasets/udacity_sim_data_1", transform=None):
        self.transform = transform
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop image
        image = image[60:-24, :, :]

        # Augmentation. Apply selected augmentation
        if augmentation_type == 1:
            # Horizontal flip (mirroring)
            image = cv2.flip(image, 1)  # 1 for horizontal flipping
            angle = angle * -1.0
        elif augmentation_type == 2:
            # Random brightness
            image = add_random_brightness(image)
        else:
            pass  # No augmentation

        # resize image to 200x66
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

        if self.transform:
            image = self.transform(image)

        return image, angle

    def set_transform(self, transform):
        self.transform = transform


class CarlaSimulatorDataset(Dataset):
    def __init__(self, csv_file="steering_data.csv", root_dir="dataset", transform=None):
        self.transform = transform
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop image
        image = image[50:, :, :]

        angle = round(float(self.data.iloc[img_idx]['steering_angle']), 4)

        # Get the type of augmentation (0=none, 1=flip, 2=shadows)
        augmentation_type = idx % 3

        # Apply selected augmentation
        if augmentation_type == 1:  # Changed from 0 to 1
            # Horizontal flip (mirroring)
            image = cv2.flip(image, 1)  # 1 for horizontal flipping
            angle = angle * -1.0
        elif augmentation_type == 2:  # Changed from 1 to 2
            # Random brightness
            image = add_random_brightness(image)
        else:
            pass  # No augmentation

        # resize image to 200x66
        image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)

        if self.transform:
            image = self.transform(image)

        return image, angle

    def set_transform(self, transform):
        self.transform = transform


def get_inference_dataset(dataset_type='carla_001', transform=transform_img) -> DataLoader:
    if dataset_type == 'carla_001':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_001_town04"
        )
    elif dataset_type == 'carla_002':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_002_town02_small"
        )
    elif dataset_type == 'carla_003':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_003_town01_small"
        )
    elif dataset_type == 'carla_004':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_004_town04_2"
        )
    elif dataset_type == 'sully':
        return SullyChenDataset(
            transform=transform,
            root_dir="datasets/sully"
        )
    elif dataset_type == 'udacity_sim_1':
        return UdacitySimulator1Dataset(
            transform=transform,
            root_dir="datasets/udacity_sim_data_1"
        )
    elif dataset_type == 'udacity_sim_2':
        return UdacitySimulator1Dataset(
            transform=transform,
            root_dir="datasets/udacity_sim_data_2"
        )
    else:
        raise ValueError("Invalid dataset type")
    

def get_dataset(dataset_type='carla_001') -> Dataset:
    dataset = get_inference_dataset(dataset_type)
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset.set_transform(transform_img)
    
    return dataset


def get_data_subsets_loaders(dataset_types=['udacity_sim_1'], batch_size=config.batch_size) -> Tuple[DataLoader, DataLoader]:
    loades_datasets = []

    for dataset_type in dataset_types:
        dataset = get_inference_dataset(dataset_type)
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset.set_transform(transform_img)
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
