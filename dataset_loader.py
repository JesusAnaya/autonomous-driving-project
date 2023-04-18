from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import config
import torch
import pandas as pd
import numpy as np
import os
import scipy
import torch


transform_img = transforms.Compose([
    transforms.Resize(config.resize, antialias=True),
    transforms.ToTensor(),
])


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
        area = (0, 110, width, height)
        cropped_img = image.crop(area)

        y = np.radians(self.dataframe.iloc[idx, 1])
        if self.transform:
            cropped_img = self.transform(cropped_img)
                    
        return cropped_img, float(y)
    
    @staticmethod
    def get_mean() -> list:
        return [0.3568, 0.3770, 0.3691]

    @staticmethod
    def get_std() -> list:
        return [0.2121, 0.2040, 0.1968]


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
    
    @staticmethod
    def get_mean() -> list:
        return [0.2957, 0.3153, 0.3688]

    @staticmethod
    def get_std() -> list:
        return [0.2556, 0.2609, 0.2822]


class CarlaSimulatorDataset(Dataset):
    def __init__(self, csv_file="steering_data.csv", root_dir="dataset", transform=None, mean=None, std=None):
        self.transform = transform
        self.dataset_folder = root_dir
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.mean = [0.2957, 0.3153, 0.3688]
        self.std = [0.2556, 0.2609, 0.2822]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(os.path.join(self.dataset_folder, 'images'), self.data.iloc[idx]['frame_name'])
        image = Image.open(img_name).convert('RGB')
        width, height = image.size
        area = (0, 115, width, height)
        cropped_img = image.crop(area)

        angle = float(self.data.iloc[idx]['steering_angle'])

        if self.transform:
            image = self.transform(cropped_img)

        return image, angle
    
    def set_transform(self, transform):
        self.transform = transform
    

def get_inference_dataset(dataset_type='carla_001', transform=transform_img) -> DataLoader:
    if dataset_type == 'carla_001':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_001_town04",
            mean=[0.5886, 0.5800, 0.5878],
            std=[0.0794, 0.0792, 0.0786]
        )
    elif dataset_type == 'carla_002':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_002_town02_small",
            mean=[0.6460, 0.6213, 0.6036],
            std=[0.2173, 0.2066, 0.1929]
        )
    elif dataset_type == 'carla_003':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_003_town01_small",
            mean=[0.5124],
            std=[0.1333]
        )
    elif dataset_type == 'carla_004':
        return CarlaSimulatorDataset(
            transform=transform,
            root_dir="datasets/dataset_carla_004_town04_2",
            mean=[0.5911, 0.5821, 0.5918],
            std=[0.0711, 0.0720, 0.0746]
        )
    else:
        raise ValueError("Invalid dataset type")
    

def get_data_subsets_loaders(dataset_type='carla_001', batch_size=config.batch_size) -> Tuple[DataLoader, DataLoader]:
    dataset = get_inference_dataset(dataset_type)

    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.resize, antialias=True),
        transforms.Normalize(dataset.mean, dataset.std)
    ])
    dataset.set_transform(transform_img)

    train_set, val_set = random_split(dataset, [config.train_split_size, config.test_split_size])

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
