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
        area = (100, 125, width, height)
        cropped_img = image.crop(area)

        y = np.radians(self.dataframe.iloc[idx, 1])
        if self.transform:
            cropped_img = self.transform(cropped_img)
                    
        return cropped_img, float(y)
    
    @staticmethod
    def get_mean():
        return [0.3568, 0.3770, 0.3691]

    @staticmethod
    def get_std():
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
    def get_mean():
        return [0.2957, 0.3153, 0.3688]

    @staticmethod
    def get_std():
        return [0.2556, 0.2609, 0.2822]


def get_data_subsets_loaders(dataset_type='sully') -> (DataLoader, DataLoader):
    dataset_class = None
    
    if dataset_type == 'sully':
        dataset_class = SullyChenDataset
    elif dataset_type == 'udacity':
        dataset_class = UdacityDataset
    else:
        raise ValueError("Invalid dataset type")

    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.resize, antialias=True),
        transforms.Normalize(dataset_class.get_mean(), dataset_class.get_std())
    ])

    dataset = dataset_class(transform=transform_img)

    train_set, val_set = random_split(dataset, [config.train_split_size, config.test_split_size])

    train_subset_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    val_subset_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )
    return train_subset_loader, val_subset_loader


def get_full_dataset_loader(dataset_type='sully') -> DataLoader:
    if dataset_type == 'sully':
        dataset = SullyChenDataset(transform=transform_img)
    elif dataset_type == 'udacity':
        dataset = UdacityDataset(transform=transform_img)

    full_dataset_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    return full_dataset_loader


def get_inference_dataset(dataset_type='sully') -> DataLoader:
    if dataset_type == 'sully':
        return SullyChenDataset(transform=transform_img)
    elif dataset_type == 'udacity':
        return UdacityDataset(transform=transform_img)
