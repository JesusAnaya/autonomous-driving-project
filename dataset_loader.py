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
    transforms.Resize(config.resize),
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
        area = (0, 90, width, height)
        cropped_img = image.crop(area)

        y = np.radians(self.dataframe.iloc[idx, 1])
        if self.transform:
            cropped_img = self.transform(cropped_img)
                    
        return cropped_img.float(), np.float(y)


def get_data_subsets_loaders() -> (DataLoader, DataLoader):    
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.resize),
        transforms.Normalize(config.mean, config.std)
    ])
    dataset = SullyChenDataset(transform=transform_img)
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


def get_full_dataset() -> DataLoader:
    dataset = SullyChenDataset(transform=transform_img)
    full_dataset_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    return full_dataset_loader


def get_inference_dataset() -> DataLoader:
    return SullyChenDataset(transform=transform_img)
