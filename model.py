import torch
import torch.nn as nn
import scipy


class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 2

        # define layers using nn.Sequential
        self.layers = nn.Sequential(
            # first convolutional layer
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            # second convolutional layer
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),

            # third convolutional layer
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # fourth convolutional layer
            nn.Conv2d(48, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # fifth convolutional layer
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # flatten
            nn.Flatten(),

            # first fully connected layer
            nn.Linear(1152, 1164),
            nn.ReLU(),
            nn.BatchNorm1d(1164),
            nn.Dropout(p=0.5),

            # second fully connected layer
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.5),

            # third fully connected layer
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(p=0.5),

            # fourth fully connected layer
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),
            nn.Dropout(p=0.5),

            # output layer
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        #x = torch.atan(x) * self.scale_factor
        return torch.flatten(x)
