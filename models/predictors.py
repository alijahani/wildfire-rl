import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import SEResidual2D

class PredictGrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 64, 3, padding=1)
        self.block1 = SEResidual2D(64)
        self.block2 = SEResidual2D(64)
        self.conv2 = nn.Conv2d(64, 16, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x[:, 13, :, :] = torch.sigmoid(x[:, 13, :, :])
        return x

class PredictReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 64, 3, padding=1)
        self.block1 = SEResidual2D(64)
        self.block2 = SEResidual2D(64)
        self.conv2 = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.conv2(x)
        return x.sum((2, 3))

class PredictQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 64, 3, padding=1)
        self.block1 = SEResidual2D(64)
        self.fc1 = nn.Linear(64, 9)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = F.silu(x.sum((2, 3)))
        a = self.fc1(x) - self.fc1(x).mean(1, keepdim=True)
        v = self.fc2(x)
        return a + v
