import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import SEResidual3D

class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(16, 16, 3, padding=1)
        self.block1 = SEResidual3D(16)
        self.block2 = SEResidual3D(16)
        self.conv2 = nn.Conv3d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        a1 = F.silu(torch.mean(self.conv2(x), 2))
        a2 = F.silu(self.conv3(x)[:, :, -1, :, :])
        return a1 + a2
