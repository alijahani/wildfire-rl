import torch
import torch.nn as nn
import torch.nn.functional as F

class SEResidual3D(nn.Module):
    def __init__(self, channel_num, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv3d(channel_num, channel_num, 3, padding=1)
        self.norm1 = nn.BatchNorm3d(channel_num)
        self.conv2 = nn.Conv3d(channel_num, channel_num, 3, padding=1)
        self.norm2 = nn.BatchNorm3d(channel_num)
        self.fc1 = nn.Linear(channel_num, channel_num // reduction)
        self.fc2 = nn.Linear(channel_num // reduction, channel_num)

    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        scale = F.avg_pool3d(x, kernel_size=x.size()[2:5])[:, :, 0, 0, 0]
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return F.silu(residual + x * scale)

class SEResidual2D(nn.Module):
    def __init__(self, channel_num, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channel_num)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(channel_num)
        self.fc1 = nn.Linear(channel_num, channel_num // reduction)
        self.fc2 = nn.Linear(channel_num // reduction, channel_num)

    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        scale = F.avg_pool2d(x, kernel_size=x.size()[2:4])[:, :, 0, 0]
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale)).unsqueeze(-1).unsqueeze(-1)
        return F.silu(residual + x * scale)
