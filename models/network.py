import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input size: 32x32x3
        
        # C1 Block - Regular Conv
        self.conv1 = nn.Sequential(
            # RF: 3x3, jin=1, rin=1, start=1
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            
            # RF: 5x5, jin=1, rin=1
            nn.Conv2d(12, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # C2 Block - Depthwise Separable Conv
        self.conv2 = nn.Sequential(
            # RF: 7x7, jin=1, rin=1
            DepthwiseSeparableConv(16, 24, kernel_size=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            # RF: 9x9, jin=1, rin=1
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # C3 Block - Dilated Conv
        self.conv3 = nn.Sequential(
            # RF: 13x13 (due to dilation=2), jin=1, rin=1
            nn.Conv2d(32, 48, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            # RF: 15x15, jin=1, rin=1
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # C4 Block - Strided Conv (C40)
        self.conv4 = nn.Sequential(
            # RF: 17x17, jin=2, rin=1
            nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            # RF: 47x47, jin=2, rin=2
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        
        # Global Average Pooling and Final Layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, 10)

    def forward(self, x):
        x = self.conv1(x)  # 32x32 -> 32x32
        x = self.conv2(x)  # 32x32 -> 32x32
        x = self.conv3(x)  # 32x32 -> 32x32
        x = self.conv4(x)  # 32x32 -> 16x16 (due to stride=2)
        x = self.gap(x)    # 16x16 -> 1x1
        x = x.view(-1, 96)
        x = self.fc(x)
        return x 