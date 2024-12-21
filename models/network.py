import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=1, padding=padding, groups=in_channels, dilation=dilation
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CIFAR10StridedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input size: 32x32x3, RF start: 1x1, jin=1

        # C1 Block - Regular Conv
        self.conv1 = nn.Sequential(
            # RF: 3x3, jin=1, rin=1, jout=1
            nn.Conv2d(3, 12, kernel_size=3, padding=1),    # 32x32x3 -> 32x32x12
            nn.BatchNorm2d(12),
            nn.ReLU(),

            # RF: 5x5, jin=1, rin=1, jout=1
            nn.Conv2d(12, 16, kernel_size=3, padding=1),   # 32x32x12 -> 32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Strided conv after C1
        self.stride1 = nn.Sequential(
            # RF: 7x7, jin=1, rin=1, jout=2 (stride=2)
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 32x32x16 -> 16x16x16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # C2 Block - Depthwise Separable Conv
        self.conv2 = nn.Sequential(
            # RF: 11x11, jin=2, rin=2, jout=2
            DepthwiseSeparableConv(16, 24, kernel_size=3),  # 16x16x16 -> 16x16x24
            nn.BatchNorm2d(24),
            nn.ReLU(),

            # RF: 15x15, jin=2, rin=2, jout=2
            nn.Conv2d(24, 32, kernel_size=3, padding=1),    # 16x16x24 -> 16x16x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Strided conv after C2
        self.stride2 = nn.Sequential(
            # RF: 19x19, jin=2, rin=2, jout=4 (stride=2)
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 16x16x32 -> 8x8x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # C3 Block - Dilated Conv
        self.conv3 = nn.Sequential(
            # RF: 35x35, jin=4, rin=4, jout=4 (dilation=2 doubles effective kernel)
            nn.Conv2d(32, 48, kernel_size=3, padding=2, dilation=2),  # 8x8x32 -> 8x8x48
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # RF: 43x43, jin=4, rin=4, jout=4
            nn.Conv2d(48, 64, kernel_size=3, padding=1),    # 8x8x48 -> 8x8x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Strided conv after C3
        self.stride3 = nn.Sequential(
            # RF: 51x51, jin=4, rin=4, jout=8 (stride=2)
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 8x8x64 -> 4x4x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # C4 Block - Final Conv
        self.conv4 = nn.Sequential(
            # RF: 67x67, jin=8, rin=8, jout=8
            nn.Conv2d(64, 96, kernel_size=3, padding=1),    # 4x4x64 -> 4x4x96
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        # Global Average Pooling and Final Layer
        self.gap = nn.AdaptiveAvgPool2d(1)                  # 4x4x96 -> 1x1x96
        self.fc = nn.Linear(96, 10)                         # 96 -> 10

    def forward(self, x):                                   # Input: Nx3x32x32
        x = self.conv1(x)      # 32x32x3 -> 32x32x16
        x = self.stride1(x)    # 32x32x16 -> 16x16x16
        x = self.conv2(x)      # 16x16x16 -> 16x16x32
        x = self.stride2(x)    # 16x16x32 -> 8x8x32
        x = self.conv3(x)      # 8x8x32 -> 8x8x64
        x = self.stride3(x)    # 8x8x64 -> 4x4x64
        x = self.conv4(x)      # 4x4x64 -> 4x4x96
        x = self.gap(x)        # 4x4x96 -> 1x1x96
        x = x.view(-1, 96)     # 1x1x96 -> 96
        x = self.fc(x)         # 96 -> 10
        return x               # Output: Nx10

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input size: 32x32x3, RF start: 1x1, jin=1, rin=1
        
        # C1 Block - Regular Conv
        self.conv1 = nn.Sequential(
            # RF: 3x3, jin=1, rin=1, jout=1
            nn.Conv2d(3, 12, kernel_size=3, padding=1),    # 32x32x3 -> 32x32x12
            nn.BatchNorm2d(12),
            nn.ReLU(),
            
            # RF: 5x5, jin=1, rin=1, jout=1
            nn.Conv2d(12, 16, kernel_size=3, padding=1),   # 32x32x12 -> 32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Dilated conv after C1
        self.dilated1 = nn.Sequential(
            # RF: 9x9 (5 + 2*2), jin=1, rin=1, jout=1
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2),  # 32x32x16 -> 32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # C2 Block - Depthwise Separable Conv
        self.conv2 = nn.Sequential(
            # RF: 11x11 (9 + 2*1), jin=1, rin=1, jout=1
            DepthwiseSeparableConv(16, 24, kernel_size=3),  # 32x32x16 -> 32x32x24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            # RF: 13x13 (11 + 2*1), jin=1, rin=1, jout=1
            nn.Conv2d(24, 32, kernel_size=3, padding=1),    # 32x32x24 -> 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Dilated conv after C2
        self.dilated2 = nn.Sequential(
            # RF: 21x21 (13 + 2*4), jin=1, rin=1, jout=1
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4),  # 32x32x32 -> 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # C3 Block - Dilated Conv
        self.conv3 = nn.Sequential(
            # RF: 37x37 (21 + 2*8), jin=1, rin=1, jout=1
            nn.Conv2d(32, 48, kernel_size=3, padding=8, dilation=8),  # 32x32x32 -> 32x32x48
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            # RF: 39x39 (37 + 2*1), jin=1, rin=1, jout=1
            nn.Conv2d(48, 64, kernel_size=3, padding=1),    # 32x32x48 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Dilated conv after C3
        self.dilated3 = nn.Sequential(
            # RF: 71x71 (39 + 2*16), jin=1, rin=1, jout=1
            nn.Conv2d(64, 64, kernel_size=3, padding=16, dilation=16),  # 32x32x64 -> 32x32x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # C4 Block - Final Conv with largest dilation
        self.conv4 = nn.Sequential(
            # RF: 135x135 (71 + 2*32), jin=1, rin=1, jout=1
            nn.Conv2d(64, 96, kernel_size=3, padding=32, dilation=32),    # 32x32x64 -> 32x32x96
            #nn.BatchNorm2d(96),
            #nn.ReLU()
        )
        
        # Global Average Pooling and Final Layer
        self.gap = nn.AdaptiveAvgPool2d(1)                  # 32x32x96 -> 1x1x96
        self.fc = nn.Linear(96, 10)                         # 96 -> 10

    def forward(self, x):                                   # Input: Nx3x32x32
        x = self.conv1(x)      # 32x32x3 -> 32x32x16
        x = self.dilated1(x)   # 32x32x16 -> 32x32x16
        x = self.conv2(x)      # 32x32x16 -> 32x32x32
        x = self.dilated2(x)   # 32x32x32 -> 32x32x32
        x = self.conv3(x)      # 32x32x32 -> 32x32x64
        x = self.dilated3(x)   # 32x32x64 -> 32x32x64
        x = self.conv4(x)      # 32x32x64 -> 32x32x96
        x = self.gap(x)        # 32x32x96 -> 1x1x96
        x = x.view(-1, 96)     # 1x1x96 -> 96
        x = self.fc(x)         # 96 -> 10
        return x               # Output: Nx10
