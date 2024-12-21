# CIFAR10 Classification with Custom CNNs

This project implements two custom CNN architectures for CIFAR10 image classification with specific architectural constraints and requirements.

## Architectures

### 1. CIFAR10Net (Dilated Architecture)
Uses dilated convolutions for increasing receptive field while maintaining spatial dimensions.

- **Input**: 32x32x3
- **Channel Progression**: 3 → 12 → 16 → 24 → 32 → 48 → 64 → 96
- **Spatial Dimensions**: Maintained at 32x32 until GAP
- **Final RF**: 135x135

Key Features:
- Progressive dilation rates (2→4→8→16→32)
- No spatial reduction until GAP
- Depthwise Separable Convolution in C2
- Multiple dilated convolutions

### 2. CIFAR10StridedNet (Strided Architecture)
Uses strided convolutions for spatial reduction.

- **Input**: 32x32x3
- **Channel Progression**: Same as above
- **Spatial Reduction**: 32→16→8→4→1
- **Final RF**: 67x67

Key Features:
- Strided convolutions for downsampling
- Depthwise Separable Convolution in C2
- Dilated Convolution in C3
- Progressive spatial reduction

## Requirements

bash
torch
torchvision
albumentations
numpy
tqdm
torchsummary

## Project Structure
├── models/
│ ├── init.py
│ └── network.py # Network architectures
├── utils/
│ ├── init.py
│ ├── data_loader.py # Data loading and augmentation
│ └── trainer.py # Training utilities
├── train.py # Main training script
├── train.log # Training logs
└── README.md

bash
python train.py


## Data Augmentation

Uses Albumentations library with:
- Horizontal Flip
- ShiftScaleRotate
- CoarseDropout (with specific parameters)
  - max_holes = 1
  - max_height = 16px
  - max_width = 16px
  - min_holes = 1
  - min_height = 16px
  - min_width = 16px
  - fill_value = dataset mean

## Model Parameters

- Input Size: 32x32x3
- Channel Progression: 3 → 12 → 16 → 24 → 32 → 48 → 64 → 96
- Total Parameters: ~191K
- Final Receptive Field: 47x47

## Training

To train the model:

bash
python train.py

## Results

Target metrics:
- Accuracy: 89%
- Training time: ~40 epochs
- Parameter efficiency: <150K parameters
- 85% in 20 epochs can be achieved as well.
## Implementation Details

### Key Components:
1. **DepthwiseSeparableConv**:
   - Separates spatial and channel-wise convolutions
   - Reduces parameters while maintaining performance

2. **Dilated Convolutions**:
   - Increases receptive field without parameter increase
   - Maintains spatial dimensions

3. **Transition Blocks**:
   - In StridedNet: Uses strided convolutions
   - In DilatedNet: Uses dilated convolutions

4. **Global Average Pooling**:
   - Replaces fully connected layers
   - Reduces parameters significantly

