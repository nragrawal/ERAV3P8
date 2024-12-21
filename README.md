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

### CIFAR10Net (Dilated Architecture)
- Input Size: 32x32x3
- Channel Progression: 3 → 12 → 16 → 24 → 32 → 48 → 64 → 96
- Spatial Dimensions: Maintained at 32x32
- Final RF: 135x135

Parameter Count:
1. C1 Block: 
   - Conv1: (3×12×3×3) + 12 = 336
   - BN1: 24
   - Conv2: (12×16×3×3) + 16 = 1,744
   - BN2: 32

2. Dilated1 (dilation=2):
   - Conv: (16×16×3×3) + 16 = 2,320
   - BN: 32

3. C2 Block:
   - DepthwiseSep: (16×1×3×3) + 16 + (16×24×1×1) + 24 = 568
   - BN: 48
   - Conv: (24×32×3×3) + 32 = 6,944
   - BN: 64

4. Dilated2 (dilation=4):
   - Conv: (32×32×3×3) + 32 = 9,248
   - BN: 64

5. C3 Block:
   - Conv1: (32×48×3×3) + 48 = 13,872
   - BN1: 96
   - Conv2: (48×64×3×3) + 64 = 27,712
   - BN2: 128

6. Dilated3 (dilation=16):
   - Conv: (64×64×3×3) + 64 = 36,928
   - BN: 128

7. C4 Block:
   - Conv: (64×96×3×3) + 96 = 55,392

8. FC Layer:
   - Linear: 96×10 + 10 = 970

Total Parameters: ~156,640

### CIFAR10StridedNet
- Input Size: 32x32x3
- Channel Progression: Same as above
- Spatial Reduction: 32→16→8→4
- Final RF: 67x67

Parameter Count: Same as above but with strided convolutions instead of dilated

Key Differences:
- Dilated version maintains spatial dimensions longer
- Strided version has progressive spatial reduction
- Both maintain similar parameter count
- Different effective receptive fields

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

