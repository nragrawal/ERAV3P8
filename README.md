# CIFAR10 Image Classification with Custom CNN

This project implements a custom CNN architecture for CIFAR10 image classification with specific architectural constraints and requirements.

## Architecture Details

The network follows a C1C2C3C40 architecture pattern with the following specifications:

1. No MaxPooling (uses strided convolutions instead)
2. Total Receptive Field > 44
3. Uses Depthwise Separable Convolution
4. Uses Dilated Convolution
5. Uses Global Average Pooling
6. Parameters < 200K

### Layer Structure
- C1: Regular Convolutions
- C2: Includes Depthwise Separable Convolution
- C3: Includes Dilated Convolution
- C40: Strided Convolution (stride=2)

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
│ └── network.py # Network architecture
├── utils/
│ ├── init.py
│ ├── data_loader.py # Data loading and augmentation
│ └── trainer.py # Training utilities
└── train.py # Main training script
bash
python train.py
 
This README.md:
Explains the project architecture
Lists requirements
Shows project structure
Details data augmentation
5. Provides training instructions
Shows model architecture details
Includes parameter counts and channel progression
Provides clear instructions for running the code
Would you like me to add or modify any section?


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


The training process includes:
- 40 epochs
- Batch size of 128
- OneCycleLR scheduler
- Adam optimizer with weight decay
- Model checkpointing (saves best model)

## Results

The model is designed to achieve:
- Target Accuracy: 85%
- Training Time: ~40 epochs
- Parameter Efficiency: <200K parameters

## Model Architecture Details

1. **C1 Block (Regular Conv)**
   - Conv 3x3 (3→12)
   - Conv 3x3 (12→16)

2. **C2 Block (with Depthwise Separable)**
   - Depthwise Separable Conv (16→24)
   - Conv 3x3 (24→32)

3. **C3 Block (with Dilation)**
   - Dilated Conv 3x3 (32→48, dilation=2)
   - Conv 3x3 (48→64)

4. **C4 Block (Strided)**
   - Strided Conv 3x3 (64→96, stride=2)
   - Conv 3x3 (96→96)

5. **Output**
   - Global Average Pooling
   - FC Layer (96→10)

## License

MIT
