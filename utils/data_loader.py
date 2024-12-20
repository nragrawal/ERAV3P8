import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

class CIFAR10DataLoader:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)
        
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=self.mean, mask_fill_value=None,
                p=0.5
            ),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        self.test_transforms = A.Compose([
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

    def get_dataloader(self):
        train_data = datasets.CIFAR10(
            './data', train=True, download=True
        )
        test_data = datasets.CIFAR10(
            './data', train=False, download=True
        )
        
        train_loader = DataLoader(
            AlbumentationsDataset(train_data, self.train_transforms),
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            AlbumentationsDataset(test_data, self.test_transforms),
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=True
        )
        
        return train_loader, test_loader

class AlbumentationsDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label
    
    def __len__(self):
        return len(self.dataset) 