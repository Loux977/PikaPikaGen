import random
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
    
def load_image_fast(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# Training Augmentation
def get_augmentation_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=(255, 255, 255)
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # mild color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

# Valid & Test Augmentations
def get_basic_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

# Helper function for setting seeds
def set_all_seeds(seed=100):
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
