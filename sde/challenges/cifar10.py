import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

from sde.models.types import DatasetDefinition, DatasetType

# --- 1. The Data Loader Factory ---

def get_cifar10_dataloaders(batch_size=64, val_split=0.1, data_dir='./data_cifar10'):
    """
    Returns training and validation DataLoaders for CIFAR-10.
    A validation set is split from the training data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])

    # Download and load the full training dataset
    full_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)

    # Create a validation split
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    # Use a fixed seed for reproducibility of the split
    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

# --- 2. Dataset Definition ---

CIFAR10_DATASET = DatasetDefinition(
    name="CIFAR-10",
    type=DatasetType.IMAGE_CLASSIFICATION,
    description="A dataset of 60,000 32x32 color images in 10 classes (e.g., airplane, dog, cat).",
    loader_factory=get_cifar10_dataloaders,
    input_shape=(3, 32, 32),
    output_shape=10,
    loss_function_factory=nn.CrossEntropyLoss,
    performance_metric_name="accuracy"
)
