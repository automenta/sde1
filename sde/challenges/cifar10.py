import torch.nn as nn
from torchvision import datasets, transforms

from sde.models.types import DatasetDefinition, DatasetType
from sde.challenges.utils import create_train_val_dataloaders

# --- 1. The Data Loader Factory ---

def get_cifar10_dataloaders(batch_size=64, data_dir='./data_cifar10'):
    """
    Returns training and validation DataLoaders for CIFAR-10 using the utility function.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return create_train_val_dataloaders(
        dataset_class=datasets.CIFAR10,
        data_dir=data_dir,
        transform=transform,
        batch_size=batch_size
    )

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
