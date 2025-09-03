import torch.nn as nn
from torchvision import datasets, transforms
from functools import partial

from sde.models.types import DatasetDefinition, DatasetType
from sde.challenges.utils import DataLoaderFactory

# --- 1. Define the transform ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# --- 2. Dataset Definition ---
CIFAR10_DATASET = DatasetDefinition(
    name="CIFAR-10",
    type=DatasetType.IMAGE_CLASSIFICATION,
    description="A dataset of 60,000 32x32 color images in 10 classes (e.g., airplane, dog, cat).",
    loader_factory=partial(
        DataLoaderFactory,
        dataset_class=datasets.CIFAR10,
        data_dir='./data_cifar10',
        transform=transform
    ),
    input_shape=(3, 32, 32),
    output_shape=10,
    loss_function_factory=nn.CrossEntropyLoss,
    performance_metric_name="accuracy"
)
