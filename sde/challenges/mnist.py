import torch.nn as nn
from torchvision import datasets, transforms

from sde.models.types import DatasetDefinition, DatasetType
from sde.challenges.utils import create_train_val_dataloaders

# --- 1. The Data Loader Factory ---

def get_mnist_dataloaders(batch_size=64, data_dir='./data_mnist'):
    """
    Returns training and validation DataLoaders for MNIST using the utility function.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
    ])
    return create_train_val_dataloaders(
        dataset_class=datasets.MNIST,
        data_dir=data_dir,
        transform=transform,
        batch_size=batch_size
    )

# --- 2. Dataset Definition ---

MNIST_DATASET = DatasetDefinition(
    name="MNIST",
    type=DatasetType.IMAGE_CLASSIFICATION,
    description="A classic dataset of 70,000 28x28 grayscale images of handwritten digits (0-9).",
    loader_factory=get_mnist_dataloaders,
    input_shape=(1, 28, 28),
    output_shape=10,
    loss_function_factory=nn.CrossEntropyLoss,
    performance_metric_name="accuracy"
)
