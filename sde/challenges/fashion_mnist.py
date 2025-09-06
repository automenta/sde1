import torch.nn as nn
from torchvision import datasets, transforms

from sde.models.types import DatasetDefinition, DatasetType
from sde.challenges.utils import create_train_val_dataloaders

# --- 1. The Data Loader Factory ---


def get_fashion_mnist_dataloaders(batch_size=64, data_dir="./data_fashion_mnist"):
    """
    Returns training and validation DataLoaders for Fashion-MNIST.
    """
    # Normalization constants for Fashion-MNIST
    # Mean: 0.2860, Std: 0.3530 (calculated from the training set)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    return create_train_val_dataloaders(
        dataset_class=datasets.FashionMNIST,
        data_dir=data_dir,
        transform=transform,
        batch_size=batch_size,
    )


# --- 2. Dataset Definition ---

FASHION_MNIST_DATASET = DatasetDefinition(
    name="Fashion-MNIST",
    type=DatasetType.IMAGE_CLASSIFICATION,
    description="A dataset of 70,000 28x28 grayscale images of 10 types of clothing items.",
    loader_factory=get_fashion_mnist_dataloaders,
    input_shape=(1, 28, 28),
    output_shape=10,
    loss_function_factory=nn.CrossEntropyLoss,
    performance_metric_name="accuracy",
)
