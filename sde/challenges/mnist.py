import torch.nn as nn
from torchvision import datasets, transforms
from functools import partial

from sde.models.types import DatasetDefinition, DatasetType
from sde.challenges.utils import DataLoaderFactory

# --- 1. Define the transform ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
])

# --- 2. Dataset Definition ---
# We use functools.partial to pre-configure the DataLoaderFactory for MNIST.
# This creates a callable that the worker can then instantiate.
MNIST_DATASET = DatasetDefinition(
    name="MNIST",
    type=DatasetType.IMAGE_CLASSIFICATION,
    description="A classic dataset of 70,000 28x28 grayscale images of handwritten digits (0-9).",
    loader_factory=partial(
        DataLoaderFactory,
        dataset_class=datasets.MNIST,
        data_dir='./data_mnist',
        transform=transform
    ),
    input_shape=(1, 28, 28),
    output_shape=10,
    loss_function_factory=nn.CrossEntropyLoss,
    performance_metric_name="accuracy"
)
