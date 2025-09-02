import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# --- 1. The Algorithm (Model) ---

class SimpleCNN(nn.Module):
    """
    A simple CNN for MNIST classification.
    Hyperparameters can be passed to its constructor.
    """
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Image size calculation: 28x28 -> MaxPool -> 14x14 -> MaxPool -> 7x7. Flattened size = 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        # The worker will use CrossEntropyLoss which combines LogSoftmax and NLLLoss.
        # So the model should return raw logits.
        return x

# --- 2. The Challenge Definition ---

def get_mnist_dataloaders(batch_size=64, val_split=0.1, data_dir='./data_mnist'):
    """
    Returns training and validation DataLoaders for MNIST.
    A validation set is split from the training data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std of MNIST
    ])

    # Download and load the full training dataset
    full_train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

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

    # Using num_workers > 0 can cause issues in some environments, especially with multiprocessing in the next steps.
    # Sticking to 0 for simplicity and robustness for now.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# --- 3. Helper functions for the SDE ---

def get_algorithm_class(name: str):
    """Maps an algorithm name to its class."""
    if name == "SimpleCNN":
        return SimpleCNN
    raise ValueError(f"Unknown algorithm name: {name}")

def get_loss_function():
    """Returns the loss function for the MNIST challenge."""
    return nn.CrossEntropyLoss()

def get_performance_metric_name():
    """Returns the name of the primary performance metric."""
    return "accuracy"
