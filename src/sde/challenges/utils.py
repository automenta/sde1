import random
from typing import Any
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset


def _generate_random_hyperparameters(parameter_space: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to generate one set of random hyperparameters."""
    hparams = {}
    for p_name, p_def in parameter_space.items():
        if isinstance(p_def, dict) and "min" in p_def and "max" in p_def:
            if p_def.get("scale") == "log":
                log_min = np.log10(p_def["min"])
                log_max = np.log10(p_def["max"])
                value = 10 ** random.uniform(log_min, log_max)
            else:
                value = random.uniform(p_def["min"], p_def["max"])

            if p_def.get("type") == "int":
                value = int(value)
        elif isinstance(p_def, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in p_def) and len(p_def) == 2:
                value = random.uniform(p_def[0], p_def[1])
            else:
                value = random.choice(p_def)
        else:
            value = p_def
        hparams[p_name] = value
    return hparams


def create_train_val_dataloaders(
    dataset_class,
    data_dir: str,
    transform,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Creates training and validation DataLoaders from a torchvision dataset.

    Args:
        dataset_class: The torchvision dataset class (e.g., datasets.MNIST).
        data_dir: The directory to store the data.
        transform: The transformations to apply to the data.
        batch_size: The batch size for the dataloaders.
        val_split: The fraction of the training data to use for validation.
        seed: The random seed for the train/validation split.

    Returns:
        A tuple containing the training and validation DataLoaders.

    """
    # Download and load the full training dataset
    full_train_dataset = dataset_class(
        data_dir, train=True, download=True, transform=transform
    )

    # Create a validation split
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    # Use a fixed seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    # Use num_workers=0 for simplicity and to avoid multiprocessing issues
    # in some environments
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader
