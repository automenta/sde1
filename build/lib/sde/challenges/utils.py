import numpy as np
from torch.utils.data import DataLoader, Subset


def create_train_val_dataloaders(
    dataset_class,
    data_dir: str,
    transform,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
):
    """
    Creates training and validation DataLoaders from a torchvision dataset.

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

    # Use num_workers=0 for simplicity and to avoid multiprocessing issues in some environments
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader
