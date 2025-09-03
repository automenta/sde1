import numpy as np
from torch.utils.data import DataLoader, Subset

class DataLoaderFactory:
    """
    A pickleable factory for creating PyTorch DataLoaders.

    It performs the expensive setup (data download, splitting) in its constructor
    and then can be called to generate DataLoaders with a specific batch size.
    This two-step process allows for trial-specific hyperparameters like batch_size
    to be injected at execution time inside a worker process.
    """
    def __init__(self,
                 dataset_class,
                 data_dir: str,
                 transform,
                 val_split: float = 0.1,
                 seed: int = 42):

        # Pre-load the dataset once to avoid repeated downloads and splits.
        full_train_dataset = dataset_class(data_dir, train=True, download=True, transform=transform)

        # Create and store the validation split indices
        num_train = len(full_train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_split * num_train))
        np.random.seed(seed)
        np.random.shuffle(indices)

        train_idx, val_idx = indices[split:], indices[:split]
        self.train_subset = Subset(full_train_dataset, train_idx)
        self.val_subset = Subset(full_train_dataset, val_idx)

    def __call__(self, batch_size: int):
        """
        Generates and returns the actual DataLoaders.
        Args:
            batch_size: The batch size to use for the DataLoaders.
        """
        # Use num_workers=0 for simplicity and to avoid multiprocessing issues in some environments
        train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader
