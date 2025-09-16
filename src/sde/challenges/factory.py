from functools import partial

import torch.nn as nn
from sde.core.definitions import DatasetDefinition, DatasetType
from torchvision.transforms import Compose

from .utils import create_train_val_dataloaders


def create_image_classification_challenge(
    name: str,
    description: str,
    dataset_class,
    transform: Compose,
    input_shape: tuple,
    output_shape: int,
    data_dir: str,
) -> DatasetDefinition:
    """Factory function to create a standardized image classification DatasetDefinition.

    This handles the boilerplate of creating a loader_factory function and
    packaging it into a DatasetDefinition object.

    Args:
        name: The display name of the challenge.
        description: A brief description of the dataset.
        dataset_class: The torchvision.datasets class for this dataset.
        transform: The torchvision transforms to apply to the data.
        input_shape: The shape of a single input tensor (e.g., (1, 28, 28)).
        output_shape: The number of output classes.
        data_dir: The directory to store the dataset files.

    Returns:
        A fully configured DatasetDefinition object.

    """
    # Use functools.partial to create a specific loader factory function
    # that has the dataset-specific arguments "baked in".
    loader_factory = partial(
        create_train_val_dataloaders,
        dataset_class=dataset_class,
        data_dir=data_dir,
        transform=transform,
    )

    return DatasetDefinition(
        name=name,
        type=DatasetType.IMAGE_CLASSIFICATION,
        description=description,
        loader_factory=loader_factory,
        input_shape=input_shape,
        output_shape=output_shape,
        # For now, these are common to all our image classification tasks
        loss_function_factory=nn.CrossEntropyLoss,
        performance_metric_name="accuracy",
    )
