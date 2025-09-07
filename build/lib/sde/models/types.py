import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Type, Any, Tuple, Callable
from enum import Enum


class SdeModel(nn.Module):
    """
    Base class for all models in the Scientific Discovery Engine.
    It standardizes the model interface.
    """

    def __init__(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape


class DatasetType(Enum):
    """
    Enum to categorize the type of a dataset.
    Used for type safety to match compatible models and datasets.
    """

    IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION"
    TABULAR_REGRESSION = "TABULAR_REGRESSION"
    # Future types can be added here
    # TEXT_GENERATION = "TEXT_GENERATION"


class ModelType(Enum):
    """
    Enum to categorize the type of a model.
    """

    IMAGE_CLASSIFIER = "IMAGE_CLASSIFIER"
    TABULAR_REGRESSOR = "TABULAR_REGRESSOR"
    # Future types can be added here
    # LANGUAGE_MODEL = "LANGUAGE_MODEL"


@dataclass(frozen=True)
class DatasetDefinition:
    """
    A metadata container for a dataset.
    """

    name: str
    type: DatasetType
    description: str
    # A callable that returns (train_loader, val_loader)
    loader_factory: Callable[..., Tuple[Any, Any]]
    # Shape of a single input sample, e.g., (1, 28, 28) for MNIST
    input_shape: Tuple[int, ...]
    # Number of output classes or features
    output_shape: int
    # A callable that returns the loss function instance
    loss_function_factory: Callable[[], Any]
    # The name of the primary performance metric, e.g., "accuracy"
    performance_metric_name: str


@dataclass(frozen=True)
class ModelDefinition:
    """
    A metadata container for a model/algorithm.
    """

    name: str
    description: str
    # The actual model class (e.g., a subclass of nn.Module)
    model_class: Type[Any]
    # List of dataset types this model is compatible with
    supported_dataset_types: List[DatasetType]
    # A dictionary defining the configurable hyperparameters and their properties
    # e.g., {'dropout_rate': {'type': 'float', 'min': 0.1, 'max': 0.7, 'default': 0.5}}
    hyperparameter_schema: dict = field(default_factory=dict)
