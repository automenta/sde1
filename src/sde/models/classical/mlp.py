import numpy as np
import torch.nn as nn

from sde.core.definitions import DatasetType
from sde.core.definitions import ModelDefinition
from sde.core.definitions import SdeModel

# --- 1. The PyTorch Model ---


class MLP(SdeModel):
    """A simple Multi-Layer Perceptron (MLP) for image classification."""

    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_sizes=[512, 256],
        dropout_rate=0.2,
        **kwargs,
    ):
        super().__init__(input_shape, output_shape, **kwargs)

        input_size = int(np.prod(self.input_shape))
        self.input_size = input_size
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h_size
        layers.append(nn.Linear(in_size, output_shape))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the image
        x = x.view(-1, self.input_size)
        return self.layers(x)


# --- 2. The Model Definition ---

MODEL = ModelDefinition(
    name="MLP",
    description="A simple Multi-Layer Perceptron for image classification.",
    model_class=MLP,
    model_type=DatasetType.IMAGE_CLASSIFICATION,
    hyperparameter_schema={
        "model_params": {
            "hidden_sizes": {
                "type": "list_int",  # This is a custom type, will need to be handled by UI
                "default": [512, 256],
                "description": "List of integers for hidden layer sizes.",
            },
            "dropout_rate": {
                "type": "float",
                "min": 0.0,
                "max": 0.7,
                "default": 0.2,
                "description": "Dropout rate for regularization.",
            },
        },
        "optimizer_params": {
            "lr": {
                "type": "float",
                "min": 1e-5,
                "max": 1e-1,
                "default": 0.001,
                "description": "Learning rate for the optimizer.",
            }
        },
    },
)
