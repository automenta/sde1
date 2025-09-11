import numpy as np
import torch
import torch.nn as nn

from sde.core.types import DatasetType
from sde.core.types import ModelDefinition
from sde.core.types import SdeModel


class LogisticRegression(SdeModel):
    """A simple Logistic Regression model implemented as a PyTorch module.
    It flattens the input and applies a single linear layer.
    """

    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__(input_shape, output_shape)
        # Calculate the total number of input features from the input shape
        input_features = int(np.prod(self.input_shape))
        self.linear = nn.Linear(input_features, self.output_shape)

    def forward(self, x):
        # Flatten the input tensor to (batch_size, num_features)
        x = torch.flatten(x, 1)
        return self.linear(x)


# --- Model Definition ---

LOGISTIC_REGRESSION_MODEL = ModelDefinition(
    name="LogisticRegression",
    description="A classical Logistic Regression model for baseline performance.",
    model_class=LogisticRegression,
    supported_dataset_types=[DatasetType.IMAGE_CLASSIFICATION],
    hyperparameter_schema={
        "optimizer_params": {
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "default": 1e-3}
        }
    },
)
