import torch
import torch.nn as nn
import torch.nn.functional as F
from sde.models.types import DatasetType
from sde.models.types import ModelDefinition
from sde.models.types import SdeModel


class SimpleCNN(SdeModel):
    """A simple, generic CNN for image classification.
    The architecture is adaptable based on input shape and output shape.
    """

    def __init__(self, input_shape, output_shape, dropout_rate=0.5, **kwargs):
        super().__init__(input_shape, output_shape)

        input_channels, image_height, image_width = self.input_shape
        num_classes = self.output_shape

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the flattened size dynamically after convolution and pooling
        # Each max pooling layer with kernel size 2 reduces height and width by half.
        final_height = image_height // 4
        final_width = image_width // 4
        fc_input_features = 64 * final_height * final_width

        self.fc1 = nn.Linear(fc_input_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
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
        return x


# --- Model Definition ---

SIMPLE_CNN_MODEL = ModelDefinition(
    name="SimpleCNN",
    description="A simple Convolutional Neural Network for image classification.",
    model_class=SimpleCNN,
    supported_dataset_types=[DatasetType.IMAGE_CLASSIFICATION],
    hyperparameter_schema={
        "model_params": {
            "dropout_rate": {"type": "float", "min": 0.0, "max": 0.9, "default": 0.5}
        },
        "optimizer_params": {
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "default": 1e-3}
        },
    },
)
