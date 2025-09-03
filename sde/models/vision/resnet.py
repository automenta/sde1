import torch
import torch.nn as nn
import torch.nn.functional as F

from sde.models.types import ModelDefinition, DatasetType, SdeModel

class ResidualBlock(nn.Module):
    """A standard residual block with two convolutional layers."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ConfigurableResNet(SdeModel):
    """A ResNet-style model where the number of blocks is configurable."""
    def __init__(self, input_shape, output_shape, block_config=[2, 2, 2, 2], num_init_features=64, **kwargs):
        super().__init__(input_shape, output_shape)

        input_channels, _, _ = self.input_shape
        num_classes = self.output_shape

        self.in_channels = num_init_features
        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.layer1 = self._make_layer(ResidualBlock, num_init_features, block_config[0], stride=1)
        self.layer2 = self._make_layer(ResidualBlock, num_init_features*2, block_config[1], stride=2)
        self.layer3 = self._make_layer(ResidualBlock, num_init_features*4, block_config[2], stride=2)
        self.layer4 = self._make_layer(ResidualBlock, num_init_features*8, block_config[3], stride=2)

        self.linear = nn.Linear(num_init_features*8, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- Model Definition ---
CONFIGURABLE_RESNET_MODEL = ModelDefinition(
    name="ConfigurableResNet",
    description="A ResNet-style model where the number of blocks per layer is a configurable hyperparameter.",
    model_class=ConfigurableResNet,
    supported_dataset_types=[DatasetType.IMAGE_CLASSIFICATION],
    hyperparameter_schema={
        "model_params": {
            "block_config": {
                "type": "int_list",
                "default": [2, 2, 2, 2],
                "options": [[1,1,1,1], [2,2,2,2], [3,3,3,3]] # Example options
            },
            "num_init_features": {
                "type": "int",
                "default": 64,
                "options": [32, 64]
            }
        },
        "optimizer_params": {
            "lr": {
                "type": "float",
                "min": 1e-5,
                "max": 1e-2,
                "default": 1e-3
            }
        }
    }
)
