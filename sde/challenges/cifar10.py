from sde.challenges.factory import create_image_classification_challenge
from torchvision import datasets
from torchvision import transforms

# Define the specific transform for CIFAR-10
# Normalize to [-1, 1]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Use the factory to create the dataset definition
CIFAR10_DATASET = create_image_classification_challenge(
    name="CIFAR-10",
    description="A dataset of 60,000 32x32 color images in 10 classes (e.g., airplane, dog, cat).",
    dataset_class=datasets.CIFAR10,
    transform=transform,
    input_shape=(3, 32, 32),
    output_shape=10,
    data_dir="./data_cifar10",
)
