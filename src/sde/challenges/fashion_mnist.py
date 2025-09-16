from torchvision import datasets
from torchvision import transforms

from ..config import get_user_cache_dir
from .factory import create_image_classification_challenge

# Define the specific transform for Fashion-MNIST
# Mean: 0.2860, Std: 0.3530 (calculated from the training set)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)

# Use the factory to create the dataset definition
CHALLENGE = create_image_classification_challenge(
    name="Fashion-MNIST",
    description="70,000 28x28 grayscale images of 10 clothing types.",
    dataset_class=datasets.FashionMNIST,
    transform=transform,
    input_shape=(1, 28, 28),
    output_shape=10,
    data_dir=get_user_cache_dir() / "fashion_mnist",
    version="1.0.0",
    dependencies={"torchvision": ">=0.15.0"},
)
