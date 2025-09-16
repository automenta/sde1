from torchvision import datasets
from torchvision import transforms

from ..config import get_user_cache_dir
from .factory import create_image_classification_challenge

# Define the specific transform for MNIST
# Mean and std of MNIST are 0.1307 and 0.3081 respectively
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# Use the factory to create the dataset definition
CHALLENGE = create_image_classification_challenge(
    name="MNIST",
    description="70,000 28x28 grayscale images of handwritten digits (0-9).",
    dataset_class=datasets.MNIST,
    transform=transform,
    input_shape=(1, 28, 28),
    output_shape=10,
    data_dir=get_user_cache_dir() / "mnist",
)
