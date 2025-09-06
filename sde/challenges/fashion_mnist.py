from torchvision import datasets, transforms

from sde.challenges.factory import create_image_classification_challenge

# Define the specific transform for Fashion-MNIST
# Mean: 0.2860, Std: 0.3530 (calculated from the training set)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]
)

# Use the factory to create the dataset definition
FASHION_MNIST_DATASET = create_image_classification_challenge(
    name="Fashion-MNIST",
    description="A dataset of 70,000 28x28 grayscale images of 10 types of clothing items.",
    dataset_class=datasets.FashionMNIST,
    transform=transform,
    input_shape=(1, 28, 28),
    output_shape=10,
    data_dir="./data_fashion_mnist",
)
