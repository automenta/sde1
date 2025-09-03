from sde.challenges.mnist import MNIST_DATASET
from sde.challenges.cifar10 import CIFAR10_DATASET
from sde.challenges.fashion_mnist import FASHION_MNIST_DATASET

AVAILABLE_DATASETS = {
    MNIST_DATASET.name: MNIST_DATASET,
    FASHION_MNIST_DATASET.name: FASHION_MNIST_DATASET,
    CIFAR10_DATASET.name: CIFAR10_DATASET,
}
