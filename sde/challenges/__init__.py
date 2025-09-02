from sde.challenges.mnist import MNIST_DATASET
from sde.challenges.cifar10 import CIFAR10_DATASET

AVAILABLE_DATASETS = {
    MNIST_DATASET.name: MNIST_DATASET,
    CIFAR10_DATASET.name: CIFAR10_DATASET,
}
