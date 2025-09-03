import unittest

from sde.engine.worker import Worker
from sde.core.types import Trial
from sde.challenges.mnist import MNIST_DATASET
from sde.models.vision.cnn import SimpleCNN
from sde.models.types import ModelDefinition, DatasetType

class TestWorkerBatchSize(unittest.TestCase):

    def test_batch_size_from_hyperparameters(self):
        """
        Verify that the Worker creates DataLoaders with the batch_size
        specified in the Trial's hyperparameters.
        """
        # 1. Define a sample model (needed for Worker initialization)
        model_def = ModelDefinition(
            name="TestCNN",
            description="A test model definition.",
            model_class=SimpleCNN,
            supported_dataset_types=[DatasetType.IMAGE_CLASSIFICATION]
        )

        # 2. Instantiate the Worker with the MNIST dataset definition
        # The MNIST_DATASET object already has the partial factory.
        worker = Worker(model_def=model_def, dataset_def=MNIST_DATASET)

        # 3. Create two different trials with different batch sizes
        trial1 = Trial(
            id='trial_bs_32',
            algorithm_name='Test',
            hyperparameters={'loader_params': {'batch_size': 32}}
        )
        trial2 = Trial(
            id='trial_bs_128',
            algorithm_name='Test',
            hyperparameters={'loader_params': {'batch_size': 128}}
        )
        trial3 = Trial(
            id='trial_bs_default',
            algorithm_name='Test',
            hyperparameters={} # Should use the default of 64
        )

        # 4. Call the internal method to get the dataloaders
        train_loader_1, _ = worker._get_dataloaders_for_trial(trial1)
        train_loader_2, _ = worker._get_dataloaders_for_trial(trial2)
        train_loader_3, _ = worker._get_dataloaders_for_trial(trial3)

        # 5. Assert that the batch sizes are correct
        self.assertEqual(train_loader_1.batch_size, 32)
        self.assertEqual(train_loader_2.batch_size, 128)
        self.assertEqual(train_loader_3.batch_size, 64)

        # 6. Verify caching works (calling again should return the same objects)
        train_loader_1_cached, _ = worker._get_dataloaders_for_trial(trial1)
        self.assertIs(train_loader_1, train_loader_1_cached)


if __name__ == '__main__':
    unittest.main()
