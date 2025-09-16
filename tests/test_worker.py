import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch
from sde.core.definitions import DatasetType, ModelDefinition
from sde.core.domain import Trial, WorkUnit, WorkUnitType
from sde.engine.worker import Worker


# Mock SdeModel to avoid needing a real torch model
class MockModel(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.fc = torch.nn.Linear(input_shape[1], output_shape)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class TestWorker(unittest.TestCase):

    @patch('sde.engine.worker.torch.save')
    @patch('sde.challenges.utils.create_train_val_dataloaders')
    def test_batch_size_hyperparameter_is_used(self, mock_create_dataloaders, mock_torch_save):
        """Tests that the batch_size from trial hyperparameters is passed to the loader factory.
        """
        # 1. Setup mocks and test data
        # Mock the dataloaders to return some dummy data
        mock_train_loader = [(torch.randn(16, 784), torch.randint(0, 10, (16,)))]
        mock_val_loader = [(torch.randn(16, 784), torch.randint(0, 10, (16,)))]
        mock_create_dataloaders.return_value = (mock_train_loader, mock_val_loader)

        # Mock model and dataset definitions
        mock_model_def = ModelDefinition(
            name="test_model",
            description="A test model",
            model_class=MockModel,
            model_type=DatasetType.IMAGE_CLASSIFICATION
        )
        mock_dataset_def = MagicMock()
        mock_dataset_def.loader_factory.return_value = (mock_train_loader, mock_val_loader)
        mock_dataset_def.input_shape = (1, 784)
        mock_dataset_def.output_shape = 10
        mock_dataset_def.performance_metric_name = "accuracy"
        mock_dataset_def.loss_function_factory.return_value = torch.nn.CrossEntropyLoss()

        # 2. Create a Trial with a specific batch_size
        trial = Trial(
            id="test_trial_1",
            algorithm_name="test_model",
            hyperparameters={
                'loader_params': {'batch_size': 16},
                'optimizer_params': {'name': 'SGD', 'lr': 0.01}
            }
        )
        work_unit = WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)

        # 3. Instantiate the Worker and execute the work unit
        worker = Worker(model_def=mock_model_def, dataset_def=mock_dataset_def)
        # We need to manually call the internal _get_dataloaders to trigger the call
        # to the factory we want to test.
        worker._get_dataloaders(trial)

        # 4. Assert that the dataloader factory was called with the correct batch size
        # The worker now calls the loader_factory on the DatasetDefinition object
        mock_dataset_def.loader_factory.assert_called_once_with(batch_size=16)

        # Let's also test the default case
        trial_default = Trial(
            id="test_trial_2",
            algorithm_name="test_model",
            hyperparameters={} # No loader_params
        )
        worker._get_dataloaders(trial_default)
        mock_dataset_def.loader_factory.assert_called_with(batch_size=64) # 64 is the default


if __name__ == '__main__':
    unittest.main()
