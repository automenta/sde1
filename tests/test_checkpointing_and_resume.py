import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sde.core.domain import Trial
from sde.core.domain import WorkUnit
from sde.core.domain import WorkUnitType
from sde.discovery import discover_and_register_components
from sde.engine.worker import Worker
from sde.registry import registry


class TestCheckpointingAndResume(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        # Ensure components are registered before running tests
        registry.challenges.clear()
        registry.models.clear()
        registry.schedulers.clear()
        discover_and_register_components()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    @patch("sde.engine.trial_checkpoint_manager.get_checkpoints_dir")
    def test_training_progresses_across_epochs_with_checkpointing(
        self, mock_get_checkpoints_dir
    ):
        """Tests the core functionality of checkpointing and resuming.
        It runs two consecutive epochs and asserts that the model's performance
        improves, which proves that the state was correctly saved and reloaded.
        """
        # Make the mock return the path to our temporary directory
        mock_get_checkpoints_dir.return_value = Path(self.temp_dir)

        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # 1. Setup: Use a real model (MLP) and dataset (MNIST) for an integration test.
        from torchvision import datasets
        from torchvision import transforms

        from sde.challenges.factory import create_image_classification_challenge
        from sde.core.definitions import DatasetType
        from sde.core.definitions import ModelDefinition
        from sde.models.classical.mlp import MLP

        model_def = ModelDefinition(
            name="MLP",
            description="A test MLP model.",
            model_class=MLP,
            model_type=DatasetType.IMAGE_CLASSIFICATION,
        )
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset_def = create_image_classification_challenge(
            name="MNIST",
            dataset_class=datasets.MNIST,
            transform=transform,
            input_shape=(1, 28, 28),
            output_shape=10,
            description="",
            data_dir=self.temp_dir,
        )

        # A minimal trial object.
        trial = Trial(
            id="test_trial_checkpoint",
            algorithm_name="MLP",
            hyperparameters={
                "model_params": {"layer_sizes": [32]},
                "optimizer_params": {"name": "Adam", "lr": 0.01},
                "loader_params": {"batch_size": 128},
            },
        )

        # --- EPOCH 1 ---
        # 2. Execute the first epoch.
        worker1 = Worker(model_def=model_def, dataset_def=dataset_def)
        work_unit1 = WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
        result1 = worker1.execute_work_unit(
            work_unit1, trial, enable_checkpointing=True
        )

        # 3. Assertions for Epoch 1
        self.assertNotIn("error", result1, "Worker returned an error on first epoch.")
        metric_name = dataset_def.performance_metric_name
        self.assertIn(metric_name, result1["metrics"])
        accuracy1 = result1["metrics"][metric_name]
        self.assertGreater(
            accuracy1, 0.7, "Model should achieve decent accuracy on MNIST"
        )

        checkpoint_path1 = result1["state_updates"]["checkpoint_path"]
        self.assertIsNotNone(checkpoint_path1, "Checkpoint path should not be None.")
        self.assertTrue(
            os.path.exists(checkpoint_path1), "Checkpoint file was not created."
        )

        # --- EPOCH 2 ---
        # 4. Update trial state to simulate what the Orchestrator would do.
        trial.current_epoch = result1["state_updates"]["current_epoch"]
        trial.checkpoint_path = checkpoint_path1

        # 5. Execute the second epoch with a NEW worker to simulate a new process.
        worker2 = Worker(model_def=model_def, dataset_def=dataset_def)
        work_unit2 = WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
        result2 = worker2.execute_work_unit(
            work_unit2, trial, enable_checkpointing=True
        )

        # 6. Assertions for Epoch 2
        self.assertNotIn("error", result2, "Worker returned an error on second epoch.")
        self.assertIn(metric_name, result2["metrics"])
        accuracy2 = result2["metrics"][metric_name]

        # The key assertion: performance should improve if state was loaded.
        self.assertGreater(
            accuracy2,
            accuracy1,
            f"Accuracy should improve on the second epoch. Got: {accuracy2}, expected > {accuracy1}",
        )

        checkpoint_path2 = result2["state_updates"]["checkpoint_path"]
        self.assertEqual(
            checkpoint_path1,
            checkpoint_path2,
            "The checkpoint path should be canonical and not change between epochs.",
        )


if __name__ == "__main__":
    unittest.main()
