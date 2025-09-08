import os
import random
import shutil
import tempfile
import unittest

import numpy as np
import torch
from sde.challenges import AVAILABLE_DATASETS
from sde.core.types import Trial
from sde.core.types import WorkUnit
from sde.core.types import WorkUnitType
from sde.engine.worker import Worker
from sde.models import AVAILABLE_MODELS


class TestCheckpointingAndResume(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for checkpoints."""
        self.checkpoints_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.checkpoints_dir)

    def test_training_progresses_across_epochs_with_checkpointing(self):
        """
        Tests the core functionality of checkpointing and resuming.
        It runs two consecutive epochs and asserts that the model's performance
        improves, which proves that the state was correctly saved and reloaded.
        """
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # 1. Setup: Use a real model (MLP) and dataset (MNIST) for an integration test.
        model_def = AVAILABLE_MODELS["MLP"]
        dataset_def = AVAILABLE_DATASETS["MNIST"]

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
        worker1 = Worker(
            model_def=model_def,
            dataset_def=dataset_def,
            checkpoints_dir=self.checkpoints_dir,
        )
        work_unit1 = WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
        result1 = worker1.execute_work_unit(
            work_unit1, trial, enable_checkpointing=True
        )

        # 3. Assertions for Epoch 1
        self.assertNotIn("error", result1, "Worker returned an error on first epoch.")
        metric_name = dataset_def.performance_metric_name
        self.assertIn(metric_name, result1["metrics"])
        accuracy1 = result1["metrics"][metric_name]
        self.assertGreater(accuracy1, 0.7, "Model should achieve decent accuracy on MNIST even in one epoch.")

        checkpoint_path1 = result1["state_updates"]["checkpoint_path"]
        self.assertIsNotNone(checkpoint_path1, "Checkpoint path should not be None.")
        self.assertTrue(os.path.exists(checkpoint_path1), "Checkpoint file was not created.")

        # --- EPOCH 2 ---
        # 4. Update trial state to simulate what the Orchestrator would do.
        trial.current_epoch = result1["state_updates"]["current_epoch"]
        trial.checkpoint_path = checkpoint_path1

        # 5. Execute the second epoch with a NEW worker to simulate a new process.
        worker2 = Worker(
            model_def=model_def,
            dataset_def=dataset_def,
            checkpoints_dir=self.checkpoints_dir,
        )
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
            f"Accuracy should improve on the second epoch. Epoch 1: {accuracy1}, Epoch 2: {accuracy2}",
        )

        checkpoint_path2 = result2["state_updates"]["checkpoint_path"]
        self.assertEqual(
            checkpoint_path1,
            checkpoint_path2,
            "The checkpoint path should be canonical and not change between epochs.",
        )


if __name__ == "__main__":
    unittest.main()
