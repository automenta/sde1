import logging
import os
from typing import Optional

import torch

from sde.config import get_checkpoints_dir
from sde.core.domain import Trial

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrialCheckpointManager:
    """Handles saving and loading of PyTorch model checkpoints for a single trial."""

    def __init__(self):
        self.checkpoints_dir = get_checkpoints_dir()
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save(
        self, trial: Trial, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> Optional[str]:
        """Saves the state of a model and optimizer to a checkpoint file.

        Args:
            trial: The trial being checkpointed.
            model: The PyTorch model.
            optimizer: The PyTorch optimizer.

        Returns:
            The path to the saved checkpoint file, or None if saving fails.

        """
        checkpoint_path = trial.checkpoint_path
        if not checkpoint_path:
            checkpoint_filename = f"{trial.id}.pt"
            checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_filename)

        # Use a temporary file and atomic rename to prevent corruption
        temp_checkpoint_path = f"{checkpoint_path}.tmp"
        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                temp_checkpoint_path,
            )
            os.rename(temp_checkpoint_path, checkpoint_path)
            logger.debug(f"Saved checkpoint for trial {trial.id} to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint for trial {trial.id}: {e}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)
            return None

    def load(
        self, trial: Trial, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ):
        """Loads model and optimizer state from a checkpoint.

        Args:
            trial: The trial whose checkpoint should be loaded.
            model: The model to load state into.
            optimizer: The optimizer to load state into.

        """
        if not trial.checkpoint_path or not os.path.exists(trial.checkpoint_path):
            if trial.checkpoint_path:
                logger.warning(
                    f"Checkpoint file not found at {trial.checkpoint_path}. Starting from scratch."
                )
            return

        try:
            checkpoint = torch.load(trial.checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(
                f"Loaded checkpoint for trial {trial.id} from {trial.checkpoint_path}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load checkpoint for trial {trial.id} from {trial.checkpoint_path}: {e}"
            )
