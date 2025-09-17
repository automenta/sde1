import json
import logging
import os

from sde.core.domain import ExperimentSnapshot

logger = logging.getLogger(__name__)


class ExperimentSerializer:
    """Handles saving and loading of the main Experiment object."""

    @staticmethod
    def save(snapshot: ExperimentSnapshot, filepath: str) -> None:
        """Serializes an ExperimentSnapshot object to a JSON file.

        Args:
            snapshot: The ExperimentSnapshot object to save.
            filepath: The path to the file where the experiment will be saved.

        """
        try:
            state_dict = snapshot.to_dict()
            with open(filepath, "w") as f:
                json.dump(state_dict, f, indent=4)
                f.flush()
                os.sync()
            logger.info(f"Successfully saved experiment snapshot to {filepath}")
        except (IOError, TypeError) as e:
            logger.error(f"Error saving experiment snapshot to {filepath}: {e}")
            raise

    @staticmethod
    def load(filepath: str) -> ExperimentSnapshot:
        """Loads an ExperimentSnapshot object from a JSON file.

        Args:
            filepath: The path to the file from which to load the experiment.

        Returns:
            An ExperimentSnapshot object.

        """
        try:
            with open(filepath, "r") as f:
                state_dict = json.load(f)
            logger.info(f"Successfully loaded experiment snapshot from {filepath}")
            return ExperimentSnapshot.from_dict(state_dict)
        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading experiment snapshot from {filepath}: {e}")
            raise
