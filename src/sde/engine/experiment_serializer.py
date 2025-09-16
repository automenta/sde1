import json
import logging
import os
import traceback
from sde.core.domain import Experiment

logger = logging.getLogger(__name__)


class ExperimentSerializer:
    """Handles saving and loading of the main Experiment object."""

    @staticmethod
    def save(experiment: Experiment, filepath: str) -> None:
        """Serializes an Experiment object to a JSON file.

        Args:
            experiment: The Experiment object to save.
            filepath: The path to the file where the experiment will be saved.
        """
        try:
            state_dict = experiment.to_dict()
            with open(filepath, "w") as f:
                json.dump(state_dict, f, indent=4)
                f.flush()
                os.sync()
            logger.info(f"Successfully saved experiment to {filepath}")
        except (IOError, TypeError) as e:
            logger.error(f"Error saving experiment to {filepath}: {e}")
            raise

    @staticmethod
    def load(filepath: str) -> Experiment:
        """Loads an Experiment object from a JSON file.

        Args:
            filepath: The path to the file from which to load the experiment.

        Returns:
            An Experiment object.
        """
        try:
            with open(filepath, "r") as f:
                state_dict = json.load(f)
            logger.info(f"Successfully loaded experiment from {filepath}")
            return Experiment.from_dict(state_dict)
        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading experiment from {filepath}: {e}")
            raise
