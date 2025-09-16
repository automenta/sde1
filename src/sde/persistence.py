import json
import logging
import os
import traceback

from .core.domain import Experiment

logger = logging.getLogger(__name__)


def save_experiment(experiment: Experiment, filepath: str) -> None:
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
    except (IOError, TypeError):
        logger.error(
            f"Error saving experiment to {filepath}:\n{traceback.format_exc()}"
        )
        raise


def load_experiment(filepath: str) -> Experiment:
    """Loads an Experiment object from a JSON file.

    Args:
        filepath: The path to the file from which to load the experiment.

    Returns:
        An Experiment object.

    """
    try:
        with open(filepath, "r") as f:
            state_dict = json.load(f)
        return Experiment.from_dict(state_dict)
    except (IOError, json.JSONDecodeError, KeyError):
        logger.error(
            f"Error loading experiment from {filepath}:\n{traceback.format_exc()}"
        )
        raise
