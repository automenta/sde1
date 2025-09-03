"""
The DataStore is a single source of truth for all data related to an experiment.

It holds the state of all trials and provides a consistent, centralized API
for other components like the Scheduler, AdaptiveScheduler, and InsightEngine
to access and modify that data. This avoids state inconsistencies and makes
the system easier to reason about.
"""
from typing import Dict, List, Any

from sde.core.types import Trial, TrialStatus

class DataStore:
    """
    A centralized, in-memory data store for all experiment data.
    It provides a single, consistent interface for accessing and manipulating
    trial information, abstracting away the underlying dictionary implementation.
    """
    def __init__(self, trials: List[Trial]):
        self._trials: Dict[str, Trial] = {t.id: t for t in trials}

    def get_trial(self, trial_id: str) -> Trial:
        """Retrieves a single trial by its ID."""
        return self._trials[trial_id]

    def get_all_trials(self) -> List[Trial]:
        """Returns a list of all trials in the store."""
        return list(self._trials.values())

    def get_trials_by_status(self, status: TrialStatus) -> List[Trial]:
        """Returns a list of all trials with a given status."""
        return [t for t in self._trials.values() if t.status == status]

    def update_trial_from_result(self, trial_id: str, result: Dict[str, Any]):
        """
        Updates a trial's state based on the results from a worker.

        Args:
            trial_id: The ID of the trial to update.
            result: The result dictionary returned by the worker.
        """
        trial = self.get_trial(trial_id)

        # Update state from 'state_updates'
        state_updates = result.get('state_updates', {})
        trial.current_epoch = state_updates.get('current_epoch', trial.current_epoch)

        # Only update checkpoint path if a new one was created and is not None
        new_checkpoint_path = state_updates.get('checkpoint_path')
        if new_checkpoint_path:
            trial.checkpoint_path = new_checkpoint_path

        # Append new metrics to the results time-series
        metrics = result.get('metrics', {})
        for metric_name, value in metrics.items():
            if metric_name not in trial.results:
                trial.results[metric_name] = []
            trial.results[metric_name].append((trial.current_epoch, value))

    def set_trial_status(self, trial_id: str, status: TrialStatus):
        """Explicitly sets the status of a trial."""
        trial = self.get_trial(trial_id)
        trial.status = status
