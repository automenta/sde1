import threading
from typing import Dict, List, Optional, Tuple

from sde.core.types import Trial, TrialStatus, WorkUnit


class DataStore:
    """
    A thread-safe container for all experiment-related data, primarily the state
    of all trials.
    """

    def __init__(self, trials: List[Trial]):
        self._trials: Dict[str, Trial] = {t.id: t for t in trials}
        self._lock = threading.Lock()

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Retrieves a single trial by its ID."""
        with self._lock:
            return self._trials.get(trial_id)

    def get_all_trials(self) -> Dict[str, Trial]:
        """Returns a copy of the dictionary of all trials."""
        with self._lock:
            return self._trials.copy()

    def update_trial_state(self, work_unit: WorkUnit, result: dict):
        """
        Updates a trial's state based on the results from a work unit.
        This includes metrics, epoch count, and checkpoint paths.
        """
        with self._lock:
            trial = self._trials.get(work_unit.trial_id)
            if not trial:
                return

            # Update state from worker (epoch, checkpoint)
            state_updates = result.get('state_updates', {})
            trial.current_epoch = state_updates.get('current_epoch', trial.current_epoch)
            if state_updates.get('checkpoint_path') is not None:
                trial.checkpoint_path = state_updates['checkpoint_path']

            # Append new metrics
            for name, value in result.get('metrics', {}).items():
                trial.results.setdefault(name, []).append((trial.current_epoch, value))

    def update_trial_status(self, trial_id: str, status: TrialStatus):
        """Updates the status of a single trial."""
        with self._lock:
            trial = self._trials.get(trial_id)
            if trial:
                trial.status = status

    def update_algorithm_profile(self, algorithm_name: str, est_time_per_epoch: float):
        """Updates the estimated time per epoch for all trials of a given algorithm."""
        with self._lock:
            for trial in self._trials.values():
                if trial.algorithm_name == algorithm_name:
                    trial.est_time_per_epoch = est_time_per_epoch
