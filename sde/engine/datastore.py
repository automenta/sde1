import copy
import threading
from typing import Dict, List, Optional

from sde.core.types import Trial, TrialStatus, WorkUnit, WorkUnitType


class DataStore:
    """
    A thread-safe container for all experiment-related data, primarily the state
    of all trials. It is the sole component responsible for mutating trial state.
    """

    def __init__(self, trials: List[Trial]):
        self._trials: Dict[str, Trial] = {t.id: t for t in trials}
        self._lock = threading.Lock()

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """
        Retrieves a deep copy of a single trial by its ID to ensure the
        caller cannot mutate the datastore's state.
        """
        with self._lock:
            trial = self._trials.get(trial_id)
            return copy.deepcopy(trial) if trial else None

    def add_trial(self, trial: Trial) -> None:
        """Adds a new trial to the datastore in a thread-safe manner."""
        with self._lock:
            if trial.id in self._trials:
                # Or raise an error, but logging is safer for concurrency
                return
            self._trials[trial.id] = trial

    def get_all_trials(self) -> Dict[str, Trial]:
        """
        Returns a deep copy of the dictionary of all trials to ensure the
        caller cannot mutate the datastore's state.
        """
        with self._lock:
            return {
                trial_id: copy.deepcopy(trial)
                for trial_id, trial in self._trials.items()
            }

    def record_work_unit_result(
        self, work_unit: WorkUnit, result: dict
    ) -> Optional[Trial]:
        """
        Updates a trial's state based on the results from a work unit.
        This includes metrics, epoch count, checkpoint paths, and profile info.
        This is the single entry point for mutating trial state from work results.
        """
        with self._lock:
            trial = self._trials.get(work_unit.trial_id)
            if not trial:
                return None

            if "error" in result:
                # In the future, we might set a 'FAILED' status here.
                # For now, we just don't process the result.
                return trial  # Return the unchanged trial

            if work_unit.type == WorkUnitType.TRAIN_EPOCH:
                state_updates = result.get("state_updates", {})
                trial.current_epoch = state_updates.get(
                    "current_epoch", trial.current_epoch
                )
                if state_updates.get("checkpoint_path") is not None:
                    trial.checkpoint_path = state_updates["checkpoint_path"]

                for name, value in result.get("metrics", {}).items():
                    trial.results.setdefault(name, []).append(
                        (trial.current_epoch, value)
                    )

            elif work_unit.type == WorkUnitType.PROFILE_SPEED:
                trial.est_time_per_epoch = result.get("time")

            return trial

    def update_trial_status(self, trial_id: str, status: TrialStatus) -> None:
        """Updates the status of a single trial."""
        with self._lock:
            trial = self._trials.get(trial_id)
            if trial:
                trial.status = status
