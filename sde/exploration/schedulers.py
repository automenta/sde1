from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import math

from sde.core.types import Trial, WorkUnit, WorkUnitType, TrialStatus
from sde.engine.datastore import DataStore

class AdaptiveScheduler(ABC):
    """
    Abstract base class for an adaptive scheduler policy.
    It determines what work to do next based on intermediate results.
    """
    def __init__(self, metric: str, increasing: bool):
        self.metric = metric
        self.increasing = increasing

    @abstractmethod
    def get_initial_work_units(self, datastore: DataStore) -> List[WorkUnit]:
        """Returns the first batch of WorkUnits to start an experiment."""
        ...

    @abstractmethod
    def get_next_work_units(self, finished_trial: Trial, datastore: DataStore) -> List[WorkUnit]:
        """
        Determines the next WorkUnits to schedule based on the result of a
        just-finished trial and the state of all other trials.
        """
        ...

class SuccessiveHalvingScheduler(AdaptiveScheduler):
    """
    An asynchronous, non-blocking implementation of Successive Halving.

    This scheduler evaluates trials at periodic "rungs" (decision points defined
    by powers of the reduction_factor). When a trial completes an epoch that
    falls on a rung, it compares its performance against all other active trials
    that have also reached that rung. If the trial is in the bottom tier of
    performers, it is pruned.

    This implementation is "non-blocking" because a fast trial never has to wait
    for a slow one. It makes its pruning decision based on whatever data is
    available in the DataStore at that moment.
    """
    def __init__(self, metric: str, increasing: bool, min_epochs_per_rung: int = 1, reduction_factor: int = 2):
        super().__init__(metric, increasing)
        self.min_epochs = min_epochs_per_rung
        self.eta = reduction_factor
        # A dictionary to keep track of trials that have been pruned at a specific rung
        # to avoid re-evaluating them if they somehow get more work scheduled.
        self._pruned_at_rung = {}

    def get_initial_work_units(self, datastore: DataStore) -> List[WorkUnit]:
        """Schedules the first epoch for all pending trials."""
        work_units = []
        for trial in datastore.get_trials_by_status(TrialStatus.PENDING):
            datastore.set_trial_status(trial.id, TrialStatus.ACTIVE)
            work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))
        return work_units

    def _is_rung_epoch(self, epoch: int) -> bool:
        """Checks if a given epoch is a rung (a decision point)."""
        if epoch < self.min_epochs:
            return False

        # An epoch is a rung if (epoch / min_epochs) is a power of eta.
        # We use logs to check this, with a tolerance for floating point inaccuracies.
        if self.min_epochs == 0 and epoch > 0: return True # Edge case for 0 min_epochs
        if self.min_epochs <= 0: return False

        log_val = math.log(epoch / self.min_epochs, self.eta)
        return abs(log_val - round(log_val)) < 1e-9

    def _get_performance_at_epoch(self, trial: Trial, epoch: int) -> Optional[float]:
        """Safely retrieves a trial's performance for a specific epoch."""
        try:
            return next(m for e, m in trial.results[self.metric] if e == epoch)
        except (KeyError, StopIteration):
            return None

    def get_next_work_units(self, finished_trial: Trial, datastore: DataStore) -> List[WorkUnit]:
        """Determines if a trial should be pruned or continue."""
        current_epoch = finished_trial.current_epoch

        # 1. If the current epoch is not a decision point, continue training.
        if not self._is_rung_epoch(current_epoch):
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        # 2. This is a decision point (a "rung").
        rung_epoch = current_epoch

        # Avoid re-pruning a trial that has already been marked for pruning at this rung.
        if finished_trial.id in self._pruned_at_rung.get(rung_epoch, set()):
             datastore.set_trial_status(finished_trial.id, TrialStatus.PRUNED)
             return []

        # 3. Gather all other active trials that have also reached this rung.
        contemporaries: List[Tuple[Trial, float]] = []
        for trial in datastore.get_trials_by_status(TrialStatus.ACTIVE):
            performance = self._get_performance_at_epoch(trial, rung_epoch)
            if performance is not None:
                contemporaries.append((trial, performance))

        # 4. If there aren't enough other trials to make a comparison, let this one continue.
        # This prevents a fast trial from being unfairly pruned before others have reported results.
        if len(contemporaries) < self.eta:
             return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        # 5. Sort the contenders and check the rank of the finished trial.
        contemporaries.sort(key=lambda x: x[1], reverse=self.increasing)
        try:
            rank = [t.id for t, p in contemporaries].index(finished_trial.id)
        except ValueError:
            # Should not happen, but as a safeguard, let the trial continue if it's not in the list.
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        # 6. Prune the trial if it's in the bottom tier of performers.
        num_to_prune = len(contemporaries) // self.eta
        if rank >= (len(contemporaries) - num_to_prune):
            datastore.set_trial_status(finished_trial.id, TrialStatus.PRUNED)
            # Record the pruning decision to prevent redundant checks.
            if rung_epoch not in self._pruned_at_rung:
                self._pruned_at_rung[rung_epoch] = set()
            self._pruned_at_rung[rung_epoch].add(finished_trial.id)
            return [] # No more work for this trial.
        else:
            # It's a survivor, schedule the next epoch.
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]
