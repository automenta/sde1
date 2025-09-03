from abc import ABC, abstractmethod
from typing import List, Dict
import math

from sde.core.types import Trial, WorkUnit, WorkUnitType, TrialStatus

class AdaptiveScheduler(ABC):
    """
    Abstract base class for an adaptive scheduler policy.
    It determines what work to do next based on intermediate results.
    """
    @abstractmethod
    def get_initial_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Returns the first batch of WorkUnits to start an experiment."""
        ...

    @abstractmethod
    def get_next_work_units(self, finished_trial: Trial, all_trials: Dict[str, Trial]) -> List[WorkUnit]:
        """
        Determines the next WorkUnits to schedule based on the result of a
        just-finished trial and the state of all other trials.
        """
        ...

class SuccessiveHalvingScheduler(AdaptiveScheduler):
    """
    Implements the Successive Halving algorithm.

    This scheduler runs trials for a certain number of epochs (a "rung"),
    then prunes the worst-performing half and continues with the survivors.
    """
    def __init__(self, metric: str, increasing: bool, min_epochs_per_rung: int = 1, reduction_factor: int = 2, max_rungs: int = 5):
        self.metric = metric
        self.increasing = increasing # True if higher metric value is better (e.g., accuracy)
        self.min_epochs_per_rung = min_epochs_per_rung
        self.eta = reduction_factor
        self.rung_level = 0
        self.max_rungs = max_rungs

    def get_initial_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Schedules the first epoch for all pending trials."""
        work_units = []
        for trial in trials.values():
            if trial.status == TrialStatus.PENDING:
                trial.status = TrialStatus.ACTIVE
                work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))
        return work_units

    def get_next_work_units(self, finished_trial: Trial, all_trials: Dict[str, Trial]) -> List[WorkUnit]:
        """
        Decides the next work units. It checks if a pruning decision can be made,
        terminates if max_rungs is reached, and schedules work to prevent stalls.
        """
        # --- Termination Condition ---
        if self.rung_level >= self.max_rungs:
            for trial in all_trials.values():
                if trial.status == TrialStatus.ACTIVE:
                    trial.status = TrialStatus.COMPLETED
            return [] # Stop scheduling new work

        # --- Rung-based Pruning Logic ---
        current_rung_epoch = self.min_epochs_per_rung * (self.eta ** self.rung_level)
        active_trials = [t for t in all_trials.values() if t.status == TrialStatus.ACTIVE]

        # A decision can only be made if all active trials have reached the rung epoch.
        if all(t.current_epoch >= current_rung_epoch for t in active_trials):
            # Get performance of each trial at the rung
            rung_performance = []
            for trial in active_trials:
                try:
                    metric_value = next(m for e, m in trial.results[self.metric] if e >= current_rung_epoch)
                    rung_performance.append((trial, metric_value))
                except (KeyError, StopIteration):
                    rung_performance.append((trial, -math.inf if self.increasing else math.inf))

            rung_performance.sort(key=lambda x: x[1], reverse=self.increasing)
            num_to_keep = math.ceil(len(rung_performance) / self.eta)
            survivors = rung_performance[:num_to_keep]
            pruned = rung_performance[num_to_keep:]

            new_work_units = []
            for trial, _ in pruned:
                trial.status = TrialStatus.PRUNED
            for trial, _ in survivors:
                new_work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))

            self.rung_level += 1
            return new_work_units

        # --- Default Scheduling (No Pruning Decision) ---
        # If not all trials are at a rung, just schedule the next work unit for the finished trial
        # to prevent the system from stalling.
        if finished_trial.status == TrialStatus.ACTIVE:
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        return []
