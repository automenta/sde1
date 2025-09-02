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
    def __init__(self, metric: str, increasing: bool, min_epochs_per_rung: int = 1, reduction_factor: int = 2):
        self.metric = metric
        self.increasing = increasing # True if higher metric value is better (e.g., accuracy)
        self.min_epochs_per_rung = min_epochs_per_rung
        self.eta = reduction_factor
        self.rung_level = 0

    def get_initial_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Schedules the first epoch for all pending trials."""
        work_units = []
        for trial in trials.values():
            if trial.status == TrialStatus.PENDING:
                trial.status = TrialStatus.ACTIVE
                work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))
        return work_units

    def get_next_work_units(self, finished_trial: Trial, all_trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Checks if a rung is complete and prunes trials if so."""

        # 1. Determine the current rung's target epoch
        # This is a simplified rung calculation. A full implementation of Hyperband would be more complex.
        current_rung_epoch = self.min_epochs_per_rung * (self.eta ** self.rung_level)

        # Only proceed if the finished trial has completed the target number of epochs for this rung
        if finished_trial.current_epoch < current_rung_epoch:
            # Not at a rung decision point yet, just schedule the next epoch for this trial
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        # 2. Check if all other active trials have also completed this rung
        active_trials = [t for t in all_trials.values() if t.status == TrialStatus.ACTIVE]
        if not all(t.current_epoch >= current_rung_epoch for t in active_trials):
            # Not all trials are at the decision point yet. Wait for others to finish.
            return []

        # 3. All trials are at the decision point. Time to prune.
        # Get the performance of each trial at the current rung epoch
        rung_performance = []
        for trial in active_trials:
            try:
                metric_value = next(m for e, m in trial.results[self.metric] if e == current_rung_epoch)
                rung_performance.append((trial, metric_value))
            except (KeyError, StopIteration):
                # This trial doesn't have the required metric. Prune it by default.
                rung_performance.append((trial, -math.inf if self.increasing else math.inf))

        # Sort trials by performance
        rung_performance.sort(key=lambda x: x[1], reverse=self.increasing)

        # 4. Prune the worst trials
        num_to_keep = math.ceil(len(rung_performance) / self.eta)
        survivors = rung_performance[:num_to_keep]
        pruned = rung_performance[num_to_keep:]

        new_work_units = []
        for trial, _ in pruned:
            trial.status = TrialStatus.PRUNED
            # No work unit, effectively stopping it. The main scheduler will emit an update.

        for trial, _ in survivors:
            # Schedule the next epoch for the survivors
            new_work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))

        # 5. Advance to the next rung
        self.rung_level += 1

        return new_work_units
