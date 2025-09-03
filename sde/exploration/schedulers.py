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
    Implements the Successive Halving algorithm (SHA).

    This scheduler runs a set of trials for a certain number of epochs (a "rung"),
    then prunes the worst-performing half and continues with the survivors.
    This process is repeated, forming subsequent rungs with more epochs but fewer trials.
    """
    def __init__(self, metric: str, increasing: bool, min_epochs_per_rung: int = 1, reduction_factor: int = 2):
        self.metric = metric
        self.increasing = increasing
        self.min_epochs_per_rung = min_epochs_per_rung
        self.eta = reduction_factor
        self.rung_level = 0
        self.trials_in_rung: List[str] = [] # Holds the IDs of trials currently in the active rung

    def get_initial_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Schedules the first epoch for all pending trials and initializes the first rung."""
        work_units = []
        for trial in trials.values():
            if trial.status == TrialStatus.PENDING:
                trial.status = TrialStatus.ACTIVE
                self.trials_in_rung.append(trial.id)
                work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))
        return work_units

    def get_next_work_units(self, finished_trial: Trial, all_trials: Dict[str, Trial]) -> List[WorkUnit]:
        """
        Checks if a rung is complete, prunes underperforming trials, and schedules
        work for the survivors. This logic is now more robust against slow trials.
        """
        # 1. Determine the current rung's target epoch.
        current_rung_epoch = self.min_epochs_per_rung * (self.eta ** self.rung_level)

        # 2. If the finished trial hasn't reached the rung's epoch target, just schedule its next epoch.
        if finished_trial.current_epoch < current_rung_epoch:
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        # 3. The finished trial has reached the rung. Check if the whole rung is complete.
        # A rung is complete if all trials that started in this rung have reached the target epoch.
        rung_trials = [all_trials[tid] for tid in self.trials_in_rung if tid in all_trials]
        completed_rung_trials = [t for t in rung_trials if t.current_epoch >= current_rung_epoch]

        if len(completed_rung_trials) < len(self.trials_in_rung):
            # The rung is not yet complete. Wait for other trials to finish.
            return []

        # 4. The rung is complete. Time to prune.
        rung_performance = []
        for trial in completed_rung_trials:
            try:
                # Find the metric value at exactly the target epoch for a fair comparison.
                metric_value = next(m for e, m in trial.results[self.metric] if e == current_rung_epoch)
                rung_performance.append((trial, metric_value))
            except (KeyError, StopIteration):
                # This trial is missing the required metric. Prune it by giving it the worst possible score.
                worst_score = -math.inf if self.increasing else math.inf
                rung_performance.append((trial, worst_score))

        # Sort trials by performance (best to worst).
        rung_performance.sort(key=lambda x: x[1], reverse=self.increasing)

        # 5. Prune the worst-performing trials.
        num_to_keep = math.ceil(len(rung_performance) / self.eta)
        survivors = rung_performance[:num_to_keep]
        pruned = rung_performance[num_to_keep:]

        new_work_units = []
        for trial, _ in pruned:
            trial.status = TrialStatus.PRUNED

        # Update the list of trials for the next rung to only include survivors.
        self.trials_in_rung = [t.id for t, _ in survivors]

        for trial, _ in survivors:
            # Schedule the next epoch for the survivors.
            new_work_units.append(WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH))

        # 6. Advance to the next rung for the next decision point.
        self.rung_level += 1

        return new_work_units
