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
    Implements the Successive Halving algorithm (SHA) in a stateless manner.

    This scheduler runs a set of trials for a certain number of epochs (a "rung"),
    then prunes the worst-performing half and continues with the survivors.
    This process is repeated, forming subsequent rungs with more epochs but fewer trials.
    The state (current rung, participating trials) is derived from the trial data itself.
    """
    def __init__(self, metric: str, increasing: bool, min_epochs_per_rung: int = 1, reduction_factor: int = 2):
        self.metric = metric
        self.increasing = increasing
        self.min_epochs_per_rung = min_epochs_per_rung
        self.eta = reduction_factor

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
        Checks if a rung is complete, prunes underperforming trials, and schedules
        work for the survivors. The logic is stateless and derives progress from trial data.
        """
        # 1. Identify the set of currently active trials. These are the contenders for the current rung.
        active_trials = [t for t in all_trials.values() if t.status == TrialStatus.ACTIVE]
        if not active_trials:
            return [] # No active trials left to schedule.

        # 2. Determine the minimum epoch completed by all active trials. This defines the current frontier.
        min_completed_epoch = min(t.current_epoch for t in active_trials)

        # 3. Calculate all possible rung epoch targets.
        # Note: This assumes the number of active_trials doesn't increase mid-experiment.
        try:
            num_rungs = int(math.log(len(active_trials), self.eta)) + 1
        except ValueError: # log(0)
             num_rungs = 0
        rung_epochs = {self.min_epochs_per_rung * (self.eta ** i) for i in range(num_rungs)}

        # 4. If the minimum completed epoch is not a rung, we are between rungs. Continue training.
        if min_completed_epoch not in rung_epochs:
            if finished_trial.status == TrialStatus.ACTIVE:
                return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]
            return []

        # 5. We are at a decision point (a rung). Time to prune.
        rung_target_epoch = min_completed_epoch
        rung_performance = []
        for trial in active_trials:
            try:
                # Find the metric value at exactly the target epoch for a fair comparison.
                metric_value = next(m for e, m in trial.results[self.metric] if e == rung_target_epoch)
                rung_performance.append((trial, metric_value))
            except (KeyError, StopIteration):
                # This trial is missing the required metric. Prune it by giving it the worst possible score.
                worst_score = -math.inf if self.increasing else math.inf
                rung_performance.append((trial, worst_score))

        # Sort trials by performance (best to worst).
        rung_performance.sort(key=lambda x: x[1], reverse=self.increasing)

        # Prune the worst-performing trials.
        num_to_keep = math.ceil(len(rung_performance) / self.eta)
        survivors = rung_performance[:num_to_keep]
        pruned = rung_performance[num_to_keep:]

        for trial, _ in pruned:
            trial.status = TrialStatus.PRUNED

        # Schedule the next epoch for the survivors.
        return [WorkUnit(trial_id=t.id, type=WorkUnitType.TRAIN_EPOCH) for t, _ in survivors]


from dataclasses import dataclass, field

@dataclass
class _Bracket:
    """Helper class to manage the state of a single Hyperband bracket."""
    s: int
    num_trials: int
    rung: int = 0
    trial_ids: List[str] = field(default_factory=list)
    rung_resources: List[int] = field(default_factory=list)

class HyperbandScheduler(AdaptiveScheduler):
    """
    Implements the Hyperband algorithm.

    Hyperband is a more advanced version of Successive Halving that automates
    the trade-off between the number of trials to run and the number of epochs
    allocated to each. It does this by running several brackets of Successive
    Halving with different configurations. This implementation assigns all trials
    upfront and runs the brackets in parallel.
    """
    def __init__(self, metric: str, increasing: bool, max_resource_per_trial: int, reduction_factor: int = 3):
        self.metric = metric
        self.increasing = increasing
        self.max_resource = max_resource_per_trial
        self.eta = reduction_factor
        self.s_max = int(math.log(self.max_resource, self.eta))

        self.brackets: List[_Bracket] = []
        self.trial_to_bracket: Dict[str, _Bracket] = {}

    def get_initial_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """
        Calculates brackets, assigns trials to them, and returns the first set of work units.
        """
        trial_pool = [t for t in trials.values() if t.status == TrialStatus.PENDING]

        # 1. Calculate bracket configurations based on the original Hyperband paper.
        for s in range(self.s_max, -1, -1):
            # n_s: number of trials for a bracket `s`
            n_s = math.ceil((self.s_max + 1) / (s + 1) * (self.eta ** s))
            # r_s: min resource (e.g., epochs) per trial in bracket `s`
            r_s = self.max_resource * (self.eta ** -s)

            bracket = _Bracket(s=s, num_trials=n_s)

            # Calculate the resource for each rung within this bracket's SHA run
            for i in range(s + 1):
                rung_resource = r_s * (self.eta ** i)
                bracket.rung_resources.append(int(rung_resource))

            self.brackets.append(bracket)

        # 2. Assign trials to brackets
        for bracket in self.brackets:
            if not trial_pool:
                # Not enough trials to fill all brackets, which is acceptable.
                break

            num_to_assign = min(len(trial_pool), bracket.num_trials)
            assigned_trials = trial_pool[:num_to_assign]
            trial_pool = trial_pool[num_to_assign:]

            bracket.trial_ids = [t.id for t in assigned_trials]
            for trial in assigned_trials:
                self.trial_to_bracket[trial.id] = bracket

        # 3. Create initial work units for all assigned trials
        work_units = []
        for trial_id in self.trial_to_bracket.keys():
            trials[trial_id].status = TrialStatus.ACTIVE
            work_units.append(WorkUnit(trial_id=trial_id, type=WorkUnitType.TRAIN_EPOCH))

        return work_units

    def get_next_work_units(self, finished_trial: Trial, all_trials: Dict[str, Trial]) -> List[WorkUnit]:
        """
        Routes the trial to the correct bracket and applies SHA logic within it.
        """
        bracket = self.trial_to_bracket.get(finished_trial.id)
        if not bracket or finished_trial.status != TrialStatus.ACTIVE:
            return [] # This trial is not managed by this scheduler or has been pruned/completed.

        # If bracket is done, no more work
        if bracket.rung >= len(bracket.rung_resources):
            finished_trial.status = TrialStatus.COMPLETED
            return []

        active_bracket_trials = [
            all_trials[tid] for tid in bracket.trial_ids if tid in all_trials and all_trials[tid].status == TrialStatus.ACTIVE
        ]
        if not active_bracket_trials:
            return []

        rung_target_epoch = bracket.rung_resources[bracket.rung]

        # If this trial hasn't reached the rung target, schedule its next epoch
        if finished_trial.current_epoch < rung_target_epoch:
            return [WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)]

        # This trial is at the target. Are all other trials in the bracket also ready?
        min_epoch_in_bracket = min(t.current_epoch for t in active_bracket_trials)
        if min_epoch_in_bracket < rung_target_epoch:
            return [] # Wait for other trials to catch up

        # Decision time: All trials in the bracket's active set have reached the rung.
        rung_performance = []
        for trial in active_bracket_trials:
            try:
                metric_value = next(m for e, m in trial.results[self.metric] if e == rung_target_epoch)
                rung_performance.append((trial, metric_value))
            except (KeyError, StopIteration):
                worst_score = -math.inf if self.increasing else math.inf
                rung_performance.append((trial, worst_score))

        rung_performance.sort(key=lambda x: x[1], reverse=self.increasing)

        num_to_keep = math.ceil(len(rung_performance) / self.eta)
        survivors = rung_performance[:num_to_keep]
        pruned = rung_performance[num_to_keep:]

        for trial, _ in pruned:
            trial.status = TrialStatus.PRUNED

        # Advance the bracket to the next rung
        bracket.rung += 1

        # Schedule work for survivors
        return [WorkUnit(trial_id=t.id, type=WorkUnitType.TRAIN_EPOCH) for t, _ in survivors]
