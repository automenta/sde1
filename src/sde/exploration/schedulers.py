import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from sde.challenges.utils import _generate_random_hyperparameters
from sde.core.domain import AlgorithmConfig
from sde.core.domain import Trial
from sde.core.domain import TrialStatus
from sde.core.domain import WorkUnit
from sde.core.domain import WorkUnitType


class AdaptiveScheduler(ABC):
    """Abstract base class for an adaptive scheduler policy.
    It determines what work to do next based on intermediate results.
    """

    @abstractmethod
    def generate_initial_trials(
        self, algorithms: List[AlgorithmConfig], num_trials_per_algo: int
    ) -> List[Trial]:
        """Creates the initial set of trials for an experiment."""
        ...

    @abstractmethod
    def generate_work_units_for_new_trials(
        self, new_trials: List[Trial], all_trials: Dict[str, Trial]
    ) -> List[WorkUnit]:
        """Creates the initial WorkUnits for a list of newly created trials."""
        ...

    @abstractmethod
    def get_initial_work_units(self, trials: List[Trial]) -> List[WorkUnit]:
        """Returns the first batch of WorkUnits to start an experiment."""
        ...

    @abstractmethod
    def rehydrate_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Creates work units for a loaded experiment to resume from its saved state."""
        ...

    @abstractmethod
    def get_next_work_units(
        self, finished_trial: Trial, all_trials: Dict[str, Trial]
    ) -> List[WorkUnit]:
        """Determines the next WorkUnits to schedule based on the result of a
        just-finished trial and the state of all other trials.
        """
        ...


class SuccessiveHalvingScheduler(AdaptiveScheduler):
    """Implements the Successive Halving algorithm (SHA) in a stateless manner.

    This scheduler runs a set of trials for a certain number of epochs (a "rung"),
    then prunes the worst-performing half and continues with the survivors.
    This process is repeated, forming subsequent rungs with more epochs but fewer trials.
    The state (current rung, participating trials) is derived from the trial data itself.
    """

    def __init__(
        self,
        metric: str,
        increasing: bool,
        min_epochs_per_rung: int = 1,
        reduction_factor: int = 2,
    ):
        self.metric = metric
        self.increasing = increasing
        self.min_epochs_per_rung = min_epochs_per_rung
        self.eta = reduction_factor

    def generate_initial_trials(
        self, algorithms: List[AlgorithmConfig], num_trials_per_algo: int
    ) -> List[Trial]:
        """Generates a flat list of trials using random search."""
        trials = []
        trial_counter = 0
        for algo_config in algorithms:
            for _ in range(num_trials_per_algo):
                hparams = _generate_random_hyperparameters(algo_config.parameter_space)
                trial_id = f"trial_{algo_config.name.lower().replace(' ', '_')}_{trial_counter}"
                trial = Trial(
                    id=trial_id,
                    algorithm_name=algo_config.name,
                    hyperparameters=hparams,
                )
                trials.append(trial)
                trial_counter += 1
        return trials

    def generate_work_units_for_new_trials(
        self, new_trials: List[Trial], all_trials: Dict[str, Trial]
    ) -> List[WorkUnit]:
        """Schedules the first epoch for a given list of new trials."""
        work_units = []
        for trial in new_trials:
            if trial.status == TrialStatus.PENDING:
                trial.status = TrialStatus.ACTIVE
                work_units.append(
                    WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
                )
        return work_units

    def get_initial_work_units(self, trials: List[Trial]) -> List[WorkUnit]:
        """Schedules the first epoch for all pending trials in the experiment."""
        return self.generate_work_units_for_new_trials(trials, {t.id: t for t in trials})

    def rehydrate_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """For SHA, rehydration is simple: resume any trial that was active."""
        work_units = []
        for trial in trials.values():
            if trial.status == TrialStatus.ACTIVE:
                work_units.append(
                    WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
                )
        return work_units

    def get_next_work_units(
        self, finished_trial: Trial, all_trials: Dict[str, Trial]
    ) -> List[WorkUnit]:
        """Checks if a rung is complete, prunes underperforming trials, and schedules
        work for the survivors. The logic is stateless and derives progress from trial data.
        """
        # 1. Identify the set of currently active trials. These are the contenders for the current rung.
        active_trials = [
            t for t in all_trials.values() if t.status == TrialStatus.ACTIVE
        ]
        if not active_trials:
            return []  # No active trials left to schedule.

        # 2. Determine the minimum epoch completed by all active trials. This defines the current frontier.
        min_completed_epoch = min(t.current_epoch for t in active_trials)

        # 3. Calculate all possible rung epoch targets.
        # Note: This assumes the number of active_trials doesn't increase mid-experiment.
        try:
            num_rungs = int(math.log(len(active_trials), self.eta)) + 1
        except ValueError:  # log(0)
            num_rungs = 0
        rung_epochs = {
            self.min_epochs_per_rung * (self.eta**i) for i in range(num_rungs)
        }

        # 4. If the minimum completed epoch is not a rung, we are between rungs. Continue training.
        if min_completed_epoch not in rung_epochs:
            if finished_trial.status == TrialStatus.ACTIVE:
                return [
                    WorkUnit(trial_id=finished_trial.id, type=WorkUnitType.TRAIN_EPOCH)
                ]
            return []

        # 5. We are at a decision point (a rung). Time to prune.
        rung_target_epoch = min_completed_epoch
        rung_performance = []
        for trial in active_trials:
            try:
                # Find the metric value at exactly the target epoch for a fair comparison.
                metric_value = next(
                    m for e, m in trial.results[self.metric] if e == rung_target_epoch
                )
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
        return [
            WorkUnit(trial_id=t.id, type=WorkUnitType.TRAIN_EPOCH) for t, _ in survivors
        ]


@dataclass
class _Bracket:
    """Helper class to manage the state of a single Hyperband bracket."""

    s: int
    num_trials: int
    rung: int = 0
    trial_ids: List[str] = field(default_factory=list)
    rung_resources: List[int] = field(default_factory=list)


class HyperbandScheduler(AdaptiveScheduler):
    """Implements the Hyperband algorithm.

    Hyperband is a more advanced version of Successive Halving that automates
    the trade-off between the number of trials to run and the number of epochs
    allocated to each. It does this by running several brackets of Successive
    Halving with different configurations. This implementation assigns all trials
    upfront and runs the brackets in parallel.
    """

    def __init__(
        self,
        metric: str,
        increasing: bool,
        max_resource_per_trial: int,
        reduction_factor: int = 3,
    ):
        self.metric = metric
        self.increasing = increasing
        self.max_resource = max_resource_per_trial
        self.eta = reduction_factor
        self.s_max = int(math.log(self.max_resource, self.eta))

    def generate_initial_trials(
        self, algorithms: List[AlgorithmConfig], num_trials_per_algo: int
    ) -> List[Trial]:
        """Generates a flat list of trials using random search. Hyperband will later
        assign these trials to brackets. The num_trials_per_algo may not be
        fully respected, as Hyperband has specific requirements for the total
        number of trials.
        """
        trials = []
        trial_counter = 0
        for algo_config in algorithms:
            # Note: A more advanced implementation could make num_trials specific to
            # the total needed for all brackets, but random search is a good default.
            for _ in range(num_trials_per_algo):
                hparams = _generate_random_hyperparameters(algo_config.parameter_space)
                trial_id = f"trial_{algo_config.name.lower().replace(' ', '_')}_{trial_counter}"
                trial = Trial(
                    id=trial_id,
                    algorithm_name=algo_config.name,
                    hyperparameters=hparams,
                )
                trials.append(trial)
                trial_counter += 1
        return trials

    def _get_brackets(self) -> List[_Bracket]:
        """Deterministically calculates the bracket configurations."""
        brackets = []
        for s in range(self.s_max, -1, -1):
            n_s = math.ceil((self.s_max + 1) / (s + 1) * (self.eta**s))
            r_s = self.max_resource * (self.eta**-s)
            bracket = _Bracket(s=s, num_trials=n_s)
            for i in range(s + 1):
                rung_resource = r_s * (self.eta**i)
                bracket.rung_resources.append(int(rung_resource))
            brackets.append(bracket)
        return brackets

    def generate_work_units_for_new_trials(
        self, new_trials: List[Trial], all_trials: Dict[str, Trial]
    ) -> List[WorkUnit]:
        """Calculates brackets, assigns trials by tagging them, and returns initial work.
        NOTE: This implementation of Hyperband expects all trials to be provided at once.
        """
        trial_pool = [t for t in new_trials if t.status == TrialStatus.PENDING]
        brackets = self._get_brackets()

        # 1. Assign trials to brackets by tagging them
        assigned_trial_ids = set()
        for bracket in brackets:
            if not trial_pool:
                break
            num_to_assign = min(len(trial_pool), bracket.num_trials)
            assigned_trials = trial_pool[:num_to_assign]
            trial_pool = trial_pool[num_to_assign:]
            for trial in assigned_trials:
                trial.tags["hyperband_bracket_s"] = bracket.s
                assigned_trial_ids.add(trial.id)

        # 2. Create initial work units for newly assigned trials
        work_units = []
        for trial_id in assigned_trial_ids:
            all_trials[trial_id].status = TrialStatus.ACTIVE
            work_units.append(
                WorkUnit(trial_id=trial_id, type=WorkUnitType.TRAIN_EPOCH)
            )
        return work_units

    def get_initial_work_units(self, trials: List[Trial]) -> List[WorkUnit]:
        """Schedules the first epoch for all pending trials in the experiment."""
        return self.generate_work_units_for_new_trials(
            trials, {t.id: t for t in trials}
        )

    def rehydrate_work_units(self, trials: Dict[str, Trial]) -> List[WorkUnit]:
        """Schedules work for all trials that were active. State is on the trials."""
        work_units = []
        for trial in trials.values():
            if trial.status == TrialStatus.ACTIVE:
                work_units.append(
                    WorkUnit(trial_id=trial.id, type=WorkUnitType.TRAIN_EPOCH)
                )
        return work_units

    def get_next_work_units(
        self, finished_trial: Trial, all_trials: Dict[str, Trial]
    ) -> List[WorkUnit]:
        """Routes the trial to the correct bracket and applies SHA logic within it."""
        s_val = finished_trial.tags.get("hyperband_bracket_s")
        if s_val is None or finished_trial.status != TrialStatus.ACTIVE:
            return []  # This trial is not managed by this scheduler or has been pruned/completed.

        # Re-create bracket definitions on the fly; they are deterministic.
        brackets = self._get_brackets()
        try:
            bracket = next(b for b in brackets if b.s == s_val)
        except StopIteration:
            return []  # Should not happen if tags are set correctly

        # Populate the bracket's trial_ids for this specific run
        bracket.trial_ids = [
            tid
            for tid, t in all_trials.items()
            if t.tags.get("hyperband_bracket_s") == s_val
        ]

        # If bracket is done, no more work
        if bracket.rung >= len(bracket.rung_resources):
            finished_trial.status = TrialStatus.COMPLETED
            return []

        active_bracket_trials = [
            all_trials[tid]
            for tid in bracket.trial_ids
            if tid in all_trials and all_trials[tid].status == TrialStatus.ACTIVE
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
            return []  # Wait for other trials to catch up

        # Decision time: All trials in the bracket's active set have reached the rung.
        rung_performance = []
        for trial in active_bracket_trials:
            try:
                metric_value = next(
                    m for e, m in trial.results[self.metric] if e == rung_target_epoch
                )
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

        # If we have completed all rungs, the bracket is done.
        if bracket.rung >= len(bracket.rung_resources):
            for trial, _ in survivors:
                trial.status = TrialStatus.COMPLETED
            return []

        # Schedule work for survivors
        return [
            WorkUnit(trial_id=t.id, type=WorkUnitType.TRAIN_EPOCH) for t, _ in survivors
        ]
