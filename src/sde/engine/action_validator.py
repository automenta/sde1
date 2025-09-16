from typing import Any
from typing import Dict

from sde.core.domain import Experiment
from sde.core.domain import ExperimentStatus
from sde.core.domain import TrialStatus


class ActionValidator:
    """A dedicated class for validating actions against the current experiment state.

    This encapsulates the logic of which actions are permitted in which states.
    """

    @staticmethod
    def get_valid_actions(experiment: Experiment) -> Dict[str, Any]:
        """Inspect the current state and return a structured dictionary of valid actions.

        e.g., {
            "global": ["ADD_ALGORITHM"],
            "algorithms": { "algo_1": ["UPDATE_PARAM_SPACE", "REMOVE_ALGORITHM"] },
            "trials": { "trial_abc": ["MANUAL_PRUNE_TRIAL"] }
        }
        """
        status = experiment.status
        actions: Dict[str, Any] = {
            "global": [],
            "algorithms": {},
            "trials": {},
        }
        has_challenge = experiment.challenge is not None

        # Global actions
        if status == ExperimentStatus.DEFINING:
            actions["global"].append("LOAD_EXPERIMENT")
            if not has_challenge:
                actions["global"].append("SET_CHALLENGE")
            else:
                actions["global"].append("ADD_ALGORITHM")
                actions["global"].append("SET_ADAPTIVE_POLICY")
                actions["global"].append("SET_BUDGET")
                if experiment.algorithms:
                    actions["global"].append("START_RUN")

        elif status == ExperimentStatus.RUNNING:
            actions["global"].append("PAUSE_RUN")
            actions["global"].append("STOP_RUN")
            actions["global"].append("SAVE_EXPERIMENT")
            actions["global"].append("ADD_ALGORITHM")
            actions["global"].append("SET_ADAPTIVE_POLICY")
            actions["global"].append("SET_BUDGET")

        elif status == ExperimentStatus.PAUSED:
            actions["global"].append("RESUME_RUN")
            actions["global"].append("STOP_RUN")
            actions["global"].append("SAVE_EXPERIMENT")
            actions["global"].append("ADD_ALGORITHM")
            actions["global"].append("SET_ADAPTIVE_POLICY")
            actions["global"].append("SET_BUDGET")

        # Per-algorithm actions
        for algo_id, algo in experiment.algorithms.items():
            algo_actions = []
            if status == ExperimentStatus.DEFINING:
                algo_actions.append("UPDATE_PARAM_SPACE")
                algo_actions.append("REMOVE_ALGORITHM")
            elif (
                status == ExperimentStatus.RUNNING or status == ExperimentStatus.PAUSED
            ):
                algo_actions.append("UPDATE_PARAM_SPACE")
                algo_actions.append("REMOVE_ALGORITHM")

            if algo_actions:
                actions["algorithms"][algo_id] = algo_actions

        # Per-trial actions
        for trial_id, trial in experiment.trials.items():
            trial_actions = []
            # Can spawn from any trial that has finished at least one step
            if trial.results:
                trial_actions.append("SPAWN_SIMILAR_TRIAL")

            if trial.status == TrialStatus.ACTIVE:
                trial_actions.append("MANUAL_PRUNE_TRIAL")
                trial_actions.append("MANUAL_PRIORITIZE_TRIAL")

            if trial_actions:
                actions["trials"][trial_id] = trial_actions

        return actions

    @staticmethod
    def is_action_valid(
        action_type: str, payload: Dict[str, Any], valid_actions: Dict
    ) -> bool:
        """Check if a given action is present in the structured valid_actions dict.

        This is a strict validator that uses a "default-deny" policy.
        """
        # Determine context from payload keys
        context_id = None
        actions_dict = {}

        if "algorithm_id" in payload:
            context_id = payload["algorithm_id"]
            actions_dict = valid_actions.get("algorithms", {})
        elif "trial_id" in payload:
            context_id = payload["trial_id"]
            actions_dict = valid_actions.get("trials", {})
        elif "source_trial_id" in payload:
            context_id = payload["source_trial_id"]
            actions_dict = valid_actions.get("trials", {})
        else:
            # Global action
            return action_type in valid_actions.get("global", [])

        # Context-specific action
        if context_id:
            return action_type in actions_dict.get(context_id, [])

        return False
