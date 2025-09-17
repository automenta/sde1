"""
Handles actions dispatched from the UI, validates them, and issues commands
or state updates.
"""
import logging
from typing import Any, Callable, Dict, Optional

from sde.core.actions import ActionType
from sde.core.domain import (
    AlgorithmConfig,
    Challenge,
    ExecutionSettings,
    Experiment,
    ExperimentStatus,
    PatienceBudget,
)
from sde.core.events import EngineCommand
from .proxy import EngineProxy

logger = logging.getLogger(__name__)


class ActionHandler:
    """Processes UI actions, validates them, and dispatches commands to the engine."""

    def __init__(self, engine_proxy: EngineProxy, log_emitter: Callable):
        self.engine_proxy = engine_proxy
        self.log_emitter = log_emitter
        self._action_handlers = self._register_action_handlers()

    def _register_action_handlers(self):
        return {
            ActionType.SET_CHALLENGE: self.handle_set_challenge,
            ActionType.ADD_ALGORITHM: self.handle_add_algorithm,
            ActionType.REMOVE_ALGORITHM: self.handle_remove_algorithm,
            ActionType.UPDATE_PARAM_SPACE: self.handle_update_param_space,
            ActionType.SET_ADAPTIVE_POLICY: self.handle_set_adaptive_policy,
            ActionType.SET_BUDGET: self.handle_set_budget,
            ActionType.MANUAL_PRUNE_TRIAL: self.handle_manual_prune_trial,
            ActionType.MANUAL_PRIORITIZE_TRIAL: self.handle_manual_prioritize_trial,
            ActionType.SPAWN_SIMILAR_TRIAL: self.handle_spawn_similar_trial,
            ActionType.REQUEST_STATE_UPDATE: self.handle_request_state_update,
            ActionType.START_RUN: self.handle_start_run,
            ActionType.PAUSE_RUN: self.handle_pause_run,
            ActionType.RESUME_RUN: self.handle_resume_run,
            ActionType.STOP_RUN: self.handle_stop_run,
            ActionType.SAVE_EXPERIMENT: self.handle_save_experiment,
            ActionType.LOAD_EXPERIMENT: self.handle_load_experiment,
        }

    def handle_action(
        self, action_type: ActionType, payload: Dict[str, Any], experiment: Experiment
    ) -> Optional[Experiment]:
        handler = self._action_handlers.get(action_type)
        if not handler:
            self.log_emitter({"level": "ERROR", "message": f"No handler for action '{action_type.name}'"})
            return None
        return handler(payload, experiment)

    def handle_set_challenge(self, payload: Dict[str, Any], experiment: Experiment) -> Experiment:
        challenge = Challenge.from_dict(payload)
        experiment.challenge = challenge
        self.log_emitter({"level": "INFO", "message": f"Challenge set to '{challenge.name}'"})
        return experiment

    def handle_add_algorithm(self, payload: Dict[str, Any], experiment: Experiment) -> Optional[Experiment]:
        algo_id = f"algo_{len(experiment.algorithms)}"
        new_algo = AlgorithmConfig(id=algo_id, name=payload["name"], parameter_space=payload["parameter_space"])

        if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            num_trials = payload.get("num_trials")
            command_payload = {"algorithm": new_algo.to_dict(), "num_trials": num_trials}
            self.engine_proxy.post_command(EngineCommand.ADD_ALGORITHM_LIVE, command_payload)
            return None
        else:
            experiment.algorithms[algo_id] = new_algo
            self.log_emitter({"level": "INFO", "message": f"Added algorithm: {new_algo.name}"})
            return experiment

    def handle_remove_algorithm(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        algo_id = payload["algorithm_id"]
        if algo_id in experiment.algorithms:
            algo_name = experiment.algorithms[algo_id].name
            command_payload = {"algorithm_id": algo_id, "algorithm_name": algo_name}
            self.engine_proxy.post_command(EngineCommand.REMOVE_ALGORITHM, command_payload)
            self.log_emitter({"level": "INFO", "message": f"Dispatched command to remove algorithm '{algo_name}'."})
        else:
            self.log_emitter({"level": "WARN", "message": f"Could not find algorithm with id {algo_id} to remove."})
        return None

    def handle_update_param_space(self, payload: Dict[str, Any], experiment: Experiment) -> Optional[Experiment]:
        algo_id = payload["algorithm_id"]
        new_space = payload["new_space"]
        if algo_id not in experiment.algorithms:
            self.log_emitter({"level": "WARN", "message": f"Could not find algorithm with id {algo_id} to update."})
            return experiment

        if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            command_payload = {"algorithm_id": algo_id, "new_space": new_space}
            self.engine_proxy.post_command(EngineCommand.UPDATE_PARAM_SPACE_LIVE, command_payload)
            return None
        else:
            algo = experiment.algorithms[algo_id]
            algo.parameter_space = new_space
            self.log_emitter({"level": "INFO", "message": f"Updated parameter space for algorithm {algo.name}."})
            return experiment

    def handle_set_adaptive_policy(self, payload: Dict[str, Any], experiment: Experiment) -> Optional[Experiment]:
        policy_name = payload["policy_name"]
        if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.engine_proxy.post_command(EngineCommand.SET_ADAPTIVE_POLICY, {"policy_name": policy_name})
            return None
        else:
            experiment.adaptive_policy = policy_name
            self.log_emitter({"level": "INFO", "message": f"Adaptive policy set to '{policy_name}'."})
            return experiment

    def handle_set_budget(self, payload: Dict[str, Any], experiment: Experiment) -> Optional[Experiment]:
        if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            self.engine_proxy.post_command(EngineCommand.SET_BUDGET, payload)
            return None
        else:
            current_budget = experiment.patience_budget.to_dict() if experiment.patience_budget else {}
            current_budget.update(payload)
            new_budget = PatienceBudget.from_dict(current_budget)
            experiment.patience_budget = new_budget
            self.log_emitter({"level": "INFO", "message": f"Patience budget updated: {new_budget}"})
            return experiment

    def handle_manual_prune_trial(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        self.engine_proxy.post_command(EngineCommand.PRUNE_TRIAL, {"trial_id": payload["trial_id"]})
        self.log_emitter({"level": "INFO", "message": f"Dispatched command to prune trial {payload['trial_id']}."})
        return None

    def handle_manual_prioritize_trial(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        self.engine_proxy.post_command(EngineCommand.PRIORITIZE_TRIAL, {"trial_id": payload["trial_id"]})
        self.log_emitter({"level": "INFO", "message": f"Dispatched command to prioritize trial {payload['trial_id']}."})
        return None

    def handle_spawn_similar_trial(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        self.engine_proxy.post_command(EngineCommand.SPAWN_TRIAL, payload)
        self.log_emitter({"level": "INFO", "message": f"Dispatched command to spawn trial from {payload['source_trial_id']}."})
        return None

    def handle_request_state_update(self, payload: Dict[str, Any], experiment: Experiment) -> Experiment:
        self.log_emitter({"level": "INFO", "message": "Full state update requested by UI."})
        return experiment

    def handle_start_run(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        execution_settings = ExecutionSettings.from_dict(payload)
        command_payload = {
            "experiment_definition": experiment.to_dict(),
            "execution_settings": execution_settings.to_dict(),
        }
        self.engine_proxy.post_command(EngineCommand.START_RUN, command_payload)
        self.log_emitter({"level": "INFO", "message": "Dispatched START_RUN command to engine."})
        return None

    def handle_pause_run(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        self.engine_proxy.post_command(EngineCommand.PAUSE_RUN)
        self.log_emitter({"level": "INFO", "message": "Dispatched PAUSE_RUN command."})
        return None

    def handle_resume_run(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        self.engine_proxy.post_command(EngineCommand.RESUME_RUN)
        self.log_emitter({"level": "INFO", "message": "Dispatched RESUME_RUN command."})
        return None

    def handle_stop_run(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        self.engine_proxy.post_command(EngineCommand.STOP_RUN)
        self.log_emitter({"level": "INFO", "message": "Dispatched STOP_RUN command."})
        return None

    def handle_save_experiment(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        filepath = payload.get("filepath")
        if not filepath:
            self.log_emitter({"level": "ERROR", "message": "No filepath for save."})
            return None
        self.engine_proxy.post_command(EngineCommand.SAVE_EXPERIMENT, {"filepath": filepath, "experiment": experiment.to_dict()})
        return None

    def handle_load_experiment(self, payload: Dict[str, Any], experiment: Experiment) -> None:
        filepath = payload.get("filepath")
        if not filepath:
            self.log_emitter({"level": "ERROR", "message": "No filepath for load."})
            return None
        self.engine_proxy.post_command(EngineCommand.LOAD_EXPERIMENT, {"filepath": filepath})
        return None
