"""Module for the ExperimentOrchestrator, the central state manager."""

import logging
import traceback
from typing import Any
from typing import Dict

from sde.core.actions import ActionType
from sde.core.domain import AlgorithmConfig
from sde.core.domain import Challenge
from sde.core.domain import ExecutionSettings
from sde.core.domain import Experiment
from sde.core.domain import ExperimentStatus
from sde.core.domain import PatienceBudget
from sde.core.domain import Trial

from ..core.events import EngineCommand
from ..core.events import EngineEvent
from ..core.events import Signal
from .action_validator import ActionValidator
from .insight import Insight
from .proxy import EngineProxy

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Manage the canonical Experiment state and interface with the SDE Runtime."""

    # Signals to update the UI
    state_changed = Signal(dict)
    log_message = Signal(dict)
    operation_started = Signal(str)
    operation_finished = Signal(str)

    def __init__(self):
        """Initialize the ExperimentOrchestrator."""
        self.experiment = Experiment()
        self.engine_proxy = EngineProxy(self.on_engine_event)
        self._action_handlers = self._register_action_handlers()
        self._event_handlers = self._register_event_handlers()
        self.log_message.emit({"level": "INFO", "message": "Orchestrator initialized."})

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

    def _register_event_handlers(self):
        return {
            EngineEvent.TRIAL_UPDATED: self._on_trial_updated,
            EngineEvent.INSIGHTS_GENERATED: self._on_insights_generated,
            EngineEvent.RUN_STARTED: self._on_run_started,
            EngineEvent.RUN_PAUSED: self._on_run_paused,
            EngineEvent.RUN_RESUMED: self._on_run_resumed,
            EngineEvent.RUN_STOPPED: self._on_run_stopped,
            EngineEvent.ALGORITHM_REMOVED: self._on_algorithm_removed,
            EngineEvent.OPERATION_FINISHED: self._on_operation_finished,
            EngineEvent.LOG_MESSAGE: self._on_log_message,
            EngineEvent.EXPERIMENT_LOADED: self._on_experiment_loaded,
        }

    def dispatch(self, action_type: ActionType, payload: Dict[str, Any]) -> None:
        """Receive an action, validate it, and trigger side effects."""
        handler = self._get_action_handler(action_type)
        if not handler:
            return

        if not self._is_action_valid(action_type, payload):
            return

        self._execute_action(handler, action_type, payload)

    def _get_action_handler(self, action_type: ActionType):
        handler = self._action_handlers.get(action_type)
        if not handler:
            self.log_message.emit(
                {
                    "level": "ERROR",
                    "message": f"No handler for action '{action_type.name}'",
                }
            )
        return handler

    def _is_action_valid(self, action_type: ActionType, payload: Dict[str, Any]) -> bool:
        valid_actions = ActionValidator.get_valid_actions(self.experiment)
        if not ActionValidator.is_action_valid(
            action_type.value, payload, valid_actions
        ):
            msg = f"Action '{action_type.value}' is not valid for the current state."
            self.log_message.emit({"level": "WARN", "message": msg})
            return False
        return True

    def _execute_action(self, handler, action_type: ActionType, payload: Dict[str, Any]):
        try:
            # We only emit the state change automatically for handlers that don't
            # expect a confirmation event from the engine.
            emit_now = handler(payload)
            if emit_now:
                self.emit_state_change()
        except Exception as e:
            self.log_message.emit(
                {
                    "level": "ERROR",
                    "message": f"Failed to execute action {action_type.value}: {e}",
                }
            )
            logger.error(traceback.format_exc())

    def on_engine_event(self, event_type: EngineEvent, payload: Dict[str, Any]):
        """Dispatch engine events to the appropriate handler."""
        logger.info(f"Orchestrator received event: {event_type.name}")
        handler = self._event_handlers.get(event_type)
        if handler:
            handler(payload)
        else:
            logger.warning(f"No handler for event type: {event_type}")

        # Always emit a full state change to keep the UI in sync
        self.emit_state_change()

    # --- Event Handlers ---

    def _on_trial_updated(self, payload: Dict[str, Any]):
        trial_data = payload["trial"]
        trial = Trial.from_dict(trial_data)
        self.experiment.trials[trial.id] = trial

    def _on_insights_generated(self, payload: Dict[str, Any]):
        self.experiment.insights.extend(
            [Insight.from_dict(d) for d in payload["insights"]]
        )

    def _on_run_started(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.RUNNING
        if "trials" in payload:
            self.experiment.trials = {
                t["id"]: Trial.from_dict(t) for t in payload["trials"]
            }
        self.log_message.emit(
            {"level": "INFO", "message": "Experiment run has started."}
        )

    def _on_run_paused(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.PAUSED
        self.log_message.emit(
            {"level": "INFO", "message": "Experiment run has been paused."}
        )

    def _on_run_resumed(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.RUNNING
        self.log_message.emit(
            {"level": "INFO", "message": "Experiment run has been resumed."}
        )

    def _on_run_stopped(self, payload: Dict[str, Any]):
        self.experiment.status = ExperimentStatus.STOPPED
        self.log_message.emit(
            {"level": "INFO", "message": "Experiment run has been stopped."}
        )

    def _on_algorithm_removed(self, payload: Dict[str, Any]):
        algo_id = payload["algorithm_id"]
        if algo_id in self.experiment.algorithms:
            del self.experiment.algorithms[algo_id]
        self.experiment.trials = {
            tid: t
            for tid, t in self.experiment.trials.items()
            if t.algorithm_name != payload["algorithm_name"]
        }
        self.log_message.emit(
            {
                "level": "INFO",
                "message": f"Algorithm '{payload['algorithm_name']}' and its trials removed.",
            }
        )

    def _on_operation_finished(self, payload: Dict[str, Any]):
        self.operation_finished.emit(payload.get("message", ""))

    def _on_log_message(self, payload: Dict[str, Any]):
        self.log_message.emit(payload)

    def _on_experiment_loaded(self, payload: Dict[str, Any]):
        loaded_experiment = Experiment.from_dict(payload["experiment"])
        self.experiment = loaded_experiment
        self.log_message.emit(
            {
                "level": "INFO",
                "message": f"Experiment '{loaded_experiment.id}' loaded successfully.",
            }
        )
        self.operation_finished.emit(
            f"Experiment loaded from {payload.get('filepath', 'file')}."
        )

    def emit_state_change(self) -> None:
        """Serialize the current experiment state and emit it."""
        state_dict = self.experiment.to_dict()
        state_dict["valid_actions"] = ActionValidator.get_valid_actions(self.experiment)
        self.state_changed.emit(state_dict)

    # --- Action Handlers (Now simplified to modify state and dispatch commands) ---

    def handle_set_challenge(self, payload: Dict[str, Any]) -> bool:
        """Set the challenge for the experiment."""
        challenge = Challenge.from_dict(payload)
        self.experiment.challenge = challenge
        self.log_message.emit(
            {"level": "INFO", "message": f"Challenge set to '{challenge.name}'"}
        )
        return True

    def handle_add_algorithm(self, payload: Dict[str, Any]) -> bool:
        """Add a new algorithm to the experiment."""
        algo_id = f"algo_{len(self.experiment.algorithms)}"
        new_algo = AlgorithmConfig(
            id=algo_id, name=payload["name"], parameter_space=payload["parameter_space"]
        )
        self.experiment.algorithms[algo_id] = new_algo
        self.log_message.emit(
            {"level": "INFO", "message": f"Added algorithm: {new_algo.name}"}
        )

        if self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            num_trials = payload.get("num_trials")
            command_payload = {
                "algorithm": new_algo.to_dict(),
                "num_trials": num_trials,
            }
            self.engine_proxy.post_command(
                EngineCommand.ADD_ALGORITHM_LIVE, command_payload
            )
            return False  # Wait for engine event to confirm
        return True

    def handle_remove_algorithm(self, payload: Dict[str, Any]) -> bool:
        """Dispatch command to remove an algorithm and prune its trials."""
        algo_id = payload["algorithm_id"]
        if algo_id in self.experiment.algorithms:
            algo_name = self.experiment.algorithms[algo_id].name
            command_payload = {"algorithm_id": algo_id, "algorithm_name": algo_name}
            self.engine_proxy.post_command(
                EngineCommand.REMOVE_ALGORITHM, command_payload
            )
            self.log_message.emit(
                {
                    "level": "INFO",
                    "message": f"Dispatched command to remove algorithm '{algo_name}'.",
                }
            )
        else:
            self.log_message.emit(
                {
                    "level": "WARN",
                    "message": f"Could not find algorithm with id {algo_id} to remove.",
                }
            )
        return False  # Wait for ALGORITHM_REMOVED event

    def handle_update_param_space(self, payload: Dict[str, Any]) -> bool:
        """Update the hyperparameter space for an algorithm."""
        algo_id = payload["algorithm_id"]
        new_space = payload["new_space"]
        if algo_id not in self.experiment.algorithms:
            self.log_message.emit(
                {
                    "level": "WARN",
                    "message": f"Could not find algorithm with id {algo_id} to update.",
                }
            )
            return True
        algo = self.experiment.algorithms[algo_id]
        algo.parameter_space = new_space
        self.log_message.emit(
            {
                "level": "INFO",
                "message": f"Updated parameter space for algorithm {algo.name}.",
            }
        )
        if self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            command_payload = {"algorithm_id": algo_id, "new_space": new_space}
            self.engine_proxy.post_command(
                EngineCommand.UPDATE_PARAM_SPACE_LIVE, command_payload
            )
            return False
        return True

    def handle_set_adaptive_policy(self, payload: Dict[str, Any]) -> bool:
        """Set the adaptive scheduling policy for the experiment."""
        policy_name = payload["policy_name"]
        self.experiment.adaptive_policy = policy_name
        self.log_message.emit(
            {"level": "INFO", "message": f"Adaptive policy set to '{policy_name}'."}
        )
        if self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            self.engine_proxy.post_command(
                EngineCommand.SET_ADAPTIVE_POLICY, {"policy_name": policy_name}
            )
            return False
        return True

    def handle_set_budget(self, payload: Dict[str, Any]) -> bool:
        """Set the patience budget for the experiment."""
        # Budgets can be set incrementally, so we merge with the existing budget.
        current_budget = (
            self.experiment.patience_budget.to_dict()
            if self.experiment.patience_budget
            else {}
        )
        current_budget.update(payload)
        new_budget = PatienceBudget.from_dict(current_budget)
        self.experiment.patience_budget = new_budget

        self.log_message.emit(
            {"level": "INFO", "message": f"Patience budget updated: {new_budget}"}
        )
        if self.experiment.status in [
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
        ]:
            # Send the incremental payload, not the whole budget
            self.engine_proxy.post_command(EngineCommand.SET_BUDGET, payload)
            return False
        return True

    def handle_manual_prune_trial(self, payload: Dict[str, Any]) -> bool:
        """Dispatch a command to manually prune a single trial."""
        trial_id = payload["trial_id"]
        self.engine_proxy.post_command(
            EngineCommand.PRUNE_TRIAL, {"trial_id": trial_id}
        )
        self.log_message.emit(
            {
                "level": "INFO",
                "message": f"Dispatched command to prune trial {trial_id}.",
            }
        )
        return False  # Wait for TRIAL_UPDATED event

    def handle_manual_prioritize_trial(self, payload: Dict[str, Any]) -> bool:
        """Dispatch a command to manually increase a trial's priority."""
        trial_id = payload["trial_id"]
        self.engine_proxy.post_command(
            EngineCommand.PRIORITIZE_TRIAL, {"trial_id": trial_id}
        )
        self.log_message.emit(
            {
                "level": "INFO",
                "message": f"Dispatched command to prioritize trial {trial_id}.",
            }
        )
        return False  # Wait for confirmation

    def handle_spawn_similar_trial(self, payload: Dict[str, Any]) -> bool:
        """Dispatch a command to create a new trial based on an existing one."""
        self.engine_proxy.post_command(EngineCommand.SPAWN_TRIAL, payload)
        msg = f"Dispatched command to spawn trial from {payload['source_trial_id']}."
        self.log_message.emit({"level": "INFO", "message": msg})
        return False  # Wait for new trial to be created

    def handle_request_state_update(self, payload: Dict[str, Any]) -> bool:
        """Handle a manual request from the UI to re-emit the full state."""
        self.log_message.emit(
            {"level": "INFO", "message": "Full state update requested by UI."}
        )
        return True

    def handle_start_run(self, payload: Dict[str, Any]) -> bool:
        """Dispatch the command to start the experiment run.

        State mutation is deferred until the `RUN_STARTED` event is received.
        The action is assumed to be valid by the time it reaches this handler.
        """
        # Decouple execution settings from the experiment definition
        execution_settings = ExecutionSettings.from_dict(payload)
        command_payload = {
            "experiment_definition": self.experiment.to_dict(),
            "execution_settings": execution_settings.to_dict(),
        }
        self.engine_proxy.post_command(EngineCommand.START_RUN, command_payload)
        self.log_message.emit(
            {"level": "INFO", "message": "Dispatched START_RUN command to engine."}
        )

        # Return False to indicate that we are waiting for an event from the
        # engine before emitting a full state_changed signal.
        return False

    def handle_pause_run(self, payload: Dict[str, Any]) -> bool:
        """Dispatch the command to pause the current experiment run."""
        self.engine_proxy.post_command(EngineCommand.PAUSE_RUN)
        self.log_message.emit(
            {"level": "INFO", "message": "Dispatched PAUSE_RUN command."}
        )
        return False  # Wait for confirmation

    def handle_resume_run(self, payload: Dict[str, Any]) -> bool:
        """Dispatch the command to resume a paused experiment run."""
        self.engine_proxy.post_command(EngineCommand.RESUME_RUN)
        self.log_message.emit(
            {"level": "INFO", "message": "Dispatched RESUME_RUN command."}
        )
        return False  # Wait for confirmation

    def handle_stop_run(self, payload: Dict[str, Any]) -> bool:
        """Dispatch the command to stop the current experiment run."""
        self.engine_proxy.post_command(EngineCommand.STOP_RUN)
        self.log_message.emit(
            {"level": "INFO", "message": "Dispatched STOP_RUN command."}
        )
        return False  # Wait for confirmation

    def handle_save_experiment(self, payload: Dict[str, Any]) -> bool:
        """Dispatch the command to save the experiment state."""
        filepath = payload.get("filepath")
        if not filepath:
            self.log_message.emit(
                {"level": "ERROR", "message": "No filepath for save."}
            )
            return True
        self.operation_started.emit(f"Dispatching SAVE command for {filepath}...")
        self.engine_proxy.post_command(
            EngineCommand.SAVE_EXPERIMENT, {"filepath": filepath}
        )
        return False

    def handle_load_experiment(self, payload: Dict[str, Any]) -> bool:
        """Dispatch the command to load an experiment state."""
        filepath = payload.get("filepath")
        if not filepath:
            self.log_message.emit(
                {"level": "ERROR", "message": "No filepath for load."}
            )
            return True
        self.operation_started.emit(f"Dispatching LOAD command for {filepath}...")
        self.engine_proxy.post_command(
            EngineCommand.LOAD_EXPERIMENT, {"filepath": filepath}
        )
        return False

    def shutdown(self) -> None:
        """Gracefully shut down the connection to the runtime engine."""
        self.engine_proxy.shutdown()
        self.log_message.emit({"level": "INFO", "message": "Orchestrator shut down."})
