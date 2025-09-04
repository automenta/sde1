import logging
import traceback
from typing import List, Callable, Dict, Any

from sde.core.types import Experiment, ExperimentStatus
from sde.engine.runtime import SdeRuntimeEngine

logger = logging.getLogger(__name__)

class Signal:
    """A simple signal implementation to remove Qt dependency from the core engine."""
    def __init__(self, *arg_types):
        self._callbacks: List[Callable] = []

    def connect(self, callback: Callable):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.error(f"Error in signal callback: {traceback.format_exc()}")

class ExperimentOrchestrator:
    """
    V2 Orchestrator: Manages the canonical Experiment state object and
    orchestrates the SDE Runtime Engine based on user actions.
    """
    # Signals to update the UI
    state_changed = Signal(dict)
    log_message = Signal(str)

    def __init__(self):
        self.experiment = Experiment()
        self.runtime_engine = None # Will be initialized on START_RUN
        self.log_message.emit("INFO: Orchestrator initialized in DEFINING state.")

    def dispatch(self, action_type: str, payload: Dict[str, Any]):
        """
        Receives an action, validates it, mutates the state,
        and triggers side effects.
        """
        handler = getattr(self, f"handle_{action_type.lower()}", None)
        if not handler:
            self.log_message.emit(f"ERROR: No handler for action '{action_type}'")
            return

        # Use the validator to check if the action is allowed
        if action_type not in self.get_valid_actions():
            self.log_message.emit(f"WARN: Action '{action_type}' is not valid in state '{self.experiment.status.value}'")
            return

        try:
            handler(payload)
            self.emit_state_change()
        except Exception as e:
            self.log_message.emit(f"ERROR: Failed to execute action {action_type}: {e}")
            logger.error(traceback.format_exc())

    def emit_state_change(self):
        """Serializes the experiment state and emits it."""
        # A real implementation would have a more robust serializer
        state_dict = {
            "id": self.experiment.id,
            "status": self.experiment.status.value,
            "challenge": self.experiment.challenge,
            "algorithms": {k: v.__dict__ for k, v in self.experiment.algorithms.items()},
            "trials": {k: v.to_dict() for k, v in self.experiment.trials.items()},
            "adaptive_policy": self.experiment.adaptive_policy,
        }
        self.state_changed.emit(state_dict)

    # --- Action Handlers ---

    def handle_set_challenge(self, payload: Dict[str, Any]):
        self.experiment.challenge = payload
        self.log_message.emit(f"INFO: Challenge set to '{payload.get('name', 'Unknown')}'")

    def handle_add_algorithm(self, payload: Dict[str, Any]):
        # In a real app, this would do more validation
        from sde.core.types import AlgorithmConfig
        algo_id = f"algo_{len(self.experiment.algorithms)}"
        new_algo = AlgorithmConfig(
            id=algo_id,
            name=payload['name'],
            parameter_space=payload['parameter_space']
        )
        self.experiment.algorithms[algo_id] = new_algo
        self.log_message.emit(f"INFO: Added algorithm: {new_algo.name}")

    def handle_start_run(self, payload: Dict[str, Any]):
        self.log_message.emit("INFO: START_RUN action received. Initializing runtime.")
        self.experiment.status = ExperimentStatus.RUNNING

        # This is where the old logic would be triggered, but in a more
        # controlled way. For this refactoring, we'll just log it.
        self.log_message.emit("SIM: Would start SdeRuntimeEngine now.")
        # In a full implementation, you would initialize and start the engine here.
        # self.runtime_engine = SdeRuntimeEngine(...)
        # self.runtime_engine.start()

    def get_valid_actions(self) -> List[str]:
        """
        This is the Action Validator. It inspects the current state and returns
        a list of action types that are currently valid.
        """
        status = self.experiment.status
        actions = []

        if status == ExperimentStatus.DEFINING:
            if self.experiment.challenge is None:
                actions.append("SET_CHALLENGE")
            else:
                actions.append("ADD_ALGORITHM")
                if self.experiment.algorithms:
                    actions.append("START_RUN")

        elif status == ExperimentStatus.RUNNING:
            actions.append("PAUSE_RUN")
            actions.append("ADD_ALGORITHM") # Example of a mid-run interaction

        elif status == ExperimentStatus.PAUSED:
            actions.append("RESUME_RUN")

        return actions
