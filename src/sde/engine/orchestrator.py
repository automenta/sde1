"""Module for the ExperimentOrchestrator, the central state manager."""

import logging
import traceback
from typing import Any, Dict

from sde.core.actions import ActionType
from sde.core.domain import Experiment
from ..core.events import EngineEvent, Signal
from .action_validator import ActionValidator
from .proxy import EngineProxy
from .event_handler import EventHandler
from .action_handler import ActionHandler

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

        self.event_handler = EventHandler(self.experiment, self.log_message.emit, self.operation_finished.emit)
        self.action_handler = ActionHandler(self.engine_proxy, self.log_message.emit)

        self.log_message.emit({"level": "INFO", "message": "Orchestrator initialized."})

    def dispatch(self, action_type: ActionType, payload: Dict[str, Any]) -> None:
        """Receive an action, validate it, and trigger side effects."""
        if not self._is_action_valid(action_type, payload):
            return

        try:
            # Action handler can return a new experiment object for sync changes
            new_state = self.action_handler.handle_action(action_type, payload, self.experiment)
            if new_state:
                self.experiment = new_state
                self.emit_state_change()
        except Exception as e:
            self.log_message.emit(
                {
                    "level": "ERROR",
                    "message": f"Failed to execute action {action_type.value}: {e}",
                }
            )
            logger.error(traceback.format_exc())

    def _is_action_valid(self, action_type: ActionType, payload: Dict[str, Any]) -> bool:
        valid_actions = ActionValidator.get_valid_actions(self.experiment)
        if not ActionValidator.is_action_valid(
            action_type.value, payload, valid_actions
        ):
            msg = f"Action '{action_type.value}' is not valid for the current state."
            self.log_message.emit({"level": "WARN", "message": msg})
            return False
        return True

    def on_engine_event(self, event_type: EngineEvent, payload: Dict[str, Any]):
        """Dispatch engine events to the event handler."""
        self.event_handler.handle_event(event_type, payload)

        # Special case for loading an experiment, as it replaces the object
        if event_type == EngineEvent.EXPERIMENT_LOADED:
            self.experiment = self.event_handler.get_experiment()

        # Always emit a full state change to keep the UI in sync
        self.emit_state_change()

    def emit_state_change(self) -> None:
        """Serialize the current experiment state and emit it."""
        state_dict = self.experiment.to_dict()
        state_dict["valid_actions"] = ActionValidator.get_valid_actions(self.experiment)
        self.state_changed.emit(state_dict)

    def shutdown(self) -> None:
        """Gracefully shut down the connection to the runtime engine."""
        self.engine_proxy.shutdown()
        self.log_message.emit({"level": "INFO", "message": "Orchestrator shut down."})
