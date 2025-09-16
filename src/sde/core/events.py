import logging
import traceback
from enum import Enum
from typing import Callable
from typing import List

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


class EngineCommand(Enum):
    """Commands sent from the Orchestrator to the RuntimeEngine."""

    ADD_ALGORITHM_LIVE = "ADD_ALGORITHM_LIVE"
    REMOVE_ALGORITHM = "REMOVE_ALGORITHM"
    UPDATE_PARAM_SPACE_LIVE = "UPDATE_PARAM_SPACE_LIVE"
    SET_ADAPTIVE_POLICY = "SET_ADAPTIVE_POLICY"
    SET_BUDGET = "SET_BUDGET"
    PRUNE_TRIAL = "PRUNE_TRIAL"
    PRIORITIZE_TRIAL = "PRIORITIZE_TRIAL"
    SPAWN_TRIAL = "SPAWN_TRIAL"
    START_RUN = "START_RUN"
    PAUSE_RUN = "PAUSE_RUN"
    RESUME_RUN = "RESUME_RUN"
    STOP_RUN = "STOP_RUN"
    SAVE_EXPERIMENT = "SAVE_EXPERIMENT"
    LOAD_EXPERIMENT = "LOAD_EXPERIMENT"


class EngineEvent(Enum):
    """Events emitted by the RuntimeEngine to the Orchestrator."""

    TRIAL_UPDATED = "TRIAL_UPDATED"
    INSIGHTS_GENERATED = "INSIGHTS_GENERATED"
    RUN_STARTED = "RUN_STARTED"
    RUN_PAUSED = "RUN_PAUSED"
    RUN_RESUMED = "RUN_RESUMED"
    RUN_STOPPED = "RUN_STOPPED"
    ALGORITHM_REMOVED = "ALGORITHM_REMOVED"
    OPERATION_FINISHED = "OPERATION_FINISHED"
    LOG_MESSAGE = "LOG_MESSAGE"
    EXPERIMENT_LOADED = "EXPERIMENT_LOADED"
