from enum import Enum


class ActionType(str, Enum):
    """Defines the types of actions that can be dispatched to the ExperimentOrchestrator.
    Using a StrEnum makes it serializable and backward-compatible with the old string-based system.
    """

    # Experiment Setup
    SET_CHALLENGE = "SET_CHALLENGE"
    ADD_ALGORITHM = "ADD_ALGORITHM"
    REMOVE_ALGORITHM = "REMOVE_ALGORITHM"
    UPDATE_PARAM_SPACE = "UPDATE_PARAM_SPACE"
    SET_ADAPTIVE_POLICY = "SET_ADAPTIVE_POLICY"
    SET_BUDGET = "SET_BUDGET"

    # Execution Control
    START_RUN = "START_RUN"
    PAUSE_RUN = "PAUSE_RUN"
    RESUME_RUN = "RESUME_RUN"

    # Persistence
    SAVE_EXPERIMENT = "SAVE_EXPERIMENT"
    LOAD_EXPERIMENT = "LOAD_EXPERIMENT"

    # Live Intervention
    MANUAL_PRUNE_TRIAL = "MANUAL_PRUNE_TRIAL"
    MANUAL_PRIORITIZE_TRIAL = "MANUAL_PRIORITIZE_TRIAL"
    SPAWN_SIMILAR_TRIAL = "SPAWN_SIMILAR_TRIAL"

    def __str__(self):
        return self.value
