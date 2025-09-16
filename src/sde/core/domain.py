import dataclasses
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .definitions import Insight

# --- Core Runtime Engine & Experiment State ---


class WorkUnitType(Enum):
    """The type of task to be performed by a computational worker."""

    PROFILE_SPEED = "PROFILE_SPEED"
    TRAIN_EPOCH = "TRAIN_EPOCH"
    EVALUATE = "EVALUATE"


@dataclass(frozen=True)
class WorkUnit:
    """The smallest schedulable quantum of work. This is a stateless, immutable task."""

    trial_id: str
    type: WorkUnitType
    payload: Dict[str, Any] = field(default_factory=dict)


class TrialStatus(Enum):
    """The lifecycle status of a single trial."""

    PENDING = "PENDING"  # The trial has been defined but not yet started.
    ACTIVE = "ACTIVE"  # The trial is currently being actively trained and evaluated.
    PAUSED = "PAUSED"  # The trial is temporarily stopped but can be resumed.
    PRUNED = "PRUNED"  # The trial was terminated early by an adaptive scheduler.
    COMPLETED = "COMPLETED"  # The trial has finished all its work.
    FAILED = "FAILED"  # The trial terminated due to an unrecoverable error.


@dataclass
class Trial:
    """A container for the dynamic state and results of a single algorithm instance."""

    id: str
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    priority: int = 0  # Higher value means higher priority in the work queue

    # State for pausing/resuming and scheduling
    current_epoch: int = 0
    checkpoint_path: Optional[str] = None
    est_time_per_epoch: Optional[float] = None  # In seconds

    # Time-series results, e.g., {'accuracy': [(1, 0.8), (2, 0.9)]}
    results: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)

    # Generic key-value store for scheduler-specific metadata
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializes the Trial to a dictionary, converting enums to strings."""
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trial":
        """Creates a Trial from a dictionary, robustly handling missing/extra keys."""
        if "status" in d and isinstance(d["status"], str):
            d["status"] = TrialStatus(d["status"])
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered_dict)


class ExperimentStatus(Enum):
    """The overall status of an experiment run."""

    DEFINING = "DEFINING"  # Initial state, user is configuring the experiment.
    RUNNING = "RUNNING"  # The experiment is actively being executed by the engine.
    PAUSED = "PAUSED"  # The entire experiment run is paused.
    STOPPED = "STOPPED"  # The experiment was manually stopped by the user.
    COMPLETED = "COMPLETED"  # The experiment finished naturally.
    FAILED = "FAILED"  # The experiment terminated due to a critical error.


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm to be included in the experiment."""

    id: str
    name: str
    parameter_space: Dict[str, Any]  # Defines the search space for HPO
    is_active: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmConfig":
        return cls(**d)


@dataclass
class ExecutionSettings:
    """Settings that control the execution environment of the experiment."""

    num_trials_per_algo: int
    num_workers: int
    enable_checkpointing: bool
    work_unit_timeout_seconds: int

    def to_dict(self) -> dict:
        """Serializes the settings to a dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionSettings":
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered_dict)


@dataclass
class Experiment:
    """The single, canonical data structure holding the entire application state.
    This object is managed exclusively by the ExperimentOrchestrator.
    """

    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    status: ExperimentStatus = ExperimentStatus.DEFINING

    # Core Definition: What the experiment IS
    challenge: Optional[Dict[str, Any]] = None
    algorithms: Dict[str, AlgorithmConfig] = field(default_factory=dict)

    # Runtime State: What is HAPPENING in the experiment
    trials: Dict[str, Trial] = field(default_factory=dict)
    insights: List[Insight] = field(default_factory=list)

    # Strategy & Constraints: HOW the experiment is run
    adaptive_policy: str = "SuccessiveHalving"
    patience_budget: Optional[Dict[str, int]] = None
    execution_settings: Optional[ExecutionSettings] = None
    scheduler_state: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializes the entire experiment state to a dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "challenge": self.challenge,
            "algorithms": {k: v.to_dict() for k, v in self.algorithms.items()},
            "trials": {k: v.to_dict() for k, v in self.trials.items()},
            "insights": [insight.to_dict() for insight in self.insights],
            "adaptive_policy": self.adaptive_policy,
            "patience_budget": self.patience_budget,
            "execution_settings": self.execution_settings.to_dict()
            if self.execution_settings
            else None,
            "scheduler_state": self.scheduler_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        """Creates an Experiment instance from a dictionary."""
        d_copy = d.copy()
        if "status" in d_copy and isinstance(d_copy["status"], str):
            d_copy["status"] = ExperimentStatus(d_copy["status"])
        if "algorithms" in d_copy and d_copy["algorithms"] is not None:
            d_copy["algorithms"] = {
                k: AlgorithmConfig.from_dict(v)
                for k, v in d_copy["algorithms"].items()
            }
        if "trials" in d_copy and d_copy["trials"] is not None:
            d_copy["trials"] = {
                k: Trial.from_dict(v) for k, v in d_copy["trials"].items()
            }
        if "insights" in d_copy and d_copy["insights"] is not None:
            d_copy["insights"] = [
                Insight.from_dict(v) for v in d_copy["insights"]
            ]
        if "execution_settings" in d_copy and d_copy["execution_settings"] is not None:
            d_copy["execution_settings"] = ExecutionSettings.from_dict(
                d_copy["execution_settings"]
            )

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in d_copy.items() if k in known_fields}
        return cls(**filtered_dict)
