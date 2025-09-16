import uuid
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from dataclasses_json import DataClassJsonMixin
from dataclasses_json import Undefined
from dataclasses_json import config
from dataclasses_json import dataclass_json

from ..config import Defaults
from .definitions import DatasetType
from .definitions import Insight


# --- Base classes for type hinting ---
@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass(frozen=True)
class JsonSerializable(DataClassJsonMixin):
    """Base class for all serializable domain objects."""

    pass


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class MutableJsonSerializable(DataClassJsonMixin):
    """Base class for all mutable serializable domain objects."""

    pass


# --- Core Runtime Engine & Experiment State ---


class WorkUnitType(Enum):
    """The type of task to be performed by a computational worker."""

    PROFILE_SPEED = "PROFILE_SPEED"
    TRAIN_EPOCH = "TRAIN_EPOCH"
    EVALUATE = "EVALUATE"


@dataclass(frozen=True)
class WorkUnit(JsonSerializable):
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
class Trial(MutableJsonSerializable):
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


class ExperimentStatus(Enum):
    """The overall status of an experiment run."""

    DEFINING = "DEFINING"  # Initial state, user is configuring the experiment.
    RUNNING = "RUNNING"  # The experiment is actively being executed by the engine.
    PAUSED = "PAUSED"  # The entire experiment run is paused.
    STOPPED = "STOPPED"  # The experiment was manually stopped by the user.
    COMPLETED = "COMPLETED"  # The experiment finished naturally.
    FAILED = "FAILED"  # The experiment terminated due to a critical error.


@dataclass
class AlgorithmConfig(MutableJsonSerializable):
    """Configuration for an algorithm to be included in the experiment."""

    id: str
    name: str
    parameter_space: Dict[str, Any]  # Defines the search space for HPO
    is_active: bool = True


@dataclass
class ExecutionSettings(MutableJsonSerializable):
    """Settings that control the execution environment of the experiment."""

    num_trials_per_algo: int
    num_workers: int
    enable_checkpointing: bool
    work_unit_timeout_seconds: int


@dataclass(frozen=True)
class Challenge(JsonSerializable):
    """The definition of the problem domain for the experiment."""

    name: str
    type: DatasetType


@dataclass(frozen=True)
class PatienceBudget(JsonSerializable):
    """A composite measure of the total resources a user is willing to invest."""

    wall_clock_time_seconds: Optional[int] = None
    cpu_seconds: Optional[int] = None
    gpu_seconds: Optional[int] = None
    worker_throttle_percent: int = 100


@dataclass
class Experiment(MutableJsonSerializable):
    """The single, canonical data structure holding the entire application state.
    This object is managed exclusively by the ExperimentOrchestrator.
    """

    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    status: ExperimentStatus = ExperimentStatus.DEFINING

    # Core Definition: What the experiment IS
    challenge: Optional[Challenge] = None
    algorithms: Dict[str, AlgorithmConfig] = field(default_factory=dict)

    # Runtime State: What is HAPPENING in the experiment
    trials: Dict[str, Trial] = field(default_factory=dict)
    insights: List[Insight] = field(default_factory=list)

    # Strategy & Constraints: HOW the experiment is run
    adaptive_policy: str = Defaults.DEFAULT_ADAPTIVE_POLICY
    patience_budget: Optional[PatienceBudget] = None
    execution_settings: Optional[ExecutionSettings] = None
    scheduler_state: Dict[str, Any] = field(default_factory=dict)
