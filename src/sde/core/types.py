import dataclasses
import uuid
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class WorkUnitType(Enum):
    PROFILE_SPEED = "PROFILE_SPEED"
    TRAIN_EPOCH = "TRAIN_EPOCH"
    EVALUATE = "EVALUATE"


@dataclass(frozen=True)
class WorkUnit:
    """The smallest schedulable quantum of work."""

    trial_id: str
    type: WorkUnitType
    payload: Dict[str, Any] = field(default_factory=dict)


class TrialStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    PRUNED = "PRUNED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Trial:
    """A container for state and time-series results."""

    id: str
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    priority: int = 0  # Higher value means higher priority

    # State for pausing/resuming
    current_epoch: int = 0
    checkpoint_path: Optional[str] = None

    # Performance metrics
    est_time_per_epoch: Optional[float] = None

    # Time-series results
    results: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converts the Trial to a dictionary, handling enum serialization."""
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trial":
        """Creates a Trial instance from a dictionary.

        This method is robust to extra keys in the input dictionary,
        which allows for forward compatibility if the Trial class is
        extended with new fields.
        """
        # Ensure status is converted from string to Enum
        if "status" in d and isinstance(d["status"], str):
            d["status"] = TrialStatus(d["status"])

        # Get the names of the fields defined in the Trial dataclass
        known_fields = {f.name for f in dataclasses.fields(cls)}

        # Filter the input dictionary to only include known fields
        filtered_dict = {k: v for k, v in d.items() if k in known_fields}

        return cls(**filtered_dict)


# --- V2 Unified Specification Types ---


class ExperimentStatus(Enum):
    DEFINING = "DEFINING"  # The initial state, no compute is running.
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"  # The experiment terminated due to an unrecoverable error.


@dataclass
class AlgorithmConfig:
    id: str
    name: str
    parameter_space: Dict[str, Any]  # e.g., {'lr': (0.001, 0.1), ...}
    is_active: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmConfig":
        """Creates an AlgorithmConfig instance from a dictionary."""
        return cls(**d)


@dataclass
class ExecutionSettings:
    """Settings that control the execution of the experiment run."""

    num_trials_per_algo: int
    num_workers: int
    enable_checkpointing: bool
    work_unit_timeout_seconds: int

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionSettings":
        """Creates an ExecutionSettings instance from a dictionary."""
        # This is robust to extra keys in the dictionary
        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered_dict)


@dataclass
class Experiment:
    id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    status: ExperimentStatus = ExperimentStatus.DEFINING

    # Core Definition
    challenge: Optional[Dict[str, Any]] = None
    algorithms: Dict[str, AlgorithmConfig] = field(default_factory=dict)

    # Runtime State
    trials: Dict[str, Trial] = field(default_factory=dict)
    insights: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy & Constraints
    adaptive_policy: str = "SuccessiveHalving"  # Default policy
    patience_budget: Optional[Dict[str, int]] = None
    execution_settings: Optional[ExecutionSettings] = None  # For num_workers etc.
    scheduler_state: Dict[str, Any] = field(
        default_factory=dict
    )  # For schedulers that need to persist state

    def to_dict(self) -> dict:
        """Serializes the entire experiment state to a dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "challenge": self.challenge,
            "algorithms": {k: v.to_dict() for k, v in self.algorithms.items()},
            "trials": {k: v.to_dict() for k, v in self.trials.items()},
            "insights": self.insights,
            "adaptive_policy": self.adaptive_policy,
            "patience_budget": self.patience_budget,
            "execution_settings": dataclasses.asdict(self.execution_settings)
            if self.execution_settings
            else None,
            "scheduler_state": self.scheduler_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        """Creates an Experiment instance from a dictionary.

        This method is robust to extra keys in the input dictionary,
        which allows for forward compatibility if the Experiment class is
        extended with new fields.
        """
        # Get the names of the fields defined in the Experiment dataclass
        known_fields = {f.name for f in dataclasses.fields(cls)}
        kwargs = {}

        for name, field_type in cls.__annotations__.items():
            if name not in d or name not in known_fields:
                continue

            data = d[name]
            if data is None:
                kwargs[name] = None
                continue

            # Handle complex nested types
            origin = getattr(field_type, "__origin__", None)
            args = getattr(field_type, "__args__", ())

            if origin is dict and args and hasattr(args[1], "from_dict"):
                # e.g., Dict[str, Trial]
                item_class = args[1]
                kwargs[name] = {k: item_class.from_dict(v) for k, v in data.items()}
            elif hasattr(field_type, "from_dict"):
                # e.g., ExecutionSettings
                kwargs[name] = field_type.from_dict(data)
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                # e.g., ExperimentStatus
                kwargs[name] = field_type(data)
            elif origin is Optional or (
                origin is Union and len(args) == 2 and args[1] is type(None)
            ):
                # Handle Optional[T] which is Union[T, None]
                inner_type = args[0]
                if hasattr(inner_type, "from_dict"):
                    kwargs[name] = inner_type.from_dict(data)
                else:
                    kwargs[name] = data
            else:
                # For simple types like str, list, dict
                kwargs[name] = data

        return cls(**kwargs)
