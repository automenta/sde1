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
    COMPLETED = "COMPLETED"


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
    execution_settings: Optional[Dict[str, Any]] = None  # For num_workers etc.

    def to_dict(self) -> dict:
        """Serializes the entire experiment state to a dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "challenge": self.challenge,
            "algorithms": {
                k: v.to_dict() for k, v in self.algorithms.items()
            },
            "trials": {k: v.to_dict() for k, v in self.trials.items()},
            "insights": self.insights,
            "adaptive_policy": self.adaptive_policy,
            "patience_budget": self.patience_budget,
            "execution_settings": self.execution_settings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        """Creates an Experiment instance from a dictionary."""
        # First, create the simple fields
        exp = cls(
            id=d["id"],
            status=ExperimentStatus(d["status"]),
            challenge=d.get("challenge"),
            insights=d.get("insights", []),
            adaptive_policy=d.get("adaptive_policy", "SuccessiveHalving"),
            patience_budget=d.get("patience_budget"),
            execution_settings=d.get("execution_settings"),
        )

        # Then, deserialize the nested objects
        exp.algorithms = {
            k: AlgorithmConfig.from_dict(v) for k, v in d.get("algorithms", {}).items()
        }
        exp.trials = {
            k: Trial.from_dict(v) for k, v in d.get("trials", {}).items()
        }
        return exp
