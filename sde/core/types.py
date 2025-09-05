import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

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
    PRUNED = "PRUNED"
    COMPLETED = "COMPLETED"

@dataclass
class Trial:
    """A container for state and time-series results."""
    id: str
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    priority: int = 0 # Higher value means higher priority

    # State for pausing/resuming
    current_epoch: int = 0
    checkpoint_path: Optional[str] = None

    # Performance metrics
    est_time_per_epoch: Optional[float] = None

    # Time-series results
    results: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict) # e.g., {'accuracy': [(1, 0.8), (2, 0.9)]}

    def to_dict(self) -> dict:
        """Converts the Trial to a dictionary, handling enum serialization."""
        d = dataclasses.asdict(self)
        d['status'] = self.status.value
        return d

# --- V2 Unified Specification Types ---
import uuid

class ExperimentStatus(Enum):
    DEFINING = "DEFINING" # The initial state, no compute is running.
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"

@dataclass
class AlgorithmConfig:
    id: str
    name: str
    parameter_space: Dict[str, Any] # e.g., {'lr': (0.001, 0.1), ...}
    is_active: bool = True

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
    suggested_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy & Constraints
    adaptive_policy: str = "SuccessiveHalving" # Default policy
    patience_budget: Optional[Dict[str, int]] = None
