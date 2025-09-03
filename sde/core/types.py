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

    # State for pausing/resuming
    current_epoch: int = 0
    checkpoint_path: Optional[str] = None

    # Performance metrics
    est_time_per_epoch: Optional[float] = None

    # Time-series results
    results: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict) # e.g., {'accuracy': [(1, 0.8), (2, 0.9)]}
