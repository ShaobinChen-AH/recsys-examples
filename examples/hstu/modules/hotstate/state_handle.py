from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Set


class StateType(Enum):
    """Logical class of a managed state object."""
    EMBEDDING_HOT_ROWS = auto()   # frequently-accessed embedding rows
    EMBEDDING_COLD_ROWS = auto()  # rarely-accessed embedding rows
    SESSION_KV_USER = auto()      # KV pages allocated to a user


class Placement(Enum):
    """Where a state object's data currently resides."""
    HBM = auto()
    HOST_DRAM = auto()


class Reconstructability(Enum):
    """What happens when this object is evicted."""
    REFETCHABLE = auto()     # can reload from host memory
    RECOMPUTABLE = auto()    # derived data, can recompute from scratch


@dataclass
class StateHandle:
    """Control-plane description of a managed state object."""
    state_type: StateType
    logical_key: str
    footprint_bytes: int
    placement: Set[Placement] = field(default_factory=set)
    reconstructability: Reconstructability = Reconstructability.REFETCHABLE
    consistency_class: str = "default"
    # Populated by ValueEngine each epoch:
    reuse_imminence: float = 0.0
    stall_sensitivity_ms: float = 0.0
    movement_cost_ms: float = 0.0
    # Access tracking:
    last_access_epoch: int = -1
    access_count: int = 0


@dataclass
class ScoredHandle:
    """A StateHandle with its computed arbitration score."""
    handle: StateHandle
    score: float
    benefit_density: float
    occupancy_penalty: float
    semantic_risk: float
