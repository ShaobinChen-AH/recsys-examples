from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Set, Tuple


class ConsistencyError(RuntimeError):
    """Raised when a state transition would expose an invalid replica."""


class StateType(Enum):
    """Logical class of a managed state object."""
    EMBEDDING_HOT_ROWS = auto()   # frequently-accessed embedding rows
    EMBEDDING_COLD_ROWS = auto()  # rarely-accessed embedding rows
    SESSION_KV_USER = auto()      # KV pages allocated to a user


class Placement(Enum):
    """Where a state object's data currently resides."""
    HBM = auto()
    HOST_DRAM = auto()
    NVME = auto()
    REMOTE_GPU = auto()


class TransferState(Enum):
    """Lifecycle of a state movement request."""
    STABLE = auto()
    PLANNED = auto()
    IN_FLIGHT = auto()
    COMPLETE = auto()
    FAILED = auto()
    CANCELLED = auto()


class Reconstructability(Enum):
    """What happens when this object is evicted."""
    REFETCHABLE = auto()     # can reload from host memory
    RECOMPUTABLE = auto()    # derived data, can recompute from scratch


@dataclass
class StateHandle:
    """Versioned control-plane description of a managed state object.

    ``placement`` is the set of known replica locations.  The authoritative
    copy is tracked separately because a mutable embedding row can have a
    stale read replica, while a derived KV object may have no authoritative
    host copy at all.
    """
    state_type: StateType
    logical_key: str
    footprint_bytes: int
    placement: Set[Placement] = field(default_factory=set)
    reconstructability: Reconstructability = Reconstructability.REFETCHABLE
    consistency_class: str = "default"
    version: int = 0
    freshness_epoch: int = -1
    owner_device: Optional[str] = None
    authoritative_placement: Optional[Placement] = None
    replica_versions: Dict[Placement, int] = field(default_factory=dict)
    replica_freshness_epochs: Dict[Placement, int] = field(default_factory=dict)
    transfer_cost_ms: float = 0.0
    reconstruction_cost_ms: float = 0.0
    expected_reuse_window: int = 0
    staleness_tolerance_epochs: int = 0
    ttl_epochs: Optional[int] = None
    writeback_required: bool = False
    writeback_pending: bool = False
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    transfer_state: TransferState = TransferState.STABLE
    transfer_id: Optional[str] = None
    last_error: Optional[str] = None
    recovery_source: Optional[Placement] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Populated by ValueEngine each epoch:
    reuse_imminence: float = 0.0
    stall_sensitivity_ms: float = 0.0
    movement_cost_ms: float = 0.0
    # Access tracking:
    last_access_epoch: int = -1
    access_count: int = 0

    def __post_init__(self) -> None:
        self.placement = set(self.placement)
        self.dependencies = tuple(self.dependencies)
        for location in self.placement:
            self.replica_versions.setdefault(location, self.version)
            self.replica_freshness_epochs.setdefault(location, self.freshness_epoch)

    @property
    def placements(self) -> Set[Placement]:
        """Proposal terminology alias for the legacy ``placement`` field."""
        return self.placement

    @placements.setter
    def placements(self, value: Set[Placement]) -> None:
        self.placement = set(value)

    @property
    def access_epoch(self) -> int:
        return self.last_access_epoch

    @access_epoch.setter
    def access_epoch(self, value: int) -> None:
        self.last_access_epoch = int(value)

    def mark_access(self, epoch: int) -> None:
        self.last_access_epoch = int(epoch)
        self.access_count += 1

    def add_replica(
        self,
        location: Placement,
        *,
        version: Optional[int] = None,
        freshness_epoch: Optional[int] = None,
    ) -> None:
        self.placement.add(location)
        self.replica_versions[location] = self.version if version is None else int(version)
        self.replica_freshness_epochs[location] = (
            self.freshness_epoch
            if freshness_epoch is None
            else int(freshness_epoch)
        )

    def remove_replica(self, location: Placement) -> None:
        self.placement.discard(location)
        self.replica_versions.pop(location, None)
        self.replica_freshness_epochs.pop(location, None)
        if self.authoritative_placement == location:
            self.authoritative_placement = next(iter(self.placement), None)

    def set_authoritative(
        self,
        location: Placement,
        *,
        version: Optional[int] = None,
        freshness_epoch: Optional[int] = None,
    ) -> None:
        if version is not None and int(version) < int(self.version):
            raise ConsistencyError(
                f"cannot make version {version} authoritative for {self.logical_key}; "
                f"current version is {self.version}"
            )
        self.add_replica(location, version=version, freshness_epoch=freshness_epoch)
        self.authoritative_placement = location
        if version is not None:
            self.version = int(version)
        if freshness_epoch is not None:
            self.freshness_epoch = int(freshness_epoch)

    def is_expired(self, epoch: int) -> bool:
        return self.ttl_epochs is not None and self.last_access_epoch >= 0 and (
            int(epoch) - self.last_access_epoch > int(self.ttl_epochs)
        )

    def is_stale(self, location: Placement) -> bool:
        replica_version = self.replica_versions.get(location, -1)
        replica_epoch = self.replica_freshness_epochs.get(location, -1)
        version_stale = self.version >= 0 and replica_version >= 0 and (
            replica_version < self.version
        )
        epoch_stale = self.freshness_epoch >= 0 and (
            self.freshness_epoch - replica_epoch > self.staleness_tolerance_epochs
        )
        return bool(version_stale or epoch_stale)

    def validate_read(
        self,
        location: Placement,
        *,
        required_version: Optional[int] = None,
        allow_stale: bool = False,
    ) -> None:
        """Enforce the handle's read contract before a consumer uses a replica.

        Inference embeddings are immutable replicas and append-only KV handles
        use ``version`` as the logical token length.  A missing or stale
        replica is never silently accepted by the control plane.
        """
        if location not in self.placement:
            raise ConsistencyError(
                f"{self.logical_key} has no {location.name} replica"
            )
        if not allow_stale and self.is_stale(location):
            raise ConsistencyError(
                f"{self.logical_key} {location.name} replica is stale: "
                f"replica_version={self.replica_versions.get(location, -1)} "
                f"required_version={self.version}"
            )
        if required_version is not None:
            replica_version = self.replica_versions.get(location, -1)
            if replica_version < int(required_version):
                raise ConsistencyError(
                    f"{self.logical_key} {location.name} replica only covers "
                    f"version {replica_version}, requested {required_version}"
                )

    def validate_transition(
        self,
        source: Optional[Placement],
        target: Optional[Placement],
        *,
        version: Optional[int] = None,
        authoritative: bool = False,
    ) -> None:
        """Reject transitions that lose the only valid copy or go backwards."""
        if source is not None and source not in self.placement:
            raise ConsistencyError(
                f"cannot move {self.logical_key}: source {source.name} is absent"
            )
        if version is not None and int(version) < int(self.version):
            raise ConsistencyError(
                f"cannot move {self.logical_key} from version {self.version} "
                f"back to {version}"
            )
        if authoritative and target is None:
            raise ConsistencyError(
                f"authoritative transition for {self.logical_key} has no target"
            )

@dataclass
class ScoredHandle:
    """A StateHandle with its computed arbitration score."""
    handle: StateHandle
    score: float
    benefit_density: float = 0.0
    occupancy_penalty: float = 0.0
    semantic_risk: float = 0.0
    reuse_probability: float = 0.0
    miss_cost_ms: float = 0.0
    gross_benefit_ms: float = 0.0
    movement_cost_ms: float = 0.0
    risk_cost_ms: float = 0.0
    net_benefit_ms: float = 0.0
    value_density_ms_per_byte: float = 0.0
    decision_reason: str = ""
    opportunity_cost_ms: float = 0.0
