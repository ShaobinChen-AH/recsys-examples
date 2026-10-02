from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from modules.hotstate.state_handle import (
    ConsistencyError,
    Placement,
    StateHandle,
    TransferState,
)


@dataclass
class PlacementEntry:
    """Versioned directory record for one logical state object."""

    key: str
    location: Placement
    authoritative: bool
    pinned: bool = False
    in_flight: bool = False
    placements: Set[Placement] = field(default_factory=set)
    authoritative_placement: Optional[Placement] = None
    version: int = 0
    freshness_epoch: int = -1
    owner_device: Optional[str] = None
    replica_versions: Dict[Placement, int] = field(default_factory=dict)
    replica_freshness_epochs: Dict[Placement, int] = field(default_factory=dict)
    transfer_state: TransferState = TransferState.STABLE
    transfer_id: Optional[str] = None
    writeback_required: bool = False
    writeback_pending: bool = False
    last_error: Optional[str] = None
    recovery_source: Optional[Placement] = None
    dependencies: tuple = field(default_factory=tuple)
    last_transition_epoch: int = -1
    last_observed_epoch: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.placements = set(self.placements) or {self.location}
        if self.location not in self.placements:
            self.location = next(iter(self.placements))
        if self.authoritative_placement is None and self.authoritative:
            self.authoritative_placement = self.location
        for placement in self.placements:
            self.replica_versions.setdefault(placement, self.version)
            self.replica_freshness_epochs.setdefault(placement, self.freshness_epoch)

    def is_stale(self, placement: Placement, tolerance_epochs: int = 0) -> bool:
        replica_epoch = self.replica_freshness_epochs.get(placement, -1)
        return self.freshness_epoch >= 0 and (
            self.freshness_epoch - replica_epoch > int(tolerance_epochs)
        )


@dataclass
class TransferRecord:
    key: str
    direction: str
    target: Placement
    start_epoch: int
    source: Optional[Placement] = None
    transfer_id: Optional[str] = None
    bytes: int = 0
    state: TransferState = TransferState.IN_FLIGHT
    end_epoch: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalDirectory:
    """Directory of actual replicas and their movement lifecycle."""

    def __init__(self):
        self._entries: Dict[str, PlacementEntry] = {}
        self._in_flight: Dict[str, TransferRecord] = {}
        self._history: List[dict] = []
        self._next_transfer_id = 0

    def register(
        self,
        key: str,
        location: Placement,
        authoritative: bool = False,
        *,
        placements: Optional[Iterable[Placement]] = None,
        version: int = 0,
        freshness_epoch: int = -1,
        owner_device: Optional[str] = None,
        authoritative_placement: Optional[Placement] = None,
        replica_versions: Optional[Dict[Placement, int]] = None,
        replica_freshness_epochs: Optional[Dict[Placement, int]] = None,
        writeback_required: bool = False,
        writeback_pending: bool = False,
        dependencies: Iterable[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PlacementEntry:
        entry = PlacementEntry(
            key=key,
            location=location,
            authoritative=bool(authoritative),
            placements=set(placements or {location}),
            authoritative_placement=authoritative_placement,
            version=int(version),
            freshness_epoch=int(freshness_epoch),
            owner_device=owner_device,
            writeback_required=bool(writeback_required),
            writeback_pending=bool(writeback_pending),
            dependencies=tuple(dependencies),
            metadata=dict(metadata or {}),
        )
        if replica_versions:
            entry.replica_versions.update(
                {placement: int(version) for placement, version in replica_versions.items()}
            )
        if replica_freshness_epochs:
            entry.replica_freshness_epochs.update(
                {
                    placement: int(epoch)
                    for placement, epoch in replica_freshness_epochs.items()
                }
            )
        self._entries[key] = entry
        return entry

    def register_handle(self, handle: StateHandle) -> PlacementEntry:
        placements = set(handle.placement) or {Placement.HOST_DRAM}
        location = handle.authoritative_placement or next(iter(placements))
        return self.register(
            handle.logical_key,
            location,
            authoritative=handle.authoritative_placement is not None,
            placements=placements,
            version=handle.version,
            freshness_epoch=handle.freshness_epoch,
            owner_device=handle.owner_device,
            authoritative_placement=handle.authoritative_placement,
            replica_versions=handle.replica_versions,
            replica_freshness_epochs=handle.replica_freshness_epochs,
            writeback_required=handle.writeback_required,
            writeback_pending=handle.writeback_pending,
            dependencies=handle.dependencies,
            metadata={
                **handle.metadata,
                "state_type": handle.state_type.name,
                "consistency_class": handle.consistency_class,
                "footprint_bytes": int(handle.footprint_bytes),
            },
        )

    def upsert_handle(self, handle: StateHandle) -> PlacementEntry:
        return self.register_handle(handle)

    def observe_handle(
        self, handle: StateHandle, *, epoch: Optional[int] = None
    ) -> PlacementEntry:
        """Merge an adapter observation without erasing transfer history."""
        entry = self._entries.get(handle.logical_key)
        if entry is None:
            entry = self.register_handle(handle)
        else:
            placements = set(handle.placement) or {Placement.HOST_DRAM}
            # An adapter can observe a stale physical snapshot while a copy
            # is in flight.  Keep the lifecycle's known replicas until the
            # transfer closes; otherwise a failed copy becomes an apparently
            # valid disappearance and later reads can use the wrong tier.
            if entry.in_flight:
                entry.metadata["observed_placements"] = sorted(
                    placement.name for placement in placements
                )
                if int(handle.version) < int(entry.version):
                    raise ConsistencyError(
                        f"stale observation for {handle.logical_key}: "
                        f"version {handle.version} < {entry.version}"
                    )
            else:
                if int(handle.version) < int(entry.version):
                    raise ConsistencyError(
                        f"stale observation for {handle.logical_key}: "
                        f"version {handle.version} < {entry.version}"
                    )
                entry.placements = placements
                entry.location = handle.authoritative_placement or next(iter(placements))
                entry.authoritative = handle.authoritative_placement is not None
                entry.authoritative_placement = handle.authoritative_placement
                entry.version = int(handle.version)
                entry.freshness_epoch = int(handle.freshness_epoch)
            entry.owner_device = handle.owner_device
            if not entry.in_flight:
                entry.replica_versions = {
                    placement: int(handle.replica_versions.get(placement, handle.version))
                    for placement in placements
                }
                entry.replica_freshness_epochs = {
                    placement: int(
                        handle.replica_freshness_epochs.get(
                            placement, handle.freshness_epoch
                        )
                    )
                    for placement in placements
                }
            entry.writeback_required = bool(handle.writeback_required)
            entry.writeback_pending = bool(handle.writeback_pending)
            entry.dependencies = tuple(handle.dependencies)
            if entry.in_flight:
                entry.metadata["observed_metadata"] = dict(handle.metadata)
            else:
                entry.metadata.update(handle.metadata)
            entry.metadata.setdefault("state_type", handle.state_type.name)
            entry.metadata["consistency_class"] = handle.consistency_class
            entry.metadata["footprint_bytes"] = int(handle.footprint_bytes)
            if not entry.in_flight:
                entry.transfer_state = TransferState.STABLE
        if epoch is not None:
            entry.last_transition_epoch = int(epoch)
            entry.last_observed_epoch = int(epoch)
        return entry

    def get(self, key: str) -> Optional[PlacementEntry]:
        return self._entries.get(key)

    def _pop_transfer(
        self, key: str, transfer_id: Optional[str] = None
    ) -> Optional[TransferRecord]:
        if transfer_id is not None:
            record = self._in_flight.get(transfer_id)
            if record is None or record.key != key:
                return None
            return self._in_flight.pop(transfer_id)
        # Compatibility path for callers that only know the logical key.  The
        # most recently inserted matching transfer is the old single-record
        # behavior when exactly one transfer exists.
        for candidate_id, record in reversed(tuple(self._in_flight.items())):
            if record.key == key:
                self._in_flight.pop(candidate_id, None)
                return record
        return None

    def _latest_transfer(self, key: str) -> Optional[TransferRecord]:
        for record in reversed(tuple(self._in_flight.values())):
            if record.key == key:
                return record
        return None

    def update_location(
        self,
        key: str,
        location: Placement,
        *,
        keep_replicas: bool = False,
        epoch: Optional[int] = None,
    ) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        if not keep_replicas:
            entry.placements = {location}
            entry.replica_versions = {location: entry.version}
            entry.replica_freshness_epochs = {location: entry.freshness_epoch}
        else:
            entry.placements.add(location)
            entry.replica_versions.setdefault(location, entry.version)
            entry.replica_freshness_epochs.setdefault(location, entry.freshness_epoch)
        entry.location = location
        if epoch is not None:
            entry.last_transition_epoch = int(epoch)

    def admit_replica(
        self,
        key: str,
        location: Placement,
        *,
        version: Optional[int] = None,
        freshness_epoch: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.placements.add(location)
        entry.replica_versions[location] = entry.version if version is None else int(version)
        entry.replica_freshness_epochs[location] = (
            entry.freshness_epoch if freshness_epoch is None else int(freshness_epoch)
        )
        entry.location = location
        if epoch is not None:
            entry.last_transition_epoch = int(epoch)

    def evict_replica(
        self,
        key: str,
        location: Placement,
        *,
        epoch: Optional[int] = None,
        allow_authoritative: bool = False,
    ) -> bool:
        entry = self._entries.get(key)
        if entry is None or location not in entry.placements:
            return False
        if (
            entry.authoritative_placement == location
            and len(entry.placements) > 1
            and not allow_authoritative
        ):
            return False
        entry.placements.discard(location)
        entry.replica_versions.pop(location, None)
        entry.replica_freshness_epochs.pop(location, None)
        if entry.location == location:
            entry.location = next(iter(entry.placements), Placement.HOST_DRAM)
        if epoch is not None:
            entry.last_transition_epoch = int(epoch)
        return True

    def set_authoritative(
        self,
        key: str,
        location: Placement,
        *,
        version: Optional[int] = None,
        freshness_epoch: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        self.admit_replica(
            key, location, version=version, freshness_epoch=freshness_epoch, epoch=epoch
        )
        entry.authoritative = True
        entry.authoritative_placement = location
        if version is not None:
            entry.version = int(version)
        if freshness_epoch is not None:
            entry.freshness_epoch = int(freshness_epoch)

    def hbm_residents(self) -> List[str]:
        return [k for k, e in self._entries.items() if Placement.HBM in e.placements]

    def hbm_bytes_used(self, registry) -> int:
        total = 0
        for key in self.hbm_residents():
            handle = registry.get(key)
            if handle:
                total += handle.footprint_bytes
        return total

    def has_writeback_obligation(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and (
            entry.authoritative or entry.writeback_required or entry.writeback_pending
        )

    def begin_transfer(
        self,
        key: str,
        direction: str,
        target: Placement,
        epoch: int,
        *,
        source: Optional[Placement] = None,
        transfer_id: Optional[str] = None,
        bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if transfer_id is None:
            transfer_id = f"transfer-{self._next_transfer_id}"
            self._next_transfer_id += 1
        if transfer_id in self._in_flight:
            raise ValueError(f"duplicate in-flight transfer id: {transfer_id}")
        entry = self._entries.get(key)
        if entry is not None:
            provisional_source = source is not None and any(
                transfer.key == key and transfer.target == source
                for transfer in self._in_flight.values()
            )
            if (
                source is not None
                and source not in entry.placements
                and not provisional_source
            ):
                raise ConsistencyError(
                    f"cannot start {direction} for {key}: "
                    f"source {source.name} is not resident"
                )
        self._in_flight[transfer_id] = TransferRecord(
            key=key,
            direction=direction,
            target=target,
            start_epoch=int(epoch),
            source=source,
            transfer_id=transfer_id,
            bytes=int(bytes),
            metadata=dict(metadata or {}),
        )
        if entry is not None:
            entry.in_flight = True
            entry.transfer_state = TransferState.IN_FLIGHT
            entry.transfer_id = transfer_id
            entry.last_error = None
            entry.last_transition_epoch = int(epoch)
        self._history.append({
            "event": "transfer_begin",
            "key": key,
            "transfer_id": transfer_id,
            "direction": direction,
            "source": source.name if source else None,
            "target": target.name,
            "epoch": int(epoch),
            "bytes": int(bytes),
        })
        return transfer_id

    def complete_transfer(
        self,
        key: str,
        *,
        target: Optional[Placement] = None,
        epoch: Optional[int] = None,
        authoritative: bool = False,
        version: Optional[int] = None,
        freshness_epoch: Optional[int] = None,
        transfer_id: Optional[str] = None,
    ) -> Optional[str]:
        record = (
            self._in_flight.get(transfer_id)
            if transfer_id is not None
            else self._latest_transfer(key)
        )
        if transfer_id is not None and (record is None or record.key != key):
            raise KeyError(f"unknown in-flight transfer {transfer_id!r} for state {key!r}")
        entry = self._entries.get(key)
        if entry is None:
            if record is not None:
                self._pop_transfer(key, record.transfer_id)
            return record.transfer_id if record else transfer_id
        target = target or (record.target if record else entry.location)
        if record is not None and record.source is not None:
            # A transfer may complete only from the replica/version that was
            # captured at begin_transfer.  This prevents an overlapping
            # offload/onload from committing an older source over a newer one.
            if record.source not in entry.placements:
                raise ConsistencyError(
                    f"source {record.source.name} for {key} disappeared "
                    "before transfer completion"
                )
        if record is not None:
            self._pop_transfer(key, record.transfer_id)
        self.admit_replica(
            key, target, version=version, freshness_epoch=freshness_epoch, epoch=epoch
        )
        if (
            record is not None
            and record.source is not None
            and record.source != target
            and record.direction in ("admission", "eviction", "writeback")
        ):
            self.evict_replica(
                key,
                record.source,
                epoch=epoch,
                allow_authoritative=authoritative,
            )
        if authoritative:
            self.set_authoritative(
                key,
                target,
                version=version,
                freshness_epoch=freshness_epoch,
                epoch=epoch,
            )
        remaining = self._latest_transfer(key)
        entry.in_flight = remaining is not None
        entry.transfer_state = (
            TransferState.IN_FLIGHT if remaining is not None else TransferState.COMPLETE
        )
        entry.transfer_id = (
            remaining.transfer_id
            if remaining is not None
            else (record.transfer_id if record else entry.transfer_id)
        )
        completed_transfer_id = record.transfer_id if record else transfer_id
        self._history.append({
            "event": "transfer_complete",
            "key": key,
            "transfer_id": completed_transfer_id,
            "target": target.name,
            "epoch": epoch,
            "authoritative": bool(authoritative),
        })
        return completed_transfer_id

    def complete_release(
        self,
        key: str,
        *,
        source: Placement,
        target: Optional[Placement] = None,
        epoch: Optional[int] = None,
        authoritative: bool = False,
        transfer_id: Optional[str] = None,
    ) -> Optional[str]:
        """Close a mapping release, which may not copy any bytes.

        A KV mapping can be dropped after its host copy already exists, or it
        can be dropped as reconstructible state with no remaining replica.
        Neither case is a transfer to an invented fallback placement, but the
        directory lifecycle still has to leave the in-flight state.
        """
        record = self._pop_transfer(key, transfer_id)
        if transfer_id is not None and record is None:
            raise KeyError(
                f"unknown in-flight transfer {transfer_id!r} for state {key!r}"
            )
        entry = self._entries.get(key)
        completed_transfer_id = record.transfer_id if record else transfer_id
        if entry is None:
            return completed_transfer_id

        if target is not None:
            self.admit_replica(key, target, epoch=epoch)
            if authoritative:
                self.set_authoritative(key, target, epoch=epoch)

        self.evict_replica(
            key,
            source,
            epoch=epoch,
            allow_authoritative=True,
        )
        if entry.authoritative_placement == source:
            if target is not None and authoritative:
                entry.authoritative = True
                entry.authoritative_placement = target
            else:
                entry.authoritative = False
                entry.authoritative_placement = None

        remaining = self._latest_transfer(key)
        entry.in_flight = remaining is not None
        entry.transfer_state = (
            TransferState.IN_FLIGHT if remaining is not None else TransferState.COMPLETE
        )
        entry.transfer_id = (
            remaining.transfer_id
            if remaining is not None
            else (completed_transfer_id or entry.transfer_id)
        )
        entry.last_error = None
        self._history.append({
            "event": "release_complete",
            "key": key,
            "transfer_id": completed_transfer_id,
            "source": source.name,
            "target": target.name if target is not None else None,
            "epoch": epoch,
            "authoritative": bool(authoritative),
        })
        return completed_transfer_id

    def fail_transfer(
        self,
        key: str,
        error: str,
        *,
        epoch: Optional[int] = None,
        transfer_id: Optional[str] = None,
    ) -> Optional[str]:
        record = self._pop_transfer(key, transfer_id)
        if transfer_id is not None and record is None:
            raise KeyError(
                f"unknown in-flight transfer {transfer_id!r} for state {key!r}"
            )
        entry = self._entries.get(key)
        failed_transfer_id = record.transfer_id if record else transfer_id
        if entry is not None:
            remaining = self._latest_transfer(key)
            entry.in_flight = remaining is not None
            entry.transfer_state = (
                TransferState.IN_FLIGHT if remaining is not None else TransferState.FAILED
            )
            entry.transfer_id = (
                remaining.transfer_id
                if remaining is not None
                else (failed_transfer_id or entry.transfer_id)
            )
            entry.last_error = str(error)
            entry.recovery_source = record.source if record else entry.authoritative_placement
            if epoch is not None:
                entry.last_transition_epoch = int(epoch)
        self._history.append({
            "event": "transfer_failed",
            "key": key,
            "transfer_id": failed_transfer_id,
            "epoch": epoch,
            "error": str(error),
        })
        return failed_transfer_id

    def mark_in_flight(
        self, key: str, direction: str, target: Placement, epoch: int
    ) -> None:
        entry = self._entries.get(key)
        self.begin_transfer(
            key,
            direction,
            target,
            epoch,
            source=entry.location if entry is not None else None,
        )

    def mark_complete(self, key: str) -> None:
        self.complete_transfer(key)

    def is_in_flight(self, key: str) -> bool:
        return any(record.key == key for record in self._in_flight.values())

    def validate_read(
        self,
        key: str,
        location: Placement,
        *,
        required_version: Optional[int] = None,
        allow_stale: bool = False,
    ) -> None:
        entry = self._entries.get(key)
        if entry is None:
            raise ConsistencyError(f"unknown state handle: {key}")
        if entry.in_flight and not allow_stale:
            raise ConsistencyError(
                f"{key} is in flight ({entry.transfer_id}); read is not stable"
            )
        if location not in entry.placements:
            raise ConsistencyError(f"{key} has no {location.name} replica")
        segments = entry.metadata.get("segments") or {}
        if required_version is not None and segments.get("append_only"):
            required = int(required_version)
            if location == Placement.HOST_DRAM:
                covered = int(segments.get("host_prefix_tokens", 0))
            elif location == Placement.HBM:
                covered = int(segments.get("hbm_start_token", 0)) + int(
                    segments.get("hbm_tokens", 0)
                )
            else:
                covered = int(segments.get("logical_length", 0))
            if covered < required:
                raise ConsistencyError(
                    f"{key} {location.name} segment covers {covered} tokens, "
                    f"requested {required}"
                )
        replica_version = entry.replica_versions.get(location, -1)
        if required_version is not None and replica_version < int(required_version):
            raise ConsistencyError(
                f"{key} {location.name} version {replica_version} "
                f"is below required {required_version}"
            )
        if not allow_stale and entry.is_stale(location):
            raise ConsistencyError(f"{key} {location.name} replica is stale")

    def pending_count(self) -> int:
        return len(self._in_flight)

    def snapshot(self) -> List[dict]:
        return [
            {
                "logical_key": entry.key,
                "location": entry.location.name,
                "placements": sorted(p.name for p in entry.placements),
                "authoritative": bool(entry.authoritative),
                "authoritative_placement": (
                    entry.authoritative_placement.name
                    if entry.authoritative_placement is not None else None
                ),
                "version": int(entry.version),
                "freshness_epoch": int(entry.freshness_epoch),
                "owner_device": entry.owner_device,
                "replica_versions": {
                    p.name: int(v) for p, v in entry.replica_versions.items()
                },
                "replica_freshness_epochs": {
                    p.name: int(v)
                    for p, v in entry.replica_freshness_epochs.items()
                },
                "in_flight": bool(entry.in_flight),
                "transfer_state": entry.transfer_state.name,
                "transfer_id": entry.transfer_id,
                "writeback_required": bool(entry.writeback_required),
                "writeback_pending": bool(entry.writeback_pending),
                "last_error": entry.last_error,
                "recovery_source": (
                    entry.recovery_source.name if entry.recovery_source else None
                ),
                "dependencies": list(entry.dependencies),
                "last_transition_epoch": int(entry.last_transition_epoch),
                "last_observed_epoch": int(entry.last_observed_epoch),
                "metadata": dict(entry.metadata),
            }
            for entry in self._entries.values()
        ]

    def history(self) -> List[dict]:
        return list(self._history)
