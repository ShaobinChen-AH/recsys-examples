from collections import defaultdict
from typing import Dict, List, Optional

from modules.hotstate.state_handle import (
    StateHandle,
    StateType,
    Placement,
    TransferState,
)


class StateRegistry:
    def __init__(self):
        self._by_key: Dict[str, StateHandle] = {}
        self._by_type: Dict[StateType, List[str]] = defaultdict(list)
        self._epoch: int = -1
        self._history: List[dict] = []

    def register(self, handle: StateHandle) -> None:
        key = handle.logical_key
        previous = self._by_key.get(key)
        if previous is not None and previous.state_type != handle.state_type:
            old_keys = self._by_type.get(previous.state_type, [])
            if key in old_keys:
                old_keys.remove(key)
        self._by_key[key] = handle
        stype = handle.state_type
        if key not in self._by_type[stype]:
            self._by_type[stype].append(key)

    def upsert(self, handle: StateHandle) -> StateHandle:
        self.register(handle)
        return handle

    def get(self, key: str) -> Optional[StateHandle]:
        return self._by_key.get(key)

    def get_by_type(self, stype: StateType) -> List[StateHandle]:
        return [self._by_key[k] for k in self._by_type.get(stype, [])
                if k in self._by_key]

    def snapshot(self) -> List[StateHandle]:
        return list(self._by_key.values())

    def begin_epoch(self, epoch: int, *, clear: bool = True) -> None:
        if self._by_key:
            self._history.append({
                "epoch": self._epoch,
                "handles": self.snapshot_records(),
            })
        self._epoch = int(epoch)
        if clear:
            self._by_key.clear()
            self._by_type.clear()

    def update(self, key: str, **changes) -> StateHandle:
        handle = self._by_key.get(key)
        if handle is None:
            raise KeyError(f"state handle is not registered: {key}")
        for name, value in changes.items():
            if not hasattr(handle, name):
                raise AttributeError(f"unknown StateHandle field: {name}")
            setattr(handle, name, value)
        self.register(handle)
        return handle

    def record_transition(
        self,
        key: str,
        *,
        action: str,
        source: Optional[Placement] = None,
        target: Optional[Placement] = None,
        epoch: Optional[int] = None,
        success: bool = True,
        transfer_id: Optional[str] = None,
        error: Optional[str] = None,
        authoritative: bool = False,
        version: Optional[int] = None,
        freshness_epoch: Optional[int] = None,
    ) -> None:
        handle = self._by_key.get(key)
        if handle is not None:
            if success:
                handle.validate_transition(
                    source,
                    target,
                    version=version,
                    authoritative=authoritative,
                )
            if success and source is not None and target is not None and source != target:
                handle.remove_replica(source)
                handle.add_replica(target)
            elif success and target is not None:
                handle.add_replica(target)
            handle.transfer_id = transfer_id
            handle.transfer_state = (
                TransferState.COMPLETE if success else TransferState.FAILED
            )
            handle.last_error = error
            if not success:
                handle.recovery_source = source or handle.authoritative_placement
            if success and target is not None and authoritative:
                handle.set_authoritative(
                    target,
                    version=version,
                    freshness_epoch=freshness_epoch,
                )
            if version is not None:
                handle.version = int(version)
            if freshness_epoch is not None:
                handle.freshness_epoch = int(freshness_epoch)
        self._history.append({
            "epoch": self._epoch if epoch is None else int(epoch),
            "key": key,
            "action": action,
            "source": source.name if source is not None else None,
            "target": target.name if target is not None else None,
            "success": bool(success),
            "transfer_id": transfer_id,
            "error": error,
            "authoritative": bool(authoritative),
            "version": version,
            "freshness_epoch": freshness_epoch,
        })

    def snapshot_records(self) -> List[dict]:
        return [
            {
                "logical_key": handle.logical_key,
                "state_type": handle.state_type.name,
                "footprint_bytes": int(handle.footprint_bytes),
                "placements": sorted(p.name for p in handle.placement),
                "authoritative_placement": (
                    handle.authoritative_placement.name
                    if handle.authoritative_placement is not None else None
                ),
                "version": int(handle.version),
                "freshness_epoch": int(handle.freshness_epoch),
                "owner_device": handle.owner_device,
                "transfer_state": handle.transfer_state.name,
                "transfer_id": handle.transfer_id,
                "transfer_cost_ms": float(handle.transfer_cost_ms),
                "reconstruction_cost_ms": float(handle.reconstruction_cost_ms),
                "expected_reuse_window": int(handle.expected_reuse_window),
                "staleness_tolerance_epochs": int(
                    handle.staleness_tolerance_epochs
                ),
                "ttl_epochs": handle.ttl_epochs,
                "writeback_required": bool(handle.writeback_required),
                "writeback_pending": bool(handle.writeback_pending),
                "dependencies": list(handle.dependencies),
                "last_error": handle.last_error,
                "recovery_source": (
                    handle.recovery_source.name
                    if handle.recovery_source is not None else None
                ),
                "metadata": dict(handle.metadata),
                "last_access_epoch": int(handle.last_access_epoch),
                "access_count": int(handle.access_count),
            }
            for handle in self._by_key.values()
        ]

    def history(self) -> List[dict]:
        return list(self._history)

    def hbm_footprint(self) -> int:
        return sum(h.footprint_bytes for h in self._by_key.values()
                   if Placement.HBM in h.placement)

    def remove(self, key: str) -> None:
        if key in self._by_key:
            stype = self._by_key[key].state_type
            self._by_key.pop(key, None)
            if key in self._by_type.get(stype, []):
                self._by_type[stype].remove(key)

    def clear(self) -> None:
        self.begin_epoch(self._epoch, clear=True)
