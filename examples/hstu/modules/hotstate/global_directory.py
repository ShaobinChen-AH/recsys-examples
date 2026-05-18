from dataclasses import dataclass
from typing import Dict, List, Optional

from modules.hotstate.state_handle import Placement


@dataclass
class PlacementEntry:
    key: str
    location: Placement
    authoritative: bool
    pinned: bool = False
    in_flight: bool = False


@dataclass
class TransferRecord:
    key: str
    direction: str        # "eviction" or "admission"
    target: Placement
    start_epoch: int


class GlobalDirectory:
    def __init__(self):
        self._entries: Dict[str, PlacementEntry] = {}
        self._in_flight: Dict[str, TransferRecord] = {}

    # Placement registry

    def register(self, key: str, location: Placement,
                 authoritative: bool = False) -> None:
        self._entries[key] = PlacementEntry(
            key=key, location=location, authoritative=authoritative)

    def get(self, key: str) -> Optional[PlacementEntry]:
        return self._entries.get(key)

    def update_location(self, key: str, location: Placement) -> None:
        if key in self._entries:
            self._entries[key].location = location

    def hbm_residents(self) -> List[str]:
        return [k for k, e in self._entries.items()
                if e.location == Placement.HBM]

    def hbm_bytes_used(self, registry) -> int:
        total = 0
        for key in self.hbm_residents():
            h = registry.get(key)
            if h:
                total += h.footprint_bytes
        return total

    def has_writeback_obligation(self, key: str) -> bool:
        e = self._entries.get(key)
        return e is not None and e.authoritative

    # In-flight transfer tracking

    def mark_in_flight(self, key: str, direction: str,
                       target: Placement, epoch: int) -> None:
        self._in_flight[key] = TransferRecord(
            key=key, direction=direction, target=target, start_epoch=epoch)
        if key in self._entries:
            self._entries[key].in_flight = True

    def mark_complete(self, key: str) -> None:
        self._in_flight.pop(key, None)
        if key in self._entries:
            self._entries[key].in_flight = False

    def is_in_flight(self, key: str) -> bool:
        return key in self._in_flight

    def pending_count(self) -> int:
        return len(self._in_flight)
