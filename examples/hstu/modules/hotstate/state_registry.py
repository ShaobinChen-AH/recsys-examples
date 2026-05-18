from collections import defaultdict
from typing import Dict, List, Optional

from modules.hotstate.state_handle import StateHandle, StateType, Placement


class StateRegistry:
    def __init__(self):
        self._by_key: Dict[str, StateHandle] = {}
        self._by_type: Dict[StateType, List[str]] = defaultdict(list)

    def register(self, handle: StateHandle) -> None:
        key = handle.logical_key
        self._by_key[key] = handle
        stype = handle.state_type
        if key not in self._by_type[stype]:
            self._by_type[stype].append(key)

    def get(self, key: str) -> Optional[StateHandle]:
        return self._by_key.get(key)

    def get_by_type(self, stype: StateType) -> List[StateHandle]:
        return [self._by_key[k] for k in self._by_type.get(stype, [])
                if k in self._by_key]

    def snapshot(self) -> List[StateHandle]:
        return list(self._by_key.values())

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
        self._by_key.clear()
        self._by_type.clear()
