from collections import deque
from typing import List, Set

from modules.hotstate.state_handle import (
    StateHandle, StateType, Placement, Reconstructability
)


class EmbeddingAdapter:
    """Bridges DynamicEmbedding into HotState at hot/cold row-group granularity.

    Maintains a sliding window of recently-accessed item keys to estimate
    the current hot-working-set footprint.
    """

    DEFAULT_ROW_BYTES = 1024   # dim(512) bf16(2 bytes) = 1024
    SLIDING_WINDOW_SIZE = 10000  # max recent keys to track

    def __init__(self, embedding_module):
        self._module = embedding_module
        self._recent_keys: deque = deque(maxlen=self.SLIDING_WINDOW_SIZE)
        self._total_size_bytes = 0
        self._row_size_bytes = self.DEFAULT_ROW_BYTES

    def calibrate(self):
        """Read actual table dimensions from the embedding module."""
        if self._module is None:
            return
        try:
            storage = self._module._storage
            if hasattr(storage, 'tables'):
                for t in storage.tables:
                    # Estimate row size from buffer properties
                    if hasattr(t, 'num_bytes') and hasattr(t, 'size'):
                        self._total_size_bytes += t.num_bytes()
        except Exception:
            self._total_size_bytes = 10_000_000 * self.DEFAULT_ROW_BYTES

    def record_batch_keys(self, item_indices: List[int]):
        """Feed this batch's item_feat values into the sliding window."""
        for idx in item_indices:
            self._recent_keys.append(idx)
    
    def set_module(self, embedding_module):
        self._module = embedding_module
        self._total_size_bytes = 0
        self.calibrate()    

    def hot_key_count(self) -> int:
        """Number of unique keys seen in the recent window."""
        return len(set(self._recent_keys))

    def hot_footprint_bytes(self) -> int:
        return self.hot_key_count() * self._row_size_bytes

    def cold_footprint_bytes(self) -> int:
        return max(0, self._total_size_bytes - self.hot_footprint_bytes())

    def export_handles(self) -> List[StateHandle]:
        """Export hot and cold row-group handles."""
        handles = [
            StateHandle(
                state_type=StateType.EMBEDDING_HOT_ROWS,
                logical_key="emb:item:hot_rows",
                footprint_bytes=self.hot_footprint_bytes(),
                placement={Placement.HOST_DRAM},
                reconstructability=Reconstructability.REFETCHABLE,
                consistency_class="mutable_writeback",
            ),
            StateHandle(
                state_type=StateType.EMBEDDING_COLD_ROWS,
                logical_key="emb:item:cold_rows",
                footprint_bytes=self.cold_footprint_bytes(),
                placement={Placement.HOST_DRAM},
                reconstructability=Reconstructability.REFETCHABLE,
                consistency_class="refetchable",
            ),
        ]
        return handles
    def trigger_flush(self):
        if self._module is not None and hasattr(self._module, 'flush'):
            self._module.flush()
