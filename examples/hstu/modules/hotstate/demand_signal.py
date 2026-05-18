from dataclasses import dataclass, field
from typing import List


@dataclass
class DemandSignal:
    """Context passed to the controller before each batch."""
    current_user_id: int
    history_length: int          # per-feature history tokens
    num_candidates: int
    epoch: int
    item_indices: List[int] = field(default_factory=list)
