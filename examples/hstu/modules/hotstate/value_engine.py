import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from modules.hotstate.state_handle import (
    StateHandle, StateType, ScoredHandle, Reconstructability, Placement
)
from modules.hotstate.demand_signal import DemandSignal


class ValueEngine:
    """Compute arbitration scores using the HotState value formula.

    score(s) = benefit_density - occupancy_penalty - semantic_risk

    where:
        benefit_density = (reuse_imminence * stall_sensitivity) / max(1, footprint)
        occupancy_penalty = footprint / total_hbm_bytes
        semantic_risk = writeback_penalty if authoritative else 0
    """

    # Stall sensitivity: how much latency is saved if object is HBM-resident.
    # Measured during warmup calibration.
    DEFAULT_KV_STALL_PER_1000_TOKENS_MS = 3.5   # ms saved per 1000 history tokens
    DEFAULT_EMB_STALL_PER_MISS_MS = 1.5          # ms saved per embedding miss avoided

    # PCIe Gen4 x16: ~25 GB/s to 0.025 MB/ms to 25,000 B/ms
    PCIE_BANDWIDTH_BYTES_PER_MS = 25_000_000

    SEMANTIC_RISK_MS = 0.05

    def __init__(self, total_hbm_bytes: int):
        self.total_hbm = total_hbm_bytes
        self._access_log: Dict[str, List[int]] = defaultdict(list)
        self._hot_key_counts: Dict[str, int] = {}    # key to unique hot row count

    def _semantic_risk_ms(self, h: StateHandle) -> float:
        if h.consistency_class == "mutable_writeback":
            return self.SEMANTIC_RISK_MS
        return 0.0

    def compute_scores(self, handles: List[StateHandle],
                       demand: DemandSignal) -> List[ScoredHandle]:
        """Compute scores for all handles given the current batch context."""
        scored = []
        for h in handles:
            reuse = self._reuse_imminence(h, demand)
            h.reuse_imminence = reuse

            stall = self._stall_sensitivity(h, demand)
            h.stall_sensitivity_ms = stall

            move = h.footprint_bytes / self.PCIE_BANDWIDTH_BYTES_PER_MS
            h.movement_cost_ms = move

            gross_benefit_ms = reuse * stall
            movement_penalty_ms = 0.0 if Placement.HBM in h.placement else move
            semantic_risk_ms = self._semantic_risk_ms(h)

            net_benefit_ms = gross_benefit_ms - movement_penalty_ms - semantic_risk_ms
            value_density_ms_per_byte = net_benefit_ms / max(1, h.footprint_bytes)

            benefit_density = gross_benefit_ms / max(1, h.footprint_bytes)
            occupancy = h.footprint_bytes / self.total_hbm

            score = value_density_ms_per_byte
            scored.append(ScoredHandle(
                handle=h,
                score=score,
                benefit_density=benefit_density,
                occupancy_penalty=occupancy,
                semantic_risk=semantic_risk_ms,
                gross_benefit_ms=gross_benefit_ms,
                movement_penalty_ms=movement_penalty_ms,
                semantic_risk_ms=semantic_risk_ms,
                net_benefit_ms=net_benefit_ms,
                value_density_ms_per_byte=value_density_ms_per_byte,
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    # Stall sensitivity

    def _stall_sensitivity(self, h: StateHandle, demand: DemandSignal) -> float:
        if h.state_type == StateType.SESSION_KV_USER:
            tokens = demand.history_length
            return self.DEFAULT_KV_STALL_PER_1000_TOKENS_MS * (tokens / 1000.0)
        elif h.state_type == StateType.EMBEDDING_HOT_ROWS:
            return self.DEFAULT_EMB_STALL_PER_MISS_MS
        else:  # EMBEDDING_COLD_ROWS
            return 0.05  # cold rows: very low stall impact

    # Reuse imminence

    def _reuse_imminence(self, h: StateHandle, demand: DemandSignal) -> float:
        if h.state_type == StateType.SESSION_KV_USER:
            return self._kv_reuse(h, demand)
        elif h.state_type == StateType.EMBEDDING_HOT_ROWS:
            return 1.0    # hot rows: always accessed
        else:
            return 0.01   # cold rows: rarely accessed

    def _kv_reuse(self, h: StateHandle, demand: DemandSignal) -> float:
        """Predict when this user will be served next.

        Dataset ordering: within a seqlen group, users cycle 0 to 1 to...to 7.
        After user 7, seqlen increments and cycle restarts at user 0.
        So steps_until = (h_uid - current_uid + num_users) % num_users
                         + future_seqlen_rounds * num_users
        """
        try:
            h_uid = int(h.logical_key.split(":")[-1])
        except (ValueError, IndexError):
            return 0.5  # fallback

        current_uid = demand.current_user_id
        # Same seqlen group: (h_uid - current_uid + N) % N
        steps = (h_uid - current_uid + 8) % 8
        # For next seqlen round, add more steps
        import math
        return math.exp(-0.5 * steps)

    # Access tracking

    def record_access(self, key: str, epoch: int):
        self._access_log[key].append(epoch)

    def set_hot_key_count(self, emb_key: str, count: int):
        self._hot_key_counts[emb_key] = count

    def decay_logs(self, current_epoch: int, max_age: int = 20):
        for key in list(self._access_log.keys()):
            self._access_log[key] = [
                e for e in self._access_log[key]
                if current_epoch - e <= max_age
            ]
            if not self._access_log[key]:
                del self._access_log[key]
