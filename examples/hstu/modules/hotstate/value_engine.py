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

    def configure(self, kv_ms_per_1k_tokens=None, emb_ms_per_key=None):
        if kv_ms_per_1k_tokens is not None:
            self.DEFAULT_KV_STALL_PER_1000_TOKENS_MS = float(kv_ms_per_1k_tokens)
        if emb_ms_per_key is not None:
            self.DEFAULT_EMB_STALL_PER_MISS_MS = float(emb_ms_per_key)

    def compute_scores(self, handles: List[StateHandle], demand: DemandSignal) -> List[ScoredHandle]:
        scored = []
        for h in handles:
            reuse = self._reuse_imminence(h, demand)
            miss_cost = self._stall_sensitivity(h, demand)
            movement_cost = h.footprint_bytes / self.PCIE_BANDWIDTH_BYTES_PER_MS
            risk_cost = self.SEMANTIC_RISK_MS if "writeback" in h.consistency_class else 0.0

            gross_benefit = reuse * miss_cost
            net_benefit = gross_benefit - movement_cost - risk_cost
            value_density = net_benefit / max(1, h.footprint_bytes)

            h.reuse_imminence = reuse
            h.stall_sensitivity_ms = miss_cost
            h.movement_cost_ms = movement_cost

            scored.append(ScoredHandle(
                handle=h,
                score=value_density,
                benefit_density=value_density,
                occupancy_penalty=0.0,
                semantic_risk=risk_cost,
                reuse_probability=reuse,
                miss_cost_ms=miss_cost,
                gross_benefit_ms=gross_benefit,
                movement_cost_ms=movement_cost,
                risk_cost_ms=risk_cost,
                net_benefit_ms=net_benefit,
                value_density_ms_per_byte=value_density,
            ))

        scored.sort(
            key=lambda s: (s.value_density_ms_per_byte, s.net_benefit_ms),
            reverse=True,
        )
        return scored

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
        steps = (h_uid - current_uid + max(1, int(demand.num_users))) % max(1, int(demand.num_users))
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
    
    def rank_embedding_item_indices(self, item_indices, demand, row_size_bytes, item_sequence=None):
        sequence = [int(x) for x in (item_sequence or item_indices)]
        seen = set()
        unique_keys = []
        for raw_key in item_indices:
            key = int(raw_key)
            if key in seen:
                continue
            seen.add(key)
            unique_keys.append(key)

        counts = {}
        last_pos = {}
        for pos, key in enumerate(sequence):
            counts[key] = counts.get(key, 0) + 1
            last_pos[key] = pos

        ranked = []
        seq_len = max(1, len(sequence))
        hist_len = max(0, int(getattr(demand, "history_length", 0)))

        for ordinal, key in enumerate(unique_keys):
            count = counts.get(key, 1)
            pos = last_pos.get(key, -1)
            in_history = 1.0 if 0 <= pos < hist_len else 0.0
            position_score = (pos + 1) / seq_len if pos >= 0 else 0.0

            prior = self._access_log.get(f"emb:item:{key}", [])
            recent_score = 0.0
            if prior:
                age = max(0, demand.epoch - prior[-1])
                recent_score = math.exp(-age / 8.0)

            reuse = min(
                1.0,
                0.65
                + 0.20 * in_history
                + 0.05 * min(1.0, math.log1p(count))
                + 0.05 * position_score
                + 0.05 * recent_score,
            )

            miss_cost = self.DEFAULT_EMB_STALL_PER_MISS_MS * max(1, count)
            movement_cost = row_size_bytes / self.PCIE_BANDWIDTH_BYTES_PER_MS
            net_benefit = reuse * miss_cost - movement_cost
            value_density = net_benefit / max(1, row_size_bytes)

            if net_benefit <= 0.0:
                continue

            ranked.append((value_density, net_benefit, in_history, count, recent_score, -ordinal, key))

        ranked.sort(reverse=True)
        return [key for *_, key in ranked]

    def record_embedding_accesses(self, item_indices, epoch: int):
        seen = set()
        for raw_key in item_indices:
            key = int(raw_key)
            if key in seen:
                continue
            seen.add(key)
            self._access_log[f"emb:item:{key}"].append(epoch)
