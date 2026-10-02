import math
from collections import defaultdict
from typing import Dict, List, Optional

from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.online_cost_model import (
    BatchCostObservation,
    OnlineSystemCostModel,
)
from modules.hotstate.state_handle import (
    Placement,
    ScoredHandle,
    StateHandle,
    StateType,
)


class ValueEngine:
    """Measurement-driven online value model.

    Batch latency, actual misses, transfer timing, observed reuse distances,
    queue pressure, and the HBM allocation frontier continuously update the
    model. Cold-start priors remain visible and decay as observations arrive.
    """

    def __init__(self, total_hbm_bytes: int, cost_model=None):
        self.total_hbm = int(total_hbm_bytes)
        self.cost_model = cost_model or OnlineSystemCostModel()
        self._access_log: Dict[str, List[int]] = defaultdict(list)
        self._hot_key_counts: Dict[str, int] = {}
        self._embedding_last_access_epoch_by_item: Dict[int, int] = {}
        self._last_access_epoch_by_key: Dict[str, int] = {}
        self._reuse_interval_ewma_by_key: Dict[str, float] = {}
        self._reuse_probability_ewma_by_type: Dict[StateType, float] = {
            StateType.EMBEDDING_HOT_ROWS: 0.50,
            StateType.EMBEDDING_COLD_ROWS: 0.10,
            StateType.SESSION_KV_USER: 0.50,
        }
        self._configured_kv_prior = None
        self._configured_embedding_prior = None

    def configure(self, kv_ms_per_1k_tokens=None, emb_ms_per_key=None):
        """Set cold-start priors without disabling online updates."""
        coefficients = self.cost_model.batch_model.theta
        names = self.cost_model.batch_model.feature_names
        if kv_ms_per_1k_tokens is not None:
            coefficients[names.index("kv_uncached_k_tokens")] = max(
                0.0, float(kv_ms_per_1k_tokens)
            )
            self._configured_kv_prior = float(kv_ms_per_1k_tokens)
        if emb_ms_per_key is not None:
            coefficients[names.index("embedding_miss_hundreds")] = max(
                0.0, float(emb_ms_per_key) * 100.0
            )
            self._configured_embedding_prior = float(emb_ms_per_key)

    def observe_batch_cost(self, **kwargs) -> None:
        self.cost_model.observe_batch(BatchCostObservation(**kwargs))

    def observe_transfer(self, **kwargs) -> None:
        self.cost_model.observe_transfer(**kwargs)

    def observe_allocation(self, scored, selected, used_bytes: int) -> None:
        selected_keys = {handle.logical_key for handle in selected}
        self.cost_model.observe_allocation(
            candidate_densities_ms_per_byte=(
                item.value_density_ms_per_byte for item in scored
            ),
            selected_densities_ms_per_byte=(
                item.value_density_ms_per_byte
                for item in scored
                if item.handle.logical_key in selected_keys
            ),
            used_bytes=used_bytes,
            budget_bytes=self.total_hbm,
        )

    def cost_model_snapshot(self) -> dict:
        snapshot = self.cost_model.snapshot()
        snapshot["configured_kv_prior_ms_per_1k_tokens"] = self._configured_kv_prior
        snapshot["configured_embedding_prior_ms_per_key"] = self._configured_embedding_prior
        snapshot["tracked_access_keys"] = len(self._last_access_epoch_by_key)
        snapshot["learned_reuse_intervals"] = len(self._reuse_interval_ewma_by_key)
        return snapshot

    def _semantic_risk_ms(self, handle: StateHandle, epoch: Optional[int] = None) -> float:
        tail_unit = max(
            0.01,
            self.cost_model.observed_tail_ewma_ms * 0.01
            if self.cost_model.observed_tail_ewma_ms > 0.0
            else 0.05,
        )
        risk = tail_unit if handle.consistency_class == "mutable_writeback" else 0.0
        if handle.writeback_pending:
            risk += tail_unit
        if (
            Placement.HBM in handle.placement
            and handle.authoritative_placement is not None
            and handle.authoritative_placement != Placement.HBM
            and handle.is_stale(Placement.HBM)
        ):
            risk += tail_unit
        if epoch is not None and handle.is_expired(epoch):
            risk += max(tail_unit, handle.reconstruction_cost_ms)
        return risk

    @staticmethod
    def _state_class(handle: StateHandle) -> str:
        return "kv" if handle.state_type == StateType.SESSION_KV_USER else "embedding"

    def compute_scores(
        self, handles: List[StateHandle], demand: DemandSignal
    ) -> List[ScoredHandle]:
        scored = []
        for handle in handles:
            reuse = self._reuse_imminence(handle, demand)
            miss_cost = self._stall_sensitivity(handle, demand)
            is_resident = Placement.HBM in handle.placement
            movement_cost = 0.0
            if not is_resident:
                movement_cost = self.cost_model.estimate_transfer_ms(
                    state_class=self._state_class(handle),
                    direction="admission",
                    bytes_moved=handle.footprint_bytes,
                )
                if handle.transfer_cost_ms > 0.0 and self.cost_model.confidence < 1.0:
                    confidence = self.cost_model.confidence
                    movement_cost = (
                        confidence * movement_cost
                        + (1.0 - confidence) * handle.transfer_cost_ms
                    )
                movement_cost += self.cost_model.concurrent_transfer_penalty_ms(
                    handle.footprint_bytes,
                    self.cost_model.last_transfer_queue_depth,
                )
            risk_cost = self._semantic_risk_ms(handle, demand.epoch)
            opportunity_cost = self.cost_model.opportunity_cost_ms(
                handle.footprint_bytes
            )
            reconstruction_credit = (
                max(0.0, handle.reconstruction_cost_ms) if is_resident else 0.0
            )

            gross_benefit = reuse * (miss_cost + reconstruction_credit)
            net_benefit = (
                gross_benefit - movement_cost - risk_cost - opportunity_cost
            )
            value_density = net_benefit / max(1, handle.footprint_bytes)

            handle.reuse_imminence = reuse
            handle.stall_sensitivity_ms = miss_cost
            handle.movement_cost_ms = movement_cost
            if handle.expected_reuse_window <= 0:
                interval = self._reuse_interval_ewma_by_key.get(handle.logical_key)
                if interval is not None:
                    handle.expected_reuse_window = max(1, int(round(interval)))

            reason = (
                f"online_cost(confidence={self.cost_model.confidence:.2f},"
                f" reuse={reuse:.3f}, miss_ms={miss_cost:.4f},"
                f" move_ms={movement_cost:.4f}, opportunity_ms={opportunity_cost:.4f})"
            )
            scored.append(
                ScoredHandle(
                    handle=handle,
                    score=value_density,
                    benefit_density=gross_benefit / max(1, handle.footprint_bytes),
                    occupancy_penalty=opportunity_cost,
                    semantic_risk=risk_cost,
                    reuse_probability=reuse,
                    miss_cost_ms=miss_cost,
                    gross_benefit_ms=gross_benefit,
                    movement_cost_ms=movement_cost,
                    risk_cost_ms=risk_cost,
                    net_benefit_ms=net_benefit,
                    value_density_ms_per_byte=value_density,
                    decision_reason=reason,
                    opportunity_cost_ms=opportunity_cost,
                )
            )

        scored.sort(
            key=lambda item: (item.value_density_ms_per_byte, item.net_benefit_ms),
            reverse=True,
        )
        return scored

    def _stall_sensitivity(self, handle: StateHandle, demand: DemandSignal) -> float:
        if handle.state_type == StateType.SESSION_KV_USER:
            try:
                uid = int(handle.logical_key.rsplit(":", 1)[-1])
            except (ValueError, IndexError):
                uid = None
            page_bytes = max(1, int(handle.metadata.get("page_bytes", 1)))
            page_count = max(1, int(handle.metadata.get("page_count", 1)))
            page_tokens = max(1, int(handle.metadata.get("page_size_tokens", 1)))
            cached_tokens = page_count * page_tokens
            # A KV handle represents already-materialized session state. Its
            # residency value is the measured cost of recovering those cached
            # tokens, not the unrelated cost of appending the current batch.
            miss_tokens = max(1, cached_tokens)
            reconstruction_cost = self.cost_model.kv_recompute_ms(miss_tokens)
            if Placement.HBM not in handle.placement:
                reconstruction_cost += self.cost_model.kv_residency_miss_ms(
                    handle.footprint_bytes
                )
            return reconstruction_cost
        misses = int(handle.metadata.get("request_count", 1))
        return self.cost_model.embedding_miss_ms(max(1, misses))

    def _reuse_imminence(self, handle: StateHandle, demand: DemandSignal) -> float:
        key = handle.logical_key
        current_epoch = int(demand.epoch)
        if key in self._reuse_interval_ewma_by_key:
            interval = max(1.0, self._reuse_interval_ewma_by_key[key])
            last_access = self._last_access_epoch_by_key.get(key, current_epoch)
            age = max(0, current_epoch - last_access)
            return min(1.0, math.exp(-float(age) / interval))

        type_prior = self._reuse_probability_ewma_by_type.get(handle.state_type, 0.25)
        if handle.state_type == StateType.SESSION_KV_USER:
            try:
                uid = int(key.rsplit(":", 1)[-1])
            except (ValueError, IndexError):
                uid = None
            if uid == demand.current_user_id:
                return 1.0
        elif key.startswith("emb:item:"):
            try:
                item_id = int(key.rsplit(":", 1)[-1])
            except (ValueError, IndexError):
                item_id = None
            if item_id is not None and item_id in demand.item_indices:
                return 1.0
        return max(0.01, min(1.0, type_prior))

    def record_access(self, key: str, epoch: int, state_type: Optional[StateType] = None):
        epoch = int(epoch)
        previous = self._last_access_epoch_by_key.get(key)
        if previous is not None and epoch > previous:
            interval = float(epoch - previous)
            old_interval = self._reuse_interval_ewma_by_key.get(key, interval)
            self._reuse_interval_ewma_by_key[key] = 0.8 * old_interval + 0.2 * interval
            if state_type is None:
                if key.startswith("kv:uid:"):
                    state_type = StateType.SESSION_KV_USER
                elif key.startswith("emb:item:"):
                    state_type = StateType.EMBEDDING_HOT_ROWS
            if state_type is not None:
                probability = 1.0 / max(1.0, interval)
                old_probability = self._reuse_probability_ewma_by_type.get(
                    state_type, probability
                )
                self._reuse_probability_ewma_by_type[state_type] = (
                    0.9 * old_probability + 0.1 * probability
                )
        self._last_access_epoch_by_key[key] = epoch
        self._access_log[key].append(epoch)

    def set_hot_key_count(self, emb_key: str, count: int):
        self._hot_key_counts[emb_key] = count

    def decay_logs(self, current_epoch: int, max_age: int = 128):
        for key in list(self._access_log.keys()):
            self._access_log[key] = [
                epoch
                for epoch in self._access_log[key]
                if current_epoch - epoch <= max_age
            ]
            if not self._access_log[key]:
                del self._access_log[key]
        for key, last_epoch in list(self._embedding_last_access_epoch_by_item.items()):
            if current_epoch - last_epoch > max_age:
                del self._embedding_last_access_epoch_by_item[key]

    def rank_embedding_item_indices(
        self,
        item_indices,
        item_sequence,
        demand,
        row_size_bytes,
        return_trace: bool = False,
    ):
        sequence = [int(value) for value in (item_sequence or item_indices)]
        counts: Dict[int, int] = defaultdict(int)
        last_position: Dict[int, int] = {}
        unique_keys = []
        seen = set()
        for position, key in enumerate(sequence):
            counts[key] += 1
            last_position[key] = position
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        for raw_key in item_indices:
            key = int(raw_key)
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
                counts[key] = max(1, counts[key])

        ranked = []
        epoch = int(demand.epoch)
        row_bytes = max(1, int(row_size_bytes))
        for key in unique_keys:
            logical_key = f"emb:item:{key}"
            previous = self._embedding_last_access_epoch_by_item.get(key)
            interval = self._reuse_interval_ewma_by_key.get(logical_key)
            if previous is None:
                recency_probability = self._reuse_probability_ewma_by_type[
                    StateType.EMBEDDING_HOT_ROWS
                ]
            else:
                age = max(0, epoch - previous)
                horizon = max(1.0, interval or 8.0)
                recency_probability = math.exp(-float(age) / horizon)
            frequency_probability = 1.0 - math.exp(-float(counts[key]))
            reuse = min(1.0, 0.6 * recency_probability + 0.4 * frequency_probability)
            miss_cost = self.cost_model.embedding_miss_ms(max(1, counts[key]))
            movement_cost = self.cost_model.estimate_transfer_ms(
                state_class="embedding",
                direction="admission",
                bytes_moved=row_bytes,
            )
            opportunity_cost = self.cost_model.opportunity_cost_ms(row_bytes)
            net_benefit = reuse * miss_cost - movement_cost - opportunity_cost
            density = net_benefit / row_bytes
            ranked.append(
                (
                    density,
                    key,
                    {
                        "item_id": key,
                        "value_density_ms_per_byte": density,
                        "net_benefit_ms": net_benefit,
                        "reuse_probability": reuse,
                        "miss_cost_ms": miss_cost,
                        "movement_cost_ms": movement_cost,
                        "opportunity_cost_ms": opportunity_cost,
                        "count": counts[key],
                        "last_position": last_position.get(key, -1),
                        "model_confidence": self.cost_model.confidence,
                    },
                )
            )

        ranked.sort(key=lambda entry: (entry[0], entry[2]["net_benefit_ms"]), reverse=True)
        ranked_ids = [entry[1] for entry in ranked if entry[2]["net_benefit_ms"] > 0.0]
        if not return_trace:
            return ranked_ids
        return ranked_ids, [
            {**record, "rank": rank}
            for rank, (_density, _key, record) in enumerate(ranked)
        ]

    def record_embedding_accesses(self, item_sequence, epoch: int):
        for raw_key in item_sequence:
            key = int(raw_key)
            self.record_access(
                f"emb:item:{key}", epoch, StateType.EMBEDDING_HOT_ROWS
            )
            self._embedding_last_access_epoch_by_item[key] = int(epoch)
