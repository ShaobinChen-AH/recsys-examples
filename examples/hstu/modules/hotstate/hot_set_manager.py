from dataclasses import dataclass, field
from typing import List

from modules.hotstate.state_handle import StateType, ScoredHandle, Placement
from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.state_registry import StateRegistry
from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.value_engine import ValueEngine
from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter


@dataclass
class EpochResult:
    evicted_keys: List[str] = field(default_factory=list)
    admitted_keys: List[str] = field(default_factory=list)
    hbm_bytes_used: int = 0
    kv_page_budget: int = 0
    epoch: int = 0


class HotSetManager:
    """Per-GPU hot-set manager.

    Each control epoch:
      1. Collect all StateHandles from adapters to register in registry
      2. Score all handles via ValueEngine
      3. Compare HBM occupancy to budget
      4. Evict lowest-scored HBM residents until under budget
      5. Admit highest-scored non-resident objects until budget full
      6. Respect in-flight transfers (GlobalDirectory)
    """

    def __init__(self, total_hbm_bytes: int,
                 value_engine: ValueEngine,
                 registry: StateRegistry,
                 directory: GlobalDirectory,
                 emb_adapter: EmbeddingAdapter,
                 kv_adapter: KVAdapter):
        self.total_hbm = total_hbm_bytes
        self.value_engine = value_engine
        self.registry = registry
        self.directory = directory
        self.emb = emb_adapter
        self.kv = kv_adapter

    def run_epoch(self, epoch: int, demand: DemandSignal) -> EpochResult:
        # 1. Collect fresh handles
        self.registry.clear()
        for h in self.emb.export_handles():
            self.registry.register(h)
        for h in self.kv.export_handles():
            self.registry.register(h)

        # 2. Score
        scored = self.value_engine.compute_scores(
            self.registry.snapshot(), demand)

        # 3. Current HBM state
        hbm_used = self.registry.hbm_footprint()

        # 4. Evict: remove lowest-scored HBM residents until under budget
        hbm_residents = [s for s in scored
                         if Placement.HBM in s.handle.placement]
        hbm_residents.sort(key=lambda s: s.score)  # ascending

        evicted = []
        for s in hbm_residents:
            if hbm_used <= self.total_hbm:
                break
            key = s.handle.logical_key
            # Skip pinned entries and in-flight transfers
            entry = self.directory.get(key)
            if entry and entry.pinned:
                continue
            if self.directory.is_in_flight(key):
                continue

            hbm_used -= s.handle.footprint_bytes
            self._execute_eviction(s.handle)
            self.directory.update_location(key, Placement.HOST_DRAM)
            self.directory.mark_complete(key)
            evicted.append(key)

        # 5. Admit: load highest-scored non-HBM objects if budget allows
        non_residents = [s for s in scored
                         if Placement.HBM not in s.handle.placement
                         and s.score > 0]
        non_residents.sort(key=lambda s: s.score, reverse=True)

        admitted = []
        for s in non_residents:
            if hbm_used >= self.total_hbm:
                break
            key = s.handle.logical_key
            if self.directory.is_in_flight(key):
                continue

            hbm_used += s.handle.footprint_bytes
            self._execute_admission(s.handle)
            self.directory.update_location(key, Placement.HBM)
            self.directory.mark_complete(key)
            admitted.append(key)

        # 6. Return budget recommendations
        return EpochResult(
            evicted_keys=evicted,
            admitted_keys=admitted,
            hbm_bytes_used=hbm_used,
            kv_page_budget=self.kv.get_current_page_limit(),
            epoch=epoch,
        )

    # Type-specific eviction / admission

    PAGE_DELTA = 64   # pages to adjust per eviction/admission action

    def _execute_eviction(self, h):
        key = h.logical_key
        if h.state_type == StateType.SESSION_KV_USER:
            uid = int(key.split(":")[-1])
            self.kv.evict_user(uid)
        elif h.state_type in (StateType.EMBEDDING_HOT_ROWS,
                               StateType.EMBEDDING_COLD_ROWS):
            # Suggest embedding subsystem to flush cache
            self.emb.trigger_flush()

    def _execute_admission(self, h):
        key = h.logical_key
        if h.state_type == StateType.SESSION_KV_USER:
            # Expanding KV budget makes pages available for this user
            new_limit = min(
                self.kv.get_current_page_limit() + self.PAGE_DELTA,
                self.kv._kvcache.num_primary_cache_pages)
            self.kv.set_page_limit(new_limit)
        # Embedding admissions are passive: rows load on next access
