from dataclasses import dataclass, field
from typing import Dict, List

from modules.hotstate.state_handle import StateType, ScoredHandle, Placement
from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.state_registry import StateRegistry
from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.value_engine import ValueEngine
from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter
import math

@dataclass
class EpochResult:
    evicted_keys: List[str] = field(default_factory=list)
    admitted_keys: List[str] = field(default_factory=list)
    hbm_bytes_used: int = 0
    kv_page_budget: int = 0
    epoch: int = 0
    scored_handles: List[ScoredHandle] = field(default_factory=list)
    decision_by_key: Dict[str, str] = field(default_factory=dict)
    scored_handles: List[ScoredHandle] = field(default_factory=list)
    decision_by_key: Dict[str, str] = field(default_factory=dict)
    selected_keys: List[str] = field(default_factory=list)
    selected_hbm_bytes: int = 0
    selected_kv_bytes: int = 0
    selected_embedding_bytes: int = 0

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
        NUM_USERS = 8
        MIN_KV_PAGES = 64
        APPEND_MARGIN_PAGES = NUM_USERS * 2

        self.registry.clear()

        handles = []
        for h in self.emb.export_handles():
            self.registry.register(h)
            handles.append(h)
        for h in self.kv.export_handles():
            self.registry.register(h)
            handles.append(h)

        scored = self.value_engine.compute_scores(handles, demand)

        selected = []
        decision_by_key = {}
        used_bytes = 0

        for s in sorted(
            scored,
            key=lambda x: (
                getattr(x, "value_density_ms_per_byte", x.score),
                getattr(x, "net_benefit_ms", 0.0),
            ),
            reverse=True,
        ):
            h = s.handle
            net_benefit = getattr(s, "net_benefit_ms", 0.0)
            if h.footprint_bytes <= 0 or net_benefit <= 0.0:
                continue
            if used_bytes + h.footprint_bytes > self.total_hbm:
                continue

            selected.append(h)
            used_bytes += h.footprint_bytes
            decision_by_key[h.logical_key] = (
                "keep_selected" if Placement.HBM in h.placement else "admit_planned"
            )

        selected_keys = {h.logical_key for h in selected}

        for s in scored:
            h = s.handle
            if h.logical_key in decision_by_key:
                continue
            decision_by_key[h.logical_key] = (
                "evict_candidate" if Placement.HBM in h.placement else "not_admitted"
            )

        page_bytes = self.kv._page_bytes()
        selected_kv_bytes = sum(
            h.footprint_bytes for h in selected
            if h.state_type == StateType.SESSION_KV_USER
        )
        selected_embedding_bytes = sum(
            h.footprint_bytes for h in selected
            if h.state_type != StateType.SESSION_KV_USER
        )

        selected_kv_pages = math.ceil(selected_kv_bytes / page_bytes) if selected_kv_bytes else 0
        resident_pages = self.kv.get_resident_page_count()
        max_pages = (
            self.kv.get_physical_page_count()
            if hasattr(self.kv, "get_physical_page_count")
            else self.kv._kvcache.num_primary_cache_pages
        )

        target_pages = selected_kv_pages + APPEND_MARGIN_PAGES
        target_pages = max(target_pages, resident_pages + APPEND_MARGIN_PAGES)

        hist = demand.history_length
        pages_per_user = math.ceil((hist * 2) / self.kv.page_size_tokens)
        old_safe_pages = pages_per_user * NUM_USERS + APPEND_MARGIN_PAGES
        old_safe_pages = min(max(MIN_KV_PAGES, old_safe_pages), max_pages)

        target_pages = max(target_pages, old_safe_pages)
        target_pages = min(max(MIN_KV_PAGES, target_pages), max_pages)

        current_pages = self.kv.get_current_page_limit()
        if abs(target_pages - current_pages) > NUM_USERS:
            self.kv.set_page_limit(target_pages)

        return EpochResult(
            evicted_keys=[],
            admitted_keys=[],
            hbm_bytes_used=self.kv.total_hbm_bytes(),
            kv_page_budget=target_pages,
            epoch=epoch,
            scored_handles=scored,
            decision_by_key=decision_by_key,
            selected_keys=sorted(selected_keys),
            selected_hbm_bytes=used_bytes,
            selected_kv_bytes=selected_kv_bytes,
            selected_embedding_bytes=selected_embedding_bytes,
        )
