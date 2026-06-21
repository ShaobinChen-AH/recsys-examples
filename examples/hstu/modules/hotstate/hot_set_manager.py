from dataclasses import dataclass, field
from typing import List

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

        self.registry.clear()
        for h in self.emb.export_handles():
            self.registry.register(h)
        for h in self.kv.export_handles():
            self.registry.register(h)

        self.value_engine.compute_scores(
            self.registry.snapshot(), demand)

        # Workload-aware budget: pages = ceil(tokens / page_size) * users + margin
        hist = demand.history_length
        tokens_per_user = hist * 2
        pages_per_user = math.ceil(tokens_per_user / self.kv.page_size_tokens)
        target_pages = (pages_per_user * NUM_USERS) + NUM_USERS * 2
        max_pages = self.kv._kvcache.num_primary_cache_pages
        target_pages = min(max(64, target_pages), max_pages)

        current_pages = self.kv.get_current_page_limit()
        if abs(target_pages - current_pages) > NUM_USERS:
            self.kv.set_page_limit(target_pages)

        return EpochResult(
            evicted_keys=[],
            admitted_keys=[],
            hbm_bytes_used=self.kv.total_hbm_bytes(),
            kv_page_budget=target_pages,
            epoch=epoch,
        ) 
