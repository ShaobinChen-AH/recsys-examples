from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

from modules.hotstate.state_handle import StateType, ScoredHandle, Placement
from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.state_registry import StateRegistry
from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.value_engine import ValueEngine
import math

if TYPE_CHECKING:
    from modules.hotstate.embedding_adapter import EmbeddingAdapter
    from modules.hotstate.kv_adapter import KVAdapter

@dataclass
class EpochResult:
    evicted_keys: List[str] = field(default_factory=list)
    admitted_keys: List[str] = field(default_factory=list)
    kv_page_budget: int = 0
    epoch: int = 0
    scored_handles: List[ScoredHandle] = field(default_factory=list)
    decision_by_key: Dict[str, str] = field(default_factory=dict)
    selected_keys: List[str] = field(default_factory=list)
    planned_hbm_bytes: int = 0
    planned_kv_bytes: int = 0
    planned_embedding_bytes: int = 0
    planned_kv_page_budget: int = 0
    planned_embedding_keys: List[int] = field(default_factory=list)
    planned_embedding_eviction_keys: List[int] = field(default_factory=list)
    planned_kv_eviction_keys: List[str] = field(default_factory=list)

class HotSetManager:
    """Per-GPU hot-set manager.

    Each control epoch exports, scores, and selects handles under the policy
    budget, then emits concrete per-row/per-user actions.  The controller
    applies those actions through the storage adapters; this class never
    mutates physical allocations itself.
    """

    def __init__(self, total_hbm_bytes: int,
                 value_engine: ValueEngine,
                 registry: StateRegistry,
                 directory: GlobalDirectory,
                 emb_adapter: Any,
                 kv_adapter: Any,
                 skip_kv_handles_for_admission_smoke: bool = False):
        self.total_hbm = total_hbm_bytes
        self.value_engine = value_engine
        self.registry = registry
        self.directory = directory
        self.emb = emb_adapter
        self.kv = kv_adapter
        self.skip_kv_handles_for_admission_smoke = skip_kv_handles_for_admission_smoke

    def run_epoch(self, epoch: int, demand: DemandSignal) -> EpochResult:
        NUM_USERS = 8
        MIN_KV_PAGES = 64
        APPEND_MARGIN_PAGES = NUM_USERS * 2

        # Start a versioned observation epoch; the previous snapshot is kept
        # in registry history so lifecycle changes remain auditable.
        self.registry.begin_epoch(epoch, clear=True)

        handles = []
        for h in self.emb.export_handles():
            self.registry.register(h)
            self.directory.observe_handle(h, epoch=epoch)
            handles.append(h)
        if not self.skip_kv_handles_for_admission_smoke:
            for h in self.kv.export_handles():
                self.registry.register(h)
                self.directory.observe_handle(h, epoch=epoch)
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
                "keep_planned" if Placement.HBM in h.placement else "admit_planned"
            )

        selected_keys = {h.logical_key for h in selected}

        observe_allocation = getattr(self.value_engine, "observe_allocation", None)
        if callable(observe_allocation):
            observe_allocation(scored, selected, used_bytes)

        for s in scored:
            h = s.handle
            if h.logical_key in decision_by_key:
                continue
            decision_by_key[h.logical_key] = (
                "evict_candidate" if Placement.HBM in h.placement else "not_admitted"
            )

        if self.skip_kv_handles_for_admission_smoke:
            planned_kv_bytes = 0
            planned_embedding_bytes = sum(
                h.footprint_bytes for h in selected
                if h.state_type != StateType.SESSION_KV_USER
            )
            target_pages = self.kv.get_current_page_limit()
        else:
            page_bytes = self.kv._page_bytes()
            planned_kv_bytes = sum(
                h.footprint_bytes for h in selected
                if h.state_type == StateType.SESSION_KV_USER
            )
            planned_embedding_bytes = sum(
                h.footprint_bytes for h in selected
                if h.state_type != StateType.SESSION_KV_USER
            )

            planned_kv_pages = math.ceil(planned_kv_bytes / page_bytes) if planned_kv_bytes else 0
            resident_pages = self.kv.get_resident_page_count()
            max_pages = (
                self.kv.get_physical_page_count()
                if hasattr(self.kv, "get_physical_page_count")
                else self.kv._kvcache.num_primary_cache_pages
            )

            target_pages = planned_kv_pages + APPEND_MARGIN_PAGES
            target_pages = max(target_pages, resident_pages + APPEND_MARGIN_PAGES)

            hist = demand.history_length
            pages_per_user = math.ceil((hist * 2) / self.kv.page_size_tokens)
            old_safe_pages = pages_per_user * NUM_USERS + APPEND_MARGIN_PAGES
            old_safe_pages = min(max(MIN_KV_PAGES, old_safe_pages), max_pages)

            target_pages = max(target_pages, old_safe_pages)
            target_pages = min(max(MIN_KV_PAGES, target_pages), max_pages)

            # The fixed-envelope phase has no safe in-place KV resize API.
            # Keep this as a policy estimate only; changing the logical page
            # limit must never be presented as physical HBM enforcement.

        # Turn the aggregate score into concrete operations.  Embedding rows
        # are ranked from current demand and promoted into the already
        # allocated HybridStorage HBM tier.  KV decisions are explicit user
        # evictions; page-limit changes remain policy-only estimates.
        planned_embedding_keys: List[int] = []
        item_indices = list(getattr(demand, "item_indices", []) or [])
        if item_indices and planned_embedding_bytes > 0:
            try:
                row_size = max(1, int(self.emb.row_size_bytes()))
                target_keys = planned_embedding_bytes // row_size
                capacity_fn = getattr(self.emb, "hbm_capacity_keys", None)
                if callable(capacity_fn):
                    hbm_capacity = int(capacity_fn())
                    if hbm_capacity > 0:
                        target_keys = min(target_keys, hbm_capacity)
                if target_keys > 0:
                    rank_fn = getattr(self.value_engine, "rank_embedding_item_indices", None)
                    if callable(rank_fn):
                        planned_embedding_keys = list(
                            rank_fn(
                                item_indices=item_indices,
                                item_sequence=getattr(demand, "item_sequence", item_indices),
                                demand=demand,
                                row_size_bytes=row_size,
                                return_trace=False,
                            )[:target_keys]
                        )
                    else:
                        planned_embedding_keys = item_indices[:target_keys]
            except (RuntimeError, TypeError, ValueError):
                # Host-only/admission-smoke configurations have no physical
                # movement API; the adapter will report no applied action.
                planned_embedding_keys = []

        planned_kv_eviction_keys = []
        planned_embedding_eviction_keys: List[int] = []
        selected_key_set = set(selected_keys)
        for scored_handle in scored:
            handle = scored_handle.handle
            if (
                handle.state_type == StateType.SESSION_KV_USER
                and Placement.HBM in handle.placement
                and handle.logical_key not in selected_key_set
            ):
                planned_kv_eviction_keys.append(handle.logical_key)
            elif (
                handle.state_type in (
                    StateType.EMBEDDING_HOT_ROWS,
                    StateType.EMBEDDING_COLD_ROWS,
                )
                and Placement.HBM in handle.placement
                and handle.logical_key.startswith("emb:item:")
                and handle.logical_key not in selected_key_set
            ):
                try:
                    planned_embedding_eviction_keys.append(
                        int(handle.logical_key.rsplit(":", 1)[-1])
                    )
                except ValueError:
                    pass

        # The manager is a planner.  Keep applied-operation fields empty here;
        # HotStateController fills them only after adapter calls succeed.
        return EpochResult(
            evicted_keys=[],
            admitted_keys=[],
            kv_page_budget=self.kv.get_current_page_limit(),
            epoch=epoch,
            scored_handles=scored,
            decision_by_key=decision_by_key,
            selected_keys=sorted(selected_keys),
            planned_hbm_bytes=used_bytes,
            planned_kv_bytes=planned_kv_bytes,
            planned_embedding_bytes=planned_embedding_bytes,
            planned_kv_page_budget=target_pages,
            planned_embedding_keys=[int(key) for key in planned_embedding_keys],
            planned_embedding_eviction_keys=sorted(
                set(planned_embedding_eviction_keys)
            ),
            planned_kv_eviction_keys=planned_kv_eviction_keys,
        )
