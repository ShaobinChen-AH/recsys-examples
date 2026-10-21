from modules.hotstate.state_handle import Placement, StateType
from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.state_registry import StateRegistry
from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.value_engine import ValueEngine
from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter
from modules.hotstate.hot_set_manager import HotSetManager
from modules.hotstate.transfer_scheduler import TransferScheduler
from modules.hotstate.policy import (
    InferenceBudget,
    InferencePolicy,
    PolicyConfig,
    load_policy_config,
)
import json
import time
import os
from pathlib import Path
from typing import Optional

import torch


class HotStateController:
    """Unified HBM control plane for generative recommendation inference."""

    def __init__(
        self,
        total_hbm_bytes: int,
        kv_module,
        embedding_module=None,
        skip_kv_handles_for_admission_smoke: bool = False,
        admission_smoke_max_admitted_keys=None,
        configured_state_budget_bytes: Optional[int] = None,
        cuda_memory_before_construction: Optional[dict] = None,
        cuda_memory_after_construction: Optional[dict] = None,
        policy_config: Optional[PolicyConfig] = None,
        value_function=None,
        trace_path: Optional[str] = None,
    ):
        self.policy_config = policy_config or load_policy_config()
        configured_trace_path = trace_path or os.getenv("HOTSTATE_TRACE_PATH", "")
        self.trace_path = Path(configured_trace_path) if configured_trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_trace_record = None
        self.configured_state_budget_bytes = int(
            total_hbm_bytes
            if configured_state_budget_bytes is None
            else configured_state_budget_bytes
        )
        self._cuda_mem_before = (
            dict(cuda_memory_before_construction)
            if cuda_memory_before_construction is not None
            else self._cuda_memory_snapshot()
        )
        self.emb_adapter = EmbeddingAdapter(embedding_module)
        self.kv_adapter = KVAdapter(
            kv_module, granularity=self.policy_config.kv_granularity
        )
        self.registry = StateRegistry()
        self.directory = GlobalDirectory()
        self.value_engine = ValueEngine(
            self.configured_state_budget_bytes,
            config=self.policy_config,
            value_function=value_function,
        )
        self.hot_set = HotSetManager(
            self.configured_state_budget_bytes, self.value_engine,
            self.registry, self.directory,
            self.emb_adapter, self.kv_adapter, skip_kv_handles_for_admission_smoke=skip_kv_handles_for_admission_smoke,)
        self.hot_set.policy = InferencePolicy(self.policy_config)
        self.scheduler = TransferScheduler(
            self.kv_adapter,
            self.directory,
            self.value_engine,
            embedding_adapter=self.emb_adapter,
            registry=self.registry,
            bandwidth_fraction=self.policy_config.bandwidth_fraction,
        )
        self.epoch = 0

        # The scheduler now executes real DynamicEmb movement and stages the
        # native request-scoped KV onload consumed by model forward.
        self.enable_transfer_scheduler = True

        self.admission_smoke_max_admitted_keys = admission_smoke_max_admitted_keys

        self.trace_detail = "scalar"
        self.set_trace_detail(os.getenv("HOTSTATE_TRACE_DETAIL", "scalar"))

        self.num_users = 8
        self._protected_uids = []
        self.admission_batch_order_control = False
        self._pending_batch_cost_context = None

        # ================================================================
        # GAP 2: Budget Tracking and Enforcement (New)
        # ================================================================
        # Track current budget allocations
        self._current_embedding_budget = 0
        self._current_kv_budget = 0

        # Track budget enforcement metrics
        self._budget_enforcement_history = []  # Recent measurements
        self._enforcement_history_size = 20    # Keep last N measurements
        self._last_budget_check_batch = 0

        # Budget drift detection
        self._embedding_budget_violations = 0
        self._kv_budget_violations = 0
        self._total_budget_checks = 0

        # Reallocation frequency control
        self._default_reallocation_interval = 100
        self._adaptive_reallocation_interval = 100
        self._reallocation_interval = self._default_reallocation_interval

        # ================================================================
        # GAP 5: Semantic Differences and Consistency Policies (New)
        # ================================================================
        # Define semantic properties for each state type
        self._semantic_policies = {
            "embedding": {
                "state_type": "embedding",
                "consistency_class": "persistent_mutable",
                "description": "Persistent, potentially mutable embedding rows",
                "regenerable": True,
                "regeneration_cost_relative": 0.3,  # Relatively cheap to regenerate (0-1 scale)
                "session_bound": False,
                "authoritative_copy_required": False,
                "staleness_tolerance_epochs": 1,
                "eviction_preference": 0.7,  # Higher = more likely to evict
            },
            "kv_cache": {
                "state_type": "kv_cache",
                "consistency_class": "append_only_immutable",
                "description": "Ephemeral, session-bound KV cache blocks",
                "regenerable": True,
                "regeneration_cost_relative": 0.9,  # Very expensive to regenerate
                "session_bound": True,
                "authoritative_copy_required": True,
                "staleness_tolerance_epochs": 0,  # Must be current
                "eviction_preference": 0.2,  # Lower = less likely to evict
            },
        }

        # Track semantic policy adherence
        self._semantic_violations = 0
        self._semantic_decisions = 0

        # Calibrate adapter (read actual table sizes)
        self.emb_adapter.calibrate()

        # Initialize the directory from adapter observations.  This preserves
        # host/offloaded placements and all handle consistency metadata instead
        # of assuming every exported object is an HBM resident.
        self._sync_directory(epoch=0, include_registry=True)

        self._cuda_mem_after = (
            dict(cuda_memory_after_construction)
            if cuda_memory_after_construction is not None
            else self._cuda_memory_snapshot()
        )
        self._last_physical_validation = None

    def _sync_directory(self, *, epoch: Optional[int] = None, include_registry: bool = False):
        """Merge current adapter observations into registry and directory."""
        handles = []
        handles.extend(self.emb_adapter.export_handles())
        if not self.hot_set.skip_kv_handles_for_admission_smoke:
            handles.extend(self.kv_adapter.export_handles())
        if include_registry:
            self.registry.begin_epoch(0 if epoch is None else int(epoch), clear=True)
        for handle in handles:
            self.registry.upsert(handle)
            self.directory.observe_handle(handle, epoch=epoch)
        return handles

    def _score_state_handles(self, handles, demand=None):
        """Score state handles with the active demand or a neutral cold-start demand."""
        demand = demand or getattr(self, "_last_demand", None)
        if demand is None:
            demand = DemandSignal(
                current_user_id=0,
                history_length=0,
                num_candidates=0,
                epoch=self.epoch,
            )
        return self.value_engine.compute_scores(handles, demand)

    @staticmethod
    def _cuda_memory_snapshot():
        if not torch.cuda.is_available():
            return {"free_bytes": None, "total_bytes": None}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}

    @staticmethod
    def _host_memory_pressure() -> float:
        """Best-effort host DRAM pressure without adding a psutil dependency."""
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
            available = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
            if total > 0:
                return min(1.0, max(0.0, 1.0 - available / total))
        except (AttributeError, OSError, ValueError):
            pass
        return 0.0

    def physical_hbm_snapshot(self) -> dict:
        """Return measured state residency, separate from policy estimates."""
        embedding_bytes = int(self.emb_adapter.physical_hbm_bytes())
        kv_bytes = int(self.kv_adapter.physical_hbm_bytes())
        kv_breakdown = self.kv_adapter.physical_hbm_breakdown()
        kv_capacity_bytes = int(self.kv_adapter.physical_kv_cache_bytes())
        capacity_fn = getattr(self.emb_adapter, "hbm_value_capacity_bytes", None)
        embedding_capacity = (
            int(capacity_fn()) if callable(capacity_fn)
            else int(self.emb_adapter.hbm_capacity_keys()) * int(self.emb_adapter.row_size_bytes())
        )
        budget = InferenceBudget(
            state_bytes=self.configured_state_budget_bytes,
            embedding_capacity_bytes=embedding_capacity,
            kv_capacity_bytes=kv_capacity_bytes,
            physical_bytes=embedding_bytes + kv_bytes,
            fixed_overhead_bytes=max(
                0, embedding_bytes + kv_bytes - kv_capacity_bytes
                - embedding_capacity
            ),
            reserve_bytes=int(self.policy_config.reserve_bytes),
        )
        snapshot = {
            "configured_state_budget_bytes": self.configured_state_budget_bytes,
            "embedding_physical_hbm_bytes": embedding_bytes,
            "kv_physical_hbm_bytes": kv_bytes,
            "managed_state_physical_hbm_bytes": embedding_bytes + kv_bytes,
            "remaining_state_budget_bytes": self.configured_state_budget_bytes
            - embedding_bytes
            - kv_bytes,
            "logical_active_page_bytes": int(self.kv_adapter.logical_kv_budget_bytes()),
            "resident_page_bytes": int(self.kv_adapter.actual_resident_kv_bytes()),
            "resident_kv_pages": int(self.kv_adapter.get_resident_page_count()),
            "cuda_memory_before_construction": dict(self._cuda_mem_before),
            "cuda_memory_after_construction": dict(self._cuda_mem_after),
            # Compatibility names retained for existing JSON consumers.
            "cuda_memory_before_controller": dict(self._cuda_mem_before),
            "cuda_memory_after_controller": dict(self._cuda_mem_after),
            "cuda_memory_current": self._cuda_memory_snapshot(),
            "kv_physical_hbm_breakdown": kv_breakdown,
            "budget": budget.snapshot(),
            "policy": self.policy_config.snapshot(),
            "capacity": self.kv_adapter.capacity_snapshot(),
        }
        return snapshot

    def validate_physical_budget(self) -> dict:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        snapshot = self.physical_hbm_snapshot()
        if snapshot["managed_state_physical_hbm_bytes"] > self.configured_state_budget_bytes:
            raise RuntimeError(
                "managed physical HBM exceeds the configured state envelope: "
                f"embedding={snapshot['embedding_physical_hbm_bytes']} "
                f"kv={snapshot['kv_physical_hbm_bytes']} "
                f"budget={self.configured_state_budget_bytes}"
            )
        self._last_physical_validation = snapshot
        if not hasattr(self, "_initial_physical_snapshot"):
            self._initial_physical_snapshot = dict(snapshot)
        return snapshot

    def consistency_snapshot(self) -> dict:
        """Expose the enforceable inference state contract for traces."""
        entries = []
        for entry in self.directory.snapshot():
            entries.append({
                "logical_key": entry["logical_key"],
                "consistency_class": entry.get("metadata", {}).get(
                    "consistency_class", entry.get("metadata", {}).get("state_type")
                ),
                "version": int(entry["version"]),
                "placements": list(entry["placements"]),
                "in_flight": bool(entry["in_flight"]),
                "segments": entry.get("metadata", {}).get("segments"),
            })
        return {
            "contract": "immutable_embeddings_and_append_only_kv",
            "pending_transfers": int(self.directory.pending_count()),
            "entries": entries,
        }

    def set_trace_detail(self, trace_detail: str):
        self.trace_detail = trace_detail
    
    # Public API
    def set_embedding_module(self, embedding_module):
        """Connect the embedding adapter after both dense and sparse modules exist."""
        self.emb_adapter.set_module(embedding_module)

    def before_batch(self, batch, user_ids, total_history_lengths, batch_idx=0) -> dict:
        """Called before each inference batch. Runs control epoch + transfer planning."""
        if self._pending_batch_cost_context is not None:
            raise RuntimeError("previous inference request has not completed")
        t = time.perf_counter()

        uid = int(user_ids[0].item())
        total_history_tokens = int(total_history_lengths[0].item())
        hist_len = total_history_tokens // 2
        self.kv_adapter.record_batch_users(user_ids)
        if hasattr(user_ids, "detach"):
            protected_uids = [int(value) for value in user_ids.detach().cpu().tolist()]
        else:
            protected_uids = [int(value) for value in user_ids]
        self._protected_uids = list(protected_uids)
        if hasattr(total_history_lengths, "detach"):
            request_history_lengths = [
                int(value)
                for value in total_history_lengths.detach().cpu().tolist()
            ]
        else:
            request_history_lengths = [int(value) for value in total_history_lengths]

        item_sequence = self.emb_adapter.batch_item_sequence(batch)
        item_indices = list(dict.fromkeys(item_sequence))

        requested_unique_keys = len(item_indices)

        t1 = time.perf_counter()

        self.emb_adapter.record_batch_keys(item_indices)

        t2 = time.perf_counter()

        demand_kwargs = {
            "current_user_id": uid,
            "history_length": hist_len,
            "num_candidates": batch.max_num_candidates or 100,
            "epoch": self.epoch,
            "item_indices": item_indices,
            "user_ids": protected_uids,
            "history_lengths": request_history_lengths,
        }
        demand_fields = getattr(DemandSignal, "__dataclass_fields__", {})
        if "num_users" in demand_fields:
            demand_kwargs["num_users"] = getattr(self, "num_users", 8)
        if "item_sequence" in demand_fields:
            demand_kwargs["item_sequence"] = item_sequence

        demand = DemandSignal(**demand_kwargs)
        self._last_demand = demand

        # 1. Check completed offloads from previous epochs
        completed = []
        if self.enable_transfer_scheduler:
            completed = self.scheduler.poll_completions(self.epoch)

        # 2. Score and decide: what to keep, what to evict
        t_hotset_start = time.perf_counter()
        result = self.hot_set.run_epoch(self.epoch, demand)
        self._last_epoch_result = result
        hotset_ms = 1000 * (time.perf_counter() - t_hotset_start)

        planned_hbm_bytes = int(getattr(result, "planned_hbm_bytes", 0) or 0)
        planned_kv_bytes = int(getattr(result, "planned_kv_bytes", 0) or 0)
        planned_embedding_bytes = int(
            getattr(result, "planned_embedding_bytes", 0) or 0
        )

        kv_page_bytes = int(self.kv_adapter._page_bytes())
        kv_reserved_bytes = int(self.kv_adapter.physical_kv_cache_bytes())
        residual_embedding_budget_bytes = int(result.budget_snapshot["embedding_capacity_bytes"])
        embedding_budget_bytes = min(planned_embedding_bytes, residual_embedding_budget_bytes)

        row_size_bytes = max(1, self.emb_adapter.row_size_bytes())
        hotstate_admission_budget_keys = embedding_budget_bytes // row_size_bytes

        admit_all_control = bool(getattr(self, "admission_admit_all_control", False))
        batch_order_control = bool(getattr(self, "admission_batch_order_control", False))
        smoke_cap = getattr(self, "admission_smoke_max_admitted_keys", None)
        trace_detail = getattr(self, "trace_detail", "scalar")
        admission_trace = []

        admission_start = time.perf_counter()
        ranking_ms = 0.0
        if self.emb_adapter.supports_physical_residency_control():
            admission_item_indices = list(result.planned_embedding_keys)
            admission_max_keys = len(admission_item_indices)
            admission_cap_source = "unified_policy_selection"
            admission_order = "policy_score"
        elif admit_all_control:
            admission_max_keys = requested_unique_keys
            admission_item_indices = item_indices
            admission_cap_source = "admit_all_control"
            admission_order = "batch_order"
            if trace_detail == "full":
                admission_trace = [
                    {
                        "item_id": int(key),
                        "score": None,
                        "rank": rank,
                        "admitted": rank < admission_max_keys,
                    }
                    for rank, key in enumerate(admission_item_indices)
                ]
        else:
            admission_max_keys = hotstate_admission_budget_keys
            if smoke_cap is not None:
                admission_max_keys = min(admission_max_keys, max(0, int(smoke_cap)))
                admission_cap_source = "manual_smoke_clamp"
            else:
                admission_cap_source = "hotstate_residual_budget"

            if batch_order_control:
                admission_item_indices = item_indices
                admission_order = "batch_order"
                if trace_detail == "full":
                    admission_trace = [
                        {
                            "item_id": int(key),
                            "score": None,
                            "rank": rank,
                            "admitted": rank < admission_max_keys,
                        }
                        for rank, key in enumerate(admission_item_indices)
                    ]
            else:
                if hasattr(self.value_engine, "rank_embedding_item_indices"):
                    if trace_detail == "full":
                        rank_start = time.perf_counter()
                        admission_item_indices, admission_trace = self.value_engine.rank_embedding_item_indices(
                            item_indices=item_indices,
                            item_sequence=item_sequence,
                            demand=demand,
                            row_size_bytes=row_size_bytes,
                            return_trace=True,
                        )
                        ranking_ms = 1000 * (time.perf_counter() - rank_start)
                        admission_trace = [
                            {
                                **entry,
                                "admitted": int(entry["rank"]) < admission_max_keys,
                            }
                            for entry in admission_trace
                        ]
                    else:
                        rank_start = time.perf_counter()
                        admission_item_indices = self.value_engine.rank_embedding_item_indices(
                            item_indices=item_indices,
                            item_sequence=item_sequence,
                            demand=demand,
                            row_size_bytes=row_size_bytes,
                            return_trace=False,
                        )
                        ranking_ms = 1000 * (time.perf_counter() - rank_start)
                else:
                    admission_item_indices = item_indices
                    if trace_detail == "full":
                        admission_trace = [
                            {
                                "item_id": int(key),
                                "score": None,
                                "rank": rank,
                                "admitted": rank < admission_max_keys,
                            }
                            for rank, key in enumerate(admission_item_indices)
                        ]
                admission_order = "value_ranked"
       
        t_policy_start = time.perf_counter()
        self.emb_adapter.update_admission_policy(
            item_indices=admission_item_indices,
            max_admitted_keys=admission_max_keys,
            enabled=True,
        )
        policy_ms = 1000 * (time.perf_counter() - t_policy_start)

        if hasattr(self.value_engine, "record_embedding_accesses"):
            self.value_engine.record_embedding_accesses(item_sequence, self.epoch)
        else:
            for key in item_indices:
                self.value_engine.record_access(f"emb:item:{key}", self.epoch)

        policy_keys = admission_item_indices[:admission_max_keys]
        rejected_policy_keys = admission_item_indices[admission_max_keys:]
        admission_ms = 1000 * (time.perf_counter() - admission_start)

        t3 = time.perf_counter()

        # Apply the concrete hot-set plan through the dependency-aware
        # scheduler.  It orders real DynamicEmb copies and KV mapping releases,
        # owns directory/registry transitions, and reports only backend-
        # confirmed operations.
        planned_embedding_keys = list(getattr(result, "planned_embedding_keys", []))
        planned_embedding_evictions = list(
            getattr(result, "planned_embedding_eviction_keys", [])
        )
        planned_kv_evictions = list(getattr(result, "planned_kv_eviction_keys", []))
        scoring_map = {
            scored.handle.logical_key: scored for scored in result.scored_handles
        }
        transfer_execution = self.scheduler.execute_residency_plan(
            admitted_embedding_keys=planned_embedding_keys,
            evicted_embedding_keys=planned_embedding_evictions,
            evicted_kv_keys=planned_kv_evictions,
            scoring_map=scoring_map,
            protected_uids=protected_uids,
            epoch=self.epoch,
            row_size_bytes=row_size_bytes,
        )
        physical_admitted_embedding = transfer_execution.admitted_embedding_keys
        physical_evicted_embedding = transfer_execution.evicted_embedding_keys
        physical_evicted_kv = transfer_execution.released_kv_keys

        result.admitted_keys = [
            f"emb:item:{key}" for key in physical_admitted_embedding
        ]
        result.evicted_keys = [
            f"emb:item:{key}" for key in physical_evicted_embedding
        ] + physical_evicted_kv
        embedding_transfer_bytes = row_size_bytes * (
            len(physical_admitted_embedding) + len(physical_evicted_embedding)
        )
        # ``evict_if_present`` releases KV page mappings; it does not perform
        # the asynchronous host copy.  Do not mislabel its CPU duration as a
        # measured PCIe transfer. Native onload/offload pressure is learned in
        # ``after_batch`` from real page/token counters and inference latency.
        kv_released_bytes = 0
        for logical_key in physical_evicted_kv:
            handle = self.registry.get(logical_key)
            kv_released_bytes += (
                int(handle.footprint_bytes) if handle is not None else kv_page_bytes
            )
        # Refresh actual placements after the physical calls.  This also
        # captures HBM pressure displacement returned by DynamicEmb.admit_keys.
        self._sync_directory(epoch=self.epoch)

        # Correctness boundary: every row consumed by inference must already
        # exist in either DynamicEmb tier.  HotState cannot turn a placement
        # miss into a newly initialized model value.
        self.emb_adapter.validate_item_keys(item_indices)

        # Start the exact native KV prepare/onload that the upcoming model
        # forward will consume.  This is a real asynchronous operation using
        # the manager's existing worker and CUDA stream, not a page-limit hint.
        kv_stage = {}
        if self.enable_transfer_scheduler:
            kv_stage = self.scheduler.stage_kv_request(
                protected_uids,
                request_history_lengths,
                scoring_map=scoring_map,
                epoch=self.epoch,
            )

        t4 = time.perf_counter()

        # 4. Track demand access only.  Residency movement is not an access;
        # feeding admissions/evictions into reuse history would teach the
        # predictor its own decisions instead of the request stream.
        for active_uid in protected_uids:
            self.value_engine.record_access(f"kv:uid:{active_uid}", self.epoch)

        if self.epoch % 50 == 0:
            self.value_engine.decay_logs(self.epoch)

        self._pending_batch_cost_context = {
            "history_tokens": sum(request_history_lengths),
            "batch_idx": int(batch_idx),
            "seq_history_len": int(hist_len),
            "num_candidates": int(batch.max_num_candidates or 100),
            "embedding_requests": len(item_sequence),
            "embedding_request_sequence": list(item_sequence),
            "requested_embedding_keys": list(item_indices),
            "embedding_locations_for_forward": {
                int(key): (
                    set(self.directory.get(f"emb:item:{int(key)}").placements)
                    if self.directory.get(f"emb:item:{int(key)}") is not None
                    else set()
                )
                for key in item_indices
            },
            "user_id": uid,
            "user_was_offloaded": bool(self.kv_adapter.is_user_offloaded(uid)),
            # Controller residency changes are synchronous and finish before
            # the benchmark starts the inference timer.  Their timing trains
            # the transfer model above, but they are not concurrent work.
            "synchronous_control_transfer_bytes": int(
                embedding_transfer_bytes
            ),
            "synchronous_kv_released_bytes": int(kv_released_bytes),
            "staged_kv_generation": kv_stage.get("prepare_generation"),
            "staged_kv_onload_bytes": int(kv_stage.get("onload_bytes", 0)),
            "hbm_pressure": 0.0,
            "control_profile_ms": {
                "extract": 1000 * (t1 - t),
                "record_keys": 1000 * (t2 - t1),
                "run_epoch": hotset_ms,
                "scheduler": 1000 * (t4 - t3),
                "total": 1000 * (t4 - t),
            },
            "policy_inputs": (
                self._state_trace_records(result)
                if self.trace_path is not None or self.trace_detail == "full"
                else []
            ),
        }

        self.epoch += 1

        trace_detail = getattr(self, "trace_detail", "scalar")
        physical_hbm = self.physical_hbm_snapshot()
        if self._pending_batch_cost_context is not None:
            logical_resident_bytes = int(self.kv_adapter.actual_resident_kv_bytes())
            try:
                logical_resident_bytes += int(
                    self.emb_adapter.logical_hbm_resident_bytes()
                )
            except (RuntimeError, AttributeError):
                pass
            self._pending_batch_cost_context["hbm_pressure"] = min(
                1.0,
                max(
                    0.0,
                    float(logical_resident_bytes)
                    / float(max(1, self.configured_state_budget_bytes)),
                ),
            )

        return {
            "kv_page_budget": result.kv_page_budget,
            "active_kv_page_limit_pages": self.kv_adapter.get_current_page_limit(),
            "planned_kv_page_budget": result.planned_kv_page_budget,
            "hbm_bytes_used": physical_hbm["managed_state_physical_hbm_bytes"],
            "policy_planned_hbm_bytes": planned_hbm_bytes,
            "policy_planned_kv_bytes": planned_kv_bytes,
            "policy_planned_embedding_bytes": planned_embedding_bytes,
            "policy_budget": result.budget_snapshot,
            "physical_capacity_reallocation_supported": False,
            "evicted": len(result.evicted_keys),
            "admitted": len(result.admitted_keys),
            "epoch": self.epoch - 1,
            "completed_transfers": len(completed),
            "profile_ms": {
                "extract": 1000 * (t1 - t),
                "record_keys": 1000 * (t2 - t1),
                "run_epoch": 1000 * (t3 - t2),
                "scheduler": 1000 * (t4 - t3),
                "total": 1000 * (t4 - t),
            },
            "state_trace": (
                self._state_trace_records(result)
                if trace_detail == "full"
                else []
            ),
            "planned_hbm_bytes": planned_hbm_bytes,
            "planned_kv_bytes": planned_kv_bytes,
            "planned_embedding_bytes": planned_embedding_bytes,
            "embedding_requested_unique_keys": requested_unique_keys,
            "embedding_planned_budget_bytes": planned_embedding_bytes,
            "embedding_kv_reserved_bytes": kv_reserved_bytes,
            "embedding_residual_budget_bytes": residual_embedding_budget_bytes,
            "embedding_admission_budget_bytes": embedding_budget_bytes,
            "embedding_admission_budget_keys": hotstate_admission_budget_keys,
            "embedding_admission_policy_size": len(policy_keys),
            "embedding_admission_smoke_max_keys": smoke_cap,
            "embedding_admission_max_keys": admission_max_keys,
            "embedding_admission_cap_source": admission_cap_source,
            "embedding_admission_order": admission_order,
            "embedding_admission_trace": admission_trace if trace_detail == "full" else [],
            "embedding_requested_keys": item_indices if trace_detail == "full" else [],
            "embedding_planned_policy_keys": policy_keys if trace_detail == "full" else [],
            "embedding_rejected_policy_keys": rejected_policy_keys if trace_detail == "full" else [],
            "planned_embedding_action_keys": list(
                getattr(result, "planned_embedding_keys", [])
            )
            if trace_detail == "full"
            else [],
            "planned_kv_eviction_keys": list(
                getattr(result, "planned_kv_eviction_keys", [])
            )
            if trace_detail == "full"
            else [],
            "physical_admitted_keys": list(result.admitted_keys),
            "physical_evicted_keys": list(result.evicted_keys),
            "transfer_execution": transfer_execution.snapshot(),
            "staged_kv_request": dict(kv_stage),
            "transfer_scheduler": self.scheduler.snapshot(
                history_tail=32 if trace_detail == "full" else 0
            ),
            "physical_hbm": physical_hbm,
            "consistency": self.consistency_snapshot(),
            "online_cost_model": self.value_engine.cost_model_snapshot(),
            "policy_inputs": self._pending_batch_cost_context.get("policy_inputs", []),
            "physical_kv_pool_bytes": self.kv_adapter.physical_kv_cache_bytes(),
            "logical_kv_page_limit_pages": self.kv_adapter.get_current_page_limit(),
            "active_kv_page_limit_bytes": self.kv_adapter.logical_kv_budget_bytes(),
            "directory_pending_transfers": self.directory.pending_count(),
            "directory_state": (
                self.directory.snapshot() if trace_detail == "full" else []
            ),
            "directory_history_tail": (
                self.directory.history()[-32:] if trace_detail == "full" else []
            ),
            # ============================================================
            # CRITICAL: Forward path injection points
            # These are the actual decisions that must be consumed by model.forward
            # ============================================================
            "embedding_prefetch_keys": policy_keys,  # Keys to prefetch
            "kv_staged_handles": kv_stage,  # Staged KV prepare handles
            "admitted_keys": result.admitted_keys,  # For logging
            "evicted_keys": result.evicted_keys,  # For logging
        }

    def after_batch(self, batch, latency_ms: float):
        """Called after inference. Updates access statistics and snapshots post-forward state."""
        for feature_name in batch.features.keys():
            self.value_engine.record_access(f"emb:{feature_name}", self.epoch)

        context = self._pending_batch_cost_context
        scheduler_post = {}
        if context is not None:
            staged_generation = context.get("staged_kv_generation")
            completed_onloads = []
            if self.enable_transfer_scheduler and staged_generation is not None:
                completed_onloads = self.scheduler.complete_staged_kv_request(
                    int(staged_generation), self.epoch
                )
            if self.enable_transfer_scheduler:
                native_offload = self.scheduler.observe_native_kv_activity(self.epoch)
                completed_native = self.scheduler.poll_completions(self.epoch)
                scheduler_post = {
                    "completed_onloads": completed_onloads,
                    "native_offload": native_offload,
                    "completed_native": completed_native,
                    "snapshot": self.scheduler.snapshot(
                        history_tail=32 if self.trace_detail == "full" else 0
                    ),
                }
            embedding_misses = sum(
                1
                for item_id in context["embedding_request_sequence"]
                if Placement.HBM not in context[
                    "embedding_locations_for_forward"
                ].get(int(item_id), set())
            )
            async_kvcache = self.kv_adapter._kvcache
            new_tokens = getattr(async_kvcache, "last_new_tokens", None)
            origin_lengths = getattr(
                async_kvcache, "last_origin_cached_lengths", None
            )
            origin = sum(int(length) for length in origin_lengths) if origin_lengths else 0
            if new_tokens is None:
                new_tokens = max(0, int(context["history_tokens"]) - origin)
            offload_pages = int(
                getattr(async_kvcache, "last_num_offload_pages", 0) or 0
            )
            automatic_transfer_bytes = offload_pages * self.kv_adapter._page_bytes()
            automatic_transfer_bytes += int(
                context.get("staged_kv_onload_bytes", 0) or 0
            )
            automatic_queue_depth = self.scheduler.pending_count()
            self.value_engine.observe_batch_cost(
                latency_ms=float(latency_ms),
                history_tokens=int(context["history_tokens"]),
                num_candidates=int(context["num_candidates"]),
                embedding_requests=int(context["embedding_requests"]),
                embedding_misses=int(embedding_misses),
                kv_miss_tokens=max(0, int(new_tokens)),
                concurrent_transfer_bytes=int(automatic_transfer_bytes),
                transfer_queue_depth=automatic_queue_depth,
                hbm_pressure=float(context["hbm_pressure"]),
                host_memory_pressure=self._host_memory_pressure(),
            )
            self._pending_batch_cost_context = None

        # The forward may have caused native KV onload/offload activity and
        # DynamicEmb lookups may have changed tier membership.  Observe both
        # adapters after the kernels complete, even for scalar traces.
        self._sync_directory(epoch=self.epoch)

        if self.trace_detail != "full":
            result = {
                "post_num_state_handles": 0,
                "post_state_trace": [],
                "post_directory_pending_transfers": self.directory.pending_count(),
                "consistency": self.consistency_snapshot(),
                "online_cost_model": self.value_engine.cost_model_snapshot(),
                "transfer_scheduler": scheduler_post,
            }
            return result

        demand = getattr(self, "_last_demand", None)
        post_state_trace = []

        if demand is not None:
            handles = []
            handles.extend(self.emb_adapter.export_handles())
            handles.extend(self.kv_adapter.export_handles())

            scored_handles = self.value_engine.compute_scores(handles, demand)
            decision_by_key = {}
            for scored in scored_handles:
                handle = scored.handle
                decision_by_key[handle.logical_key] = (
                    "resident" if Placement.HBM in handle.placement else "not_resident"
                )

            post_state_trace = self._state_trace_records_from_scored(
                scored_handles,
                decision_by_key,
            )

        result = {
            "post_num_state_handles": len(post_state_trace),
            "post_state_trace": post_state_trace,
            "post_directory_state": self.directory.snapshot(),
            "consistency": self.consistency_snapshot(),
            "online_cost_model": self.value_engine.cost_model_snapshot(),
            "transfer_scheduler": scheduler_post,
        }
        return result

    def record_inference_trace(self, record: dict) -> None:
        """Send a completed measured request to the configured JSONL sink."""
        self.last_trace_record = record
        if self.trace_path is not None:
            with self.trace_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def abort_batch(self, error: str) -> None:
        """Drain native owners and release HotState request ownership on failure."""
        try:
            abort = getattr(self.kv_adapter._kvcache, "abort_active_prepare", None)
            if callable(abort):
                abort()
        finally:
            for generation in list(self.scheduler._users_by_generation):
                self.scheduler.abort_staged_kv_request(generation, self.epoch, error)
            self._pending_batch_cost_context = None

    def abort_staged_kv_request(self, generation: int, error: str) -> None:
        """Unwind request ownership when inference raises before after_batch."""
        self.scheduler.abort_staged_kv_request(
            int(generation), self.epoch, str(error)
        )
   
    def _state_trace_records(self, result):
        return self._state_trace_records_from_scored(
            result.scored_handles,
            result.decision_by_key,
        )
    
    def _state_trace_records_from_scored(self, scored_handles, decision_by_key):
        records = []
        for scored in scored_handles:
            handle = scored.handle
            records.append({
                "logical_key": handle.logical_key,
                "state_type": handle.state_type.name,
                "footprint_bytes": int(handle.footprint_bytes),
                "placement": sorted(p.name for p in handle.placement),
                "authoritative_placement": (
                    handle.authoritative_placement.name
                    if handle.authoritative_placement is not None else None
                ),
                "replica_versions": {
                    placement.name: int(version)
                    for placement, version in handle.replica_versions.items()
                },
                "replica_freshness_epochs": {
                    placement.name: int(epoch)
                    for placement, epoch in handle.replica_freshness_epochs.items()
                },
                "version": int(handle.version),
                "freshness_epoch": int(handle.freshness_epoch),
                "owner_device": handle.owner_device,
                "transfer_state": handle.transfer_state.name,
                "transfer_id": handle.transfer_id,
                "writeback_required": bool(handle.writeback_required),
                "writeback_pending": bool(handle.writeback_pending),
                "dependencies": list(handle.dependencies),
                "last_error": handle.last_error,
                "recovery_source": (
                    handle.recovery_source.name
                    if handle.recovery_source is not None else None
                ),
                "metadata": dict(handle.metadata),
                "reconstructability": handle.reconstructability.name,
                "consistency_class": handle.consistency_class,
                "reuse_imminence": float(handle.reuse_imminence),
                "stall_sensitivity_ms": float(handle.stall_sensitivity_ms),
                "movement_cost_ms": float(handle.movement_cost_ms),
                "score": float(scored.score),
                "benefit_density": float(scored.benefit_density),
                "occupancy_penalty": float(scored.occupancy_penalty),
                "semantic_risk": float(scored.semantic_risk),
                "reuse_probability": float(
                    getattr(scored, "reuse_probability", handle.reuse_imminence)
                ),
                "miss_cost_ms": float(getattr(scored, "miss_cost_ms", 0.0)),
                "gross_benefit_ms": float(getattr(scored, "gross_benefit_ms", 0.0)),
                "movement_penalty_ms": float(
                    getattr(scored, "movement_cost_ms", handle.movement_cost_ms)
                ),
                "semantic_risk_ms": float(
                    getattr(scored, "risk_cost_ms", scored.semantic_risk)
                ),
                "opportunity_cost_ms": float(
                    getattr(scored, "opportunity_cost_ms", 0.0)
                ),
                "net_benefit_ms": float(getattr(scored, "net_benefit_ms", 0.0)),
                "value_density_ms_per_byte": float(
                    getattr(scored, "value_density_ms_per_byte", 0.0)
                ),
                "decision_reason": getattr(scored, "decision_reason", ""),
                "decision": decision_by_key.get(handle.logical_key, "unknown"),
                "policy_name": getattr(scored, "policy_name", ""),
                "policy_config_id": getattr(scored, "policy_config_id", ""),
                "policy_inputs": dict(getattr(scored, "policy_inputs", {})),
            })
        return records

    # ============================================================================
    # Dynamic Allocation API: Runtime HBM Arbitration for Gap 1 Closure
    # ============================================================================

    def get_current_occupancy(self) -> dict:
        """Query current HBM occupancy of embedding and KV states.

        Returns dict with:
        - embedding_physical_hbm_bytes: actual embedding HBM usage
        - kv_physical_hbm_bytes: actual KV cache HBM usage
        - embedding_utilization_ratio: occupancy / budget
        - kv_utilization_ratio: occupancy / budget
        """
        embedding_bytes = int(self.emb_adapter.physical_hbm_bytes())
        kv_bytes = int(self.kv_adapter.physical_hbm_bytes())

        baseline = getattr(self, "_initial_physical_snapshot", {})
        embedding_budget = int(baseline.get("embedding_physical_hbm_bytes", embedding_bytes))
        kv_budget = int(baseline.get("kv_physical_hbm_bytes", kv_bytes))

        embedding_util = embedding_bytes / embedding_budget if embedding_budget > 0 else 0.0
        kv_util = kv_bytes / kv_budget if kv_budget > 0 else 0.0

        return {
            "embedding_physical_hbm_bytes": embedding_bytes,
            "kv_physical_hbm_bytes": kv_bytes,
            "embedding_budget_bytes": embedding_budget,
            "kv_budget_bytes": kv_budget,
            "embedding_utilization_ratio": min(1.0, max(0.0, embedding_util)),
            "kv_utilization_ratio": min(1.0, max(0.0, kv_util)),
            "total_configured_bytes": self.configured_state_budget_bytes,
        }

    def compute_value_densities(self) -> dict:
        """Compute current value densities for embedding and KV states.

        Returns dict with:
        - embedding_value_density_ms_per_byte: benefit per byte for embeddings
        - kv_value_density_ms_per_byte: benefit per byte for KV
        - embedding_is_bottleneck: True if embedding has high value density
        - kv_is_bottleneck: True if KV has high value density
        """
        # Query value engine for current scoring
        handles = self._sync_directory(epoch=self.epoch, include_registry=False)

        scored_by_key = {
            scored.handle.logical_key: scored
            for scored in self._score_state_handles(handles)
        }
        embedding_handles = [
            handle
            for handle in handles
            if handle.state_type
            in (StateType.EMBEDDING_HOT_ROWS, StateType.EMBEDDING_COLD_ROWS)
        ]
        kv_handles = [
            handle
            for handle in handles
            if handle.is_kv
        ]

        # Aggregate value densities
        emb_total_density = 0.0
        emb_total_bytes = 0
        for handle in embedding_handles:
            scored = scored_by_key.get(handle.logical_key)
            if scored is None:
                continue
            density = float(getattr(scored, 'value_density_ms_per_byte', 0.0))
            footprint = int(handle.footprint_bytes)
            emb_total_density += density * footprint
            emb_total_bytes += footprint

        emb_avg_density = emb_total_density / emb_total_bytes if emb_total_bytes > 0 else 0.0

        kv_total_density = 0.0
        kv_total_bytes = 0
        for handle in kv_handles:
            scored = scored_by_key.get(handle.logical_key)
            if scored is None:
                continue
            density = float(getattr(scored, 'value_density_ms_per_byte', 0.0))
            footprint = int(handle.footprint_bytes)
            kv_total_density += density * footprint
            kv_total_bytes += footprint

        kv_avg_density = kv_total_density / kv_total_bytes if kv_total_bytes > 0 else 0.0

        return {
            "embedding_value_density_ms_per_byte": emb_avg_density,
            "kv_value_density_ms_per_byte": kv_avg_density,
            "embedding_total_bytes": emb_total_bytes,
            "kv_total_bytes": kv_total_bytes,
            "embedding_is_bottleneck": emb_avg_density > kv_avg_density,
            "kv_is_bottleneck": kv_avg_density > emb_avg_density,
        }

    def compute_adaptive_allocation(self) -> dict:
        """Return the latest policy recommendation without changing capacity."""
        result = getattr(self, "_last_epoch_result", None)
        return {
            "recommended_embedding_hbm_bytes": result.planned_embedding_bytes if result else 0,
            "recommended_kv_hbm_bytes": result.planned_kv_bytes if result else 0,
            "reallocation_triggered": False,
            "reallocation_ratio": 0.0,
            "decision_reason": "residency_plan_within_existing_pools",
            "physical_capacity_reallocation_supported": False,
            "allocation_semantics": "policy_recommendation_only",
            "policy": self.policy_config.snapshot(),
        }

    def enforce_adaptive_allocation(self, allocation_plan: dict) -> bool:
        """Reject physical resize requests; before_batch executes residency plans.

        Changing a KV active limit or admission list cannot resize a pool.
        This compatibility method never reports such changes as enforcement.
        """
        allocation_plan["physical_capacity_reallocation_applied"] = False
        allocation_plan["physical_capacity_reallocation_supported"] = False
        allocation_plan["allocation_semantics"] = "unsupported_physical_resize"
        return False

    def detect_budget_drift(self) -> dict:
        """Detect when actual occupancy deviates from allocated budgets.

        Returns dict with:
        - embedding_drift_ratio: (actual - budget) / budget (< 0 = under, > 0 = over)
        - kv_drift_ratio: (actual - budget) / budget
        - has_embedding_violation: actual > budget * 1.1 (10% overshoot tolerance)
        - has_kv_violation: actual > budget * 1.1
        - recommendation: "reduce_embedding", "reduce_kv", "rebalance", or "ok"
        """
        occupancy = self.get_current_occupancy()

        emb_actual = occupancy["embedding_physical_hbm_bytes"]
        kv_actual = occupancy["kv_physical_hbm_bytes"]
        emb_budget = occupancy["embedding_budget_bytes"]
        kv_budget = occupancy["kv_budget_bytes"]

        # Calculate drift ratios (positive = over budget, negative = under budget)
        emb_drift = (emb_actual - emb_budget) / emb_budget if emb_budget > 0 else 0.0
        kv_drift = (kv_actual - kv_budget) / kv_budget if kv_budget > 0 else 0.0

        # Check for violations (10% overshoot tolerance)
        emb_violation = emb_actual > emb_budget * 1.1
        kv_violation = kv_actual > kv_budget * 1.1

        # Record violations
        if emb_violation:
            self._embedding_budget_violations += 1
        if kv_violation:
            self._kv_budget_violations += 1
        self._total_budget_checks += 1

        # Determine recommendation
        recommendation = "ok"
        if emb_violation and kv_drift < -0.3:
            recommendation = "reduce_embedding_expand_kv"
        elif kv_violation and emb_drift < -0.3:
            recommendation = "reduce_kv_expand_embedding"
        elif emb_violation:
            recommendation = "reduce_embedding_global"
        elif kv_violation:
            recommendation = "reduce_kv_global"

        return {
            "embedding_drift_ratio": emb_drift,
            "kv_drift_ratio": kv_drift,
            "embedding_actual_bytes": emb_actual,
            "embedding_budget_bytes": emb_budget,
            "kv_actual_bytes": kv_actual,
            "kv_budget_bytes": kv_budget,
            "has_embedding_violation": emb_violation,
            "has_kv_violation": kv_violation,
            "recommendation": recommendation,
            "violation_rate": (self._embedding_budget_violations + self._kv_budget_violations) / max(1, self._total_budget_checks),
        }

    def validate_budget_enforcement(self) -> dict:
        """Check if allocated budgets are actually being respected by the system.

        Returns dict with enforcement quality metrics.
        """
        drift = self.detect_budget_drift()

        # Track enforcement history
        self._budget_enforcement_history.append(drift)
        if len(self._budget_enforcement_history) > self._enforcement_history_size:
            self._budget_enforcement_history.pop(0)

        # Calculate enforcement effectiveness
        recent_violations = sum(
            1 for d in self._budget_enforcement_history
            if d["has_embedding_violation"] or d["has_kv_violation"]
        )
        recent_count = len(self._budget_enforcement_history)
        enforcement_success_rate = 1.0 - (recent_violations / recent_count if recent_count > 0 else 0.0)

        return {
            "current_drift": drift,
            "enforcement_success_rate": enforcement_success_rate,
            "recent_violations": recent_violations,
            "recent_checks": recent_count,
            "total_violations": self._embedding_budget_violations + self._kv_budget_violations,
            "total_checks": self._total_budget_checks,
        }

    def should_trigger_emergency_reallocation(self) -> bool:
        """Check if budget violations warrant immediate reallocation.

        Returns True if:
        - Current check violates budget
        - Recent violation rate > 20%
        """
        drift = self.detect_budget_drift()

        # Emergency trigger: active violation
        if drift["has_embedding_violation"] or drift["has_kv_violation"]:
            return True

        # Emergency trigger: high recent violation rate
        if drift["violation_rate"] > 0.2:
            return True

        return False

    def update_reallocation_frequency(self) -> None:
        """Adapt reallocation check frequency based on budget enforcement success.

        - If enforcement is good (> 90% success): check every 100 batches
        - If enforcement is poor (< 50% success): check every 20 batches
        """
        validation = self.validate_budget_enforcement()
        success_rate = validation["enforcement_success_rate"]

        if success_rate > 0.9:
            self._reallocation_interval = 100
        elif success_rate > 0.7:
            self._reallocation_interval = 50
        elif success_rate > 0.5:
            self._reallocation_interval = 25
        else:
            # Poor enforcement: check very frequently
            self._reallocation_interval = 10

    def compute_corrective_allocation(self, protected_uids=None) -> dict:
        """Validate the physical envelope; report the current residency plan."""
        self.validate_physical_budget()
        return self.compute_adaptive_allocation()

    def report_budget_enforcement_summary(self) -> dict:
        """Generate end-of-inference report on budget enforcement effectiveness.

        Returns summary of how well budgets were maintained throughout inference.
        """
        validation = self.validate_budget_enforcement()
        drift = validation["current_drift"]

        total_violations = validation["total_violations"]
        total_checks = validation["total_checks"]
        violation_rate = total_violations / max(1, total_checks)

        summary = {
            "total_budget_checks": total_checks,
            "total_violations": total_violations,
            "violation_rate": violation_rate,
            "enforcement_success_rate": validation["enforcement_success_rate"],
            "embedding_violations": self._embedding_budget_violations,
            "kv_violations": self._kv_budget_violations,
            "final_embedding_drift": drift["embedding_drift_ratio"],
            "final_kv_drift": drift["kv_drift_ratio"],
            "final_embedding_actual_bytes": drift["embedding_actual_bytes"],
            "final_embedding_budget_bytes": drift["embedding_budget_bytes"],
            "final_kv_actual_bytes": drift["kv_actual_bytes"],
            "final_kv_budget_bytes": drift["kv_budget_bytes"],
            "status": "PASS" if violation_rate < 0.1 else "WARN" if violation_rate < 0.3 else "FAIL",
        }

        print(
            f"\n[HotState Budget Enforcement Summary]\n"
            f"  Checks: {total_checks}\n"
            f"  Violations: {total_violations} ({violation_rate:.1%})\n"
            f"  Success Rate: {validation['enforcement_success_rate']:.1%}\n"
            f"  Final Embedding: {drift['embedding_actual_bytes']/(1024**3):.3f}/{drift['embedding_budget_bytes']/(1024**3):.3f} GiB (Drift: {drift['embedding_drift_ratio']:+.1%})\n"
            f"  Final KV: {drift['kv_actual_bytes']/(1024**3):.3f}/{drift['kv_budget_bytes']/(1024**3):.3f} GiB (Drift: {drift['kv_drift_ratio']:+.1%})\n"
            f"  Status: {summary['status']}\n"
        )

        return summary

    # ============================================================================
    # GAP 3: Transfer Scheduler Telemetry and Coordination
    # ============================================================================

    def get_scheduler_state(self) -> dict:
        """Get current state of the transfer scheduler.

        Returns dict with scheduler activity:
        - pending_transfers: count of queued/in-flight transfers
        - embedding_transfers: transfers for embedding state
        - kv_transfers: transfers for KV state
        - resource_utilization: PCIe/bandwidth usage
        """
        try:
            snapshot = self.scheduler.snapshot()
            return {
                "scheduler_active": True,
                "queued_transfers": snapshot.get("queued_count", 0),
                "in_flight_transfers": snapshot.get("in_flight_count", 0),
                "pending_bytes": snapshot.get("pending_bytes", 0),
                "bandwidth_budget_bytes": snapshot.get("epoch_bandwidth_budget_bytes", 0),
                "bandwidth_fraction": snapshot.get("bandwidth_fraction", 0.0),
                "resources": snapshot.get("resources", {}),
                "counters": snapshot.get("counters", {}),
            }
        except Exception as e:
            return {"scheduler_active": False, "error": str(e)}

    def get_transfer_coordination_metrics(self) -> dict:
        """Analyze how well embedding and KV transfers are coordinated.

        Returns metrics on transfer coordination:
        - embedding_transfer_count: embeddings currently being transferred
        - kv_transfer_count: KV currently being transferred
        - overlapped_transfers: simultaneous embedding+KV transfers (good coordination)
        - coordination_ratio: percentage of transfers that overlap
        """
        try:
            snapshot = self.scheduler.snapshot()
            queued = snapshot.get("queued", [])
            in_flight = snapshot.get("in_flight", [])

            all_transfers = queued + in_flight

            # Count by state class
            embedding_transfers = [t for t in all_transfers if "emb" in t.get("logical_key", "").lower()]
            kv_transfers = [t for t in all_transfers if "kv" in t.get("logical_key", "").lower()]

            # Calculate metrics
            total_transfers = len(all_transfers)
            in_flight_count = len(in_flight)

            # Coordination: if both embedding and KV are in-flight simultaneously
            embedding_in_flight = sum(1 for t in in_flight if "emb" in t.get("logical_key", "").lower())
            kv_in_flight = sum(1 for t in in_flight if "kv" in t.get("logical_key", "").lower())
            overlapped_transfers = min(embedding_in_flight, kv_in_flight)

            coordination_ratio = (overlapped_transfers / max(1, in_flight_count)) if in_flight_count > 0 else 0.0

            return {
                "total_transfers": total_transfers,
                "embedding_transfers": len(embedding_transfers),
                "kv_transfers": len(kv_transfers),
                "in_flight_transfers": in_flight_count,
                "embedding_in_flight": embedding_in_flight,
                "kv_in_flight": kv_in_flight,
                "overlapped_transfers": overlapped_transfers,
                "coordination_ratio": coordination_ratio,
                "transfer_bytes": sum(t.get("bytes", 0) for t in all_transfers),
            }
        except Exception as e:
            return {"coordination_error": str(e)}

    def get_value_scoring_metrics(self) -> dict:
        """Get metrics on unified value scoring across embedding and KV.

        Returns statistics on how value engine scores different state types.
        """
        try:
            handles = self._sync_directory(epoch=self.epoch, include_registry=False)
            scored_by_key = {
                scored.handle.logical_key: scored
                for scored in self._score_state_handles(handles)
            }

            embedding_handles = [
                handle
                for handle in handles
                if handle.state_type
                in (StateType.EMBEDDING_HOT_ROWS, StateType.EMBEDDING_COLD_ROWS)
            ]
            kv_handles = [
                handle
                for handle in handles
                if handle.is_kv
            ]

            # Score each handle
            emb_scores = []
            for handle in embedding_handles:
                try:
                    scored = scored_by_key.get(handle.logical_key)
                    if scored is None:
                        continue
                    emb_scores.append(float(getattr(scored, 'value_density_ms_per_byte', 0.0)))
                except:
                    pass

            kv_scores = []
            for handle in kv_handles:
                try:
                    scored = scored_by_key.get(handle.logical_key)
                    if scored is None:
                        continue
                    kv_scores.append(float(getattr(scored, 'value_density_ms_per_byte', 0.0)))
                except:
                    pass

            # Calculate statistics
            emb_avg = sum(emb_scores) / len(emb_scores) if emb_scores else 0.0
            kv_avg = sum(kv_scores) / len(kv_scores) if kv_scores else 0.0

            return {
                "embedding_handles_scored": len(emb_scores),
                "kv_handles_scored": len(kv_scores),
                "embedding_avg_value_density": emb_avg,
                "kv_avg_value_density": kv_avg,
                "unified_scoring_active": True,
                "score_comparison": {
                    "embedding_higher": emb_avg > kv_avg,
                    "score_ratio": (kv_avg + 1e-6) / (emb_avg + 1e-6),
                },
            }
        except Exception as e:
            return {"scoring_error": str(e)}

    def report_scheduler_summary(self) -> dict:
        """Generate end-of-inference report on transfer scheduler activity.

        Returns summary of how scheduler coordinated state transfers.
        """
        scheduler_state = self.get_scheduler_state()
        coordination = self.get_transfer_coordination_metrics()
        value_scoring = self.get_value_scoring_metrics()

        summary = {
            "scheduler_active": scheduler_state.get("scheduler_active", False),
            "total_queued_transfers": scheduler_state.get("queued_transfers", 0),
            "total_in_flight_transfers": scheduler_state.get("in_flight_transfers", 0),
            "total_transfer_bytes": scheduler_state.get("pending_bytes", 0),
            "bandwidth_fraction": scheduler_state.get("bandwidth_fraction", 0.0),
            "coordination_ratio": coordination.get("coordination_ratio", 0.0),
            "overlapped_transfers": coordination.get("overlapped_transfers", 0),
            "unified_value_scoring": value_scoring.get("unified_scoring_active", False),
            "embedding_kv_value_ratio": value_scoring.get("score_comparison", {}).get("score_ratio", 1.0),
        }

        print(
            f"\n[HotState Transfer Scheduler Summary]\n"
            f"  Active: {scheduler_state.get('scheduler_active', False)}\n"
            f"  Queued Transfers: {scheduler_state.get('queued_transfers', 0)}\n"
            f"  In-Flight Transfers: {scheduler_state.get('in_flight_transfers', 0)}\n"
            f"  Total Transfer Bytes: {scheduler_state.get('pending_bytes', 0)/(1024**2):.1f} MiB\n"
            f"  Coordination Ratio: {coordination.get('coordination_ratio', 0.0):.1%} "
            f"(embedding+KV overlap)\n"
            f"  Unified Value Scoring: {value_scoring.get('unified_scoring_active', False)}\n"
            f"  Embedding Value Density: {value_scoring.get('embedding_avg_value_density', 0.0):.4f} ms/byte\n"
            f"  KV Value Density: {value_scoring.get('kv_avg_value_density', 0.0):.4f} ms/byte\n"
        )

        return summary

    # ============================================================================
    # GAP 4: Cross-State Value Comparison and Unified Eviction Oracle
    # ============================================================================

    def compute_normalized_value_densities(self) -> dict:
        """Compute value densities normalized to enable cross-state comparison.

        Handles embedding rows and KV pages have different sizes, so we normalize
        by computing: value_per_byte that can be directly compared.

        Returns dict with normalized metrics for each state type.
        """
        occupancy = self.get_current_occupancy()
        densities = self.compute_value_densities()

        emb_density = densities.get("embedding_value_density_ms_per_byte", 0.0)
        kv_density = densities.get("kv_value_density_ms_per_byte", 0.0)

        # Already normalized to per-byte, but add context
        return {
            "embedding_value_density_ms_per_byte": emb_density,
            "kv_value_density_ms_per_byte": kv_density,
            "embedding_total_value": densities.get("embedding_total_bytes", 0) * emb_density,
            "kv_total_value": densities.get("kv_total_bytes", 0) * kv_density,
            "embedding_is_more_valuable": emb_density > kv_density,
            "kv_is_more_valuable": kv_density > emb_density,
            "value_density_ratio": (kv_density + 1e-6) / (emb_density + 1e-6),
        }

    def build_unified_eviction_candidates(self) -> dict:
        """Build unified pool of eviction candidates across embedding and KV.

        Returns dict with ranked candidates for eviction ordered by value density
        (lowest value = highest priority for eviction).
        """
        try:
            handles = self._sync_directory(epoch=self.epoch, include_registry=False)
            scored_by_key = {
                scored.handle.logical_key: scored
                for scored in self._score_state_handles(handles)
            }

            # Separate handles by type
            embedding_handles = [
                handle
                for handle in handles
                if handle.state_type
                in (StateType.EMBEDDING_HOT_ROWS, StateType.EMBEDDING_COLD_ROWS)
            ]
            kv_handles = [
                handle
                for handle in handles
                if handle.is_kv
            ]

            # Score all handles
            candidates = []

            for handle in embedding_handles:
                try:
                    scored = scored_by_key.get(handle.logical_key)
                    if scored is None:
                        continue
                    density = float(getattr(scored, 'value_density_ms_per_byte', 0.0))
                    candidates.append({
                        "logical_key": handle.logical_key,
                        "state_type": "embedding",
                        "value_density_ms_per_byte": density,
                        "footprint_bytes": int(handle.footprint_bytes),
                        "total_value_ms": density * int(handle.footprint_bytes),
                        "handle": handle,
                        "scored": scored,
                    })
                except Exception as e:
                    pass

            for handle in kv_handles:
                try:
                    scored = scored_by_key.get(handle.logical_key)
                    if scored is None:
                        continue
                    density = float(getattr(scored, 'value_density_ms_per_byte', 0.0))
                    candidates.append({
                        "logical_key": handle.logical_key,
                        "state_type": "kv_cache",
                        "value_density_ms_per_byte": density,
                        "footprint_bytes": int(handle.footprint_bytes),
                        "total_value_ms": density * int(handle.footprint_bytes),
                        "handle": handle,
                        "scored": scored,
                    })
                except Exception as e:
                    pass

            # Sort by value density (ascending = lowest value first = highest eviction priority)
            candidates.sort(key=lambda c: c["value_density_ms_per_byte"])

            # Separate counts by type
            embedding_candidates = [c for c in candidates if c["state_type"] == "embedding"]
            kv_candidates = [c for c in candidates if c["state_type"] == "kv_cache"]

            return {
                "total_candidates": len(candidates),
                "embedding_candidates": len(embedding_candidates),
                "kv_candidates": len(kv_candidates),
                "all_candidates_ranked": candidates,
                "embedding_candidates_ranked": embedding_candidates,
                "kv_candidates_ranked": kv_candidates,
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_candidates": 0,
                "all_candidates_ranked": [],
            }

    def compute_cross_state_eviction_plan(self, target_bytes_to_free: int) -> dict:
        """Compute eviction plan by comparing value across state types.

        Uses unified eviction oracle to decide: which state type(s) should be
        evicted to free target_bytes_to_free while preserving maximum value.

        Returns eviction plan with:
        - embedding_candidates: list of embedding rows to evict
        - kv_candidates: list of KV pages to evict
        - total_bytes_freed: expected bytes freed
        - total_value_lost_ms: milliseconds of value lost by eviction
        """
        target_bytes = int(target_bytes_to_free)
        candidates_pool = self.build_unified_eviction_candidates()

        all_candidates = [
            candidate
            for candidate in candidates_pool.get("all_candidates_ranked", [])
            if Placement.HBM
            in (
                getattr(candidate.get("handle"), "placement", None) or set()
            )
        ]

        if not all_candidates:
            return {
                "embedding_candidates": [],
                "kv_candidates": [],
                "total_bytes_freed": 0,
                "total_value_lost_ms": 0.0,
                "decision_reason": "no_candidates",
            }

        # Greedily select lowest-value candidates until target is reached
        embedding_to_evict = []
        kv_to_evict = []
        bytes_freed = 0
        value_lost = 0.0

        for candidate in all_candidates:
            if bytes_freed >= target_bytes:
                break

            if candidate["state_type"] == "embedding":
                embedding_to_evict.append(candidate)
            else:
                kv_to_evict.append(candidate)

            bytes_freed += candidate["footprint_bytes"]
            value_lost += candidate["total_value_ms"]

        # Analyze what was evicted
        emb_bytes_evicted = sum(c["footprint_bytes"] for c in embedding_to_evict)
        kv_bytes_evicted = sum(c["footprint_bytes"] for c in kv_to_evict)
        emb_value_lost = sum(c["total_value_ms"] for c in embedding_to_evict)
        kv_value_lost = sum(c["total_value_ms"] for c in kv_to_evict)

        # Decision reasoning
        if kv_bytes_evicted > 0 and emb_bytes_evicted == 0:
            reason = "evicting_only_kv_lower_value"
        elif emb_bytes_evicted > 0 and kv_bytes_evicted == 0:
            reason = "evicting_only_embedding_lower_value"
        elif emb_bytes_evicted > 0 and kv_bytes_evicted > 0:
            reason = "cross_state_eviction_mix"
        else:
            reason = "no_eviction_needed"

        return {
            "embedding_candidates": embedding_to_evict,
            "kv_candidates": kv_to_evict,
            "embedding_bytes_freed": emb_bytes_evicted,
            "kv_bytes_freed": kv_bytes_evicted,
            "total_bytes_freed": bytes_freed,
            "embedding_value_lost_ms": emb_value_lost,
            "kv_value_lost_ms": kv_value_lost,
            "total_value_lost_ms": value_lost,
            "decision_reason": reason,
            "cross_state_comparison": True,
        }

    def apply_cross_state_eviction_plan(
        self, plan: dict, protected_uids=None
    ) -> bool:
        """Apply the unified eviction plan to both embedding and KV.

        Returns True if eviction succeeded.
        """
        try:
            embedding_to_evict = plan.get("embedding_candidates", [])
            kv_to_evict = plan.get("kv_candidates", [])
            embedding_keys = []
            kv_keys = []
            scoring_map = {}

            for candidate in embedding_to_evict + kv_to_evict:
                logical_key = str(candidate.get("logical_key", ""))
                if not logical_key:
                    continue
                scored = candidate.get("scored")
                if scored is not None:
                    scoring_map[logical_key] = scored
                if logical_key.startswith("emb:item:"):
                    try:
                        embedding_keys.append(int(logical_key.rsplit(":", 1)[-1]))
                    except ValueError:
                        continue
                elif logical_key.startswith("kv:uid:"):
                    kv_keys.append(logical_key)

            if not embedding_keys and not kv_keys:
                return int(plan.get("total_bytes_freed", 0)) <= 0

            if protected_uids is not None:
                if hasattr(protected_uids, "detach"):
                    protected_uids = [
                        int(value)
                        for value in protected_uids.detach().cpu().reshape(-1).tolist()
                    ]
                else:
                    protected_uids = [int(value) for value in protected_uids]
            else:
                protected_uids = list(getattr(self, "_protected_uids", []))
            if not protected_uids:
                demand = getattr(self, "_last_demand", None)
                current_user_id = getattr(demand, "current_user_id", None)
                if current_user_id is not None:
                    protected_uids = [int(current_user_id)]

            row_size_bytes = 1
            if embedding_keys:
                row_size_bytes = max(1, int(self.emb_adapter.row_size_bytes()))
            execution = self.scheduler.execute_residency_plan(
                admitted_embedding_keys=[],
                evicted_embedding_keys=embedding_keys,
                evicted_kv_keys=kv_keys,
                scoring_map=scoring_map,
                protected_uids=protected_uids,
                epoch=self.epoch,
                row_size_bytes=row_size_bytes,
            )
            self._sync_directory(epoch=self.epoch)

            confirmed_embedding = {
                int(key) for key in execution.evicted_embedding_keys
            }
            confirmed_kv = set(execution.released_kv_keys)
            return (
                set(embedding_keys).issubset(confirmed_embedding)
                and set(kv_keys).issubset(confirmed_kv)
                and not execution.backend_errors
                and not execution.failed_ids
                and not execution.deferred_ids
            )
        except Exception:
            return False

    def report_cross_state_comparison(self) -> dict:
        """Generate report on cross-state value comparison and unified eviction.

        Returns summary showing that embedding and KV are being compared and managed
        as a unified pool rather than separate systems.
        """
        normalized_densities = self.compute_normalized_value_densities()
        candidates = self.build_unified_eviction_candidates()

        total_candidates = candidates.get("total_candidates", 0)
        embedding_count = candidates.get("embedding_candidates", 0)
        kv_count = candidates.get("kv_candidates", 0)

        summary = {
            "unified_comparison_active": True,
            "total_candidates_compared": total_candidates,
            "embedding_candidates": embedding_count,
            "kv_candidates": kv_count,
            "embedding_value_density_ms_per_byte": normalized_densities.get("embedding_value_density_ms_per_byte", 0.0),
            "kv_value_density_ms_per_byte": normalized_densities.get("kv_value_density_ms_per_byte", 0.0),
            "embedding_is_more_valuable": normalized_densities.get("embedding_is_more_valuable", False),
            "kv_is_more_valuable": normalized_densities.get("kv_is_more_valuable", False),
            "value_density_ratio": normalized_densities.get("value_density_ratio", 1.0),
        }

        print(
            f"\n[HotState Cross-State Value Comparison]\n"
            f"  Unified Pool: {total_candidates} candidates (Embedding: {embedding_count}, KV: {kv_count})\n"
            f"  Embedding Value Density: {normalized_densities.get('embedding_value_density_ms_per_byte', 0.0):.6f} ms/byte\n"
            f"  KV Value Density: {normalized_densities.get('kv_value_density_ms_per_byte', 0.0):.6f} ms/byte\n"
            f"  More Valuable: {'Embedding' if normalized_densities.get('embedding_is_more_valuable') else 'KV'}\n"
            f"  Value Ratio (KV/Embedding): {normalized_densities.get('value_density_ratio', 1.0):.2f}x\n"
        )

        return summary

    # ============================================================================
    # GAP 5: Semantic Differences and Consistency Policy Enforcement
    # ============================================================================

    def get_semantic_policy(self, state_type: str) -> dict:
        """Get the semantic policy for a state type.

        Returns policy defining consistency model, regenerability, and constraints.
        """
        return self._semantic_policies.get(state_type, {})

    def compute_semantic_allocation(self) -> dict:
        """Compute allocation that respects semantic differences between state types.

        Embedding: flexible, can be evicted more aggressively
        KV: must preserve session state, higher minimum budget

        Returns allocation with semantic minimums enforced.
        """
        total_budget = self.configured_state_budget_bytes
        occupancy = self.get_current_occupancy()

        # Semantic minimums
        kv_semantic_min_ratio = 0.35  # KV must have at least 35% (session criticality)
        embedding_semantic_min_ratio = 0.20  # Embedding can go lower (regenerable)

        kv_min_budget = int(total_budget * kv_semantic_min_ratio)
        embedding_min_budget = int(total_budget * embedding_semantic_min_ratio)

        # Allocate remaining budget based on current utilization
        current_emb = occupancy["embedding_utilization_ratio"]
        current_kv = occupancy["kv_utilization_ratio"]

        # Adjust allocation if utilization patterns suggest rebalancing
        if current_kv > 0.80 and current_emb < 0.60:
            # KV is heavily used, embedding is underutilized
            # Increase KV within semantic bounds
            kv_ratio = min(0.70, 0.35 + (current_kv - current_emb) * 0.2)
        elif current_emb > 0.80 and current_kv < 0.60:
            # Embedding is heavily used, KV is underutilized
            # Increase embedding but maintain KV minimum
            kv_ratio = max(kv_semantic_min_ratio, 0.35 - (current_emb - current_kv) * 0.15)
        else:
            # Balanced or moderate pressure
            kv_ratio = 0.5

        # Ensure minimums are respected
        kv_ratio = max(kv_semantic_min_ratio, kv_ratio)
        embedding_ratio = 1.0 - kv_ratio

        embedding_budget = int(total_budget * embedding_ratio)
        kv_budget = total_budget - embedding_budget

        return {
            "embedding_budget_bytes": embedding_budget,
            "kv_budget_bytes": kv_budget,
            "embedding_ratio": embedding_ratio,
            "kv_ratio": kv_ratio,
            "semantic_constraints_enforced": True,
            "kv_semantic_min": kv_min_budget,
            "embedding_semantic_min": embedding_min_budget,
        }

    def check_semantic_eviction_constraint(self, candidate: dict) -> tuple:
        """Check if evicting a candidate violates semantic constraints.

        Returns (is_violation, reason).
        """
        state_type = candidate.get("state_type", "unknown")
        policy = self.get_semantic_policy(state_type)

        if not policy:
            return False, "unknown_state_type"

        # KV: Check if eviction violates session-bound requirement
        if state_type == "kv_cache":
            # Check if this is critical for current session
            if policy.get("session_bound", False):
                # In real implementation, would check session requirement
                # For now, be conservative: avoid evicting KV if possible
                occupancy = self.get_current_occupancy()
                kv_util = occupancy.get("kv_utilization_ratio", 0.5)

                # If KV is highly utilized, eviction would violate consistency
                if kv_util > 0.75:
                    return True, "kv_session_critical_high_utilization"

        # Embedding: Can regenerate, so eviction is generally safe
        if state_type == "embedding":
            # Only a violation if we're below semantic minimum
            pass

        return False, "allowed"

    def apply_semantic_aware_eviction(self, eviction_plan: dict) -> dict:
        """Apply eviction while respecting semantic constraints.

        Reorders eviction candidates to prefer:
        1. Embedding (regenerable, lower semantic cost)
        2. Only evict KV if necessary (session-critical)

        Returns modified eviction plan.
        """
        embedding_candidates = eviction_plan.get("embedding_candidates", [])
        kv_candidates = eviction_plan.get("kv_candidates", [])

        # Check for semantic violations
        semantic_violations = []
        for kv_candidate in kv_candidates:
            is_violation, reason = self.check_semantic_eviction_constraint(kv_candidate)
            if is_violation:
                semantic_violations.append((kv_candidate, reason))
                self._semantic_violations += 1

        # If we have semantic violations, try to substitute with embedding eviction
        if semantic_violations and embedding_candidates:
            # Log the semantic override
            print(
                f"[HotState Semantic Policy] Avoiding KV eviction due to: "
                f"{semantic_violations[0][1] if semantic_violations else 'unknown'}. "
                f"Substituting with embedding eviction."
            )
            # In this case, keep the embedding eviction, drop KV eviction
            kv_candidates = []

        self._semantic_decisions += 1

        return {
            "embedding_candidates": embedding_candidates,
            "kv_candidates": kv_candidates,
            "semantic_violations_detected": len(semantic_violations) > 0,
            "semantic_violations": semantic_violations,
            "total_bytes_freed": sum(c.get("footprint_bytes", 0) for c in embedding_candidates + kv_candidates),
        }

    def compute_semantic_aware_allocation(self) -> dict:
        """Compute adaptive allocation while respecting semantic differences.

        Combines value-based allocation with semantic constraints.

        Returns allocation plan.
        """
        adaptive_plan = self.compute_adaptive_allocation()
        semantic_plan = self.compute_semantic_allocation()

        # Merge: semantic constraints override if necessary
        emb_adaptive = adaptive_plan.get("recommended_embedding_hbm_bytes", 0)
        kv_adaptive = adaptive_plan.get("recommended_kv_hbm_bytes", 0)

        emb_semantic = semantic_plan.get("embedding_budget_bytes", 0)
        kv_semantic = semantic_plan.get("kv_budget_bytes", 0)

        # Final allocation: use semantic if stricter, else adaptive
        emb_final = max(emb_adaptive, emb_semantic)
        kv_final = max(kv_adaptive, kv_semantic)

        # Normalize to stay within total budget
        total = emb_final + kv_final
        if total > self.configured_state_budget_bytes:
            # Reduce over-budget state while respecting semantic minimums
            total_budget = self.configured_state_budget_bytes
            if emb_final > emb_semantic:
                emb_final = int(emb_final * self.configured_state_budget_bytes / total)
            if kv_final > kv_semantic:
                kv_final = int(kv_final * self.configured_state_budget_bytes / total)

        # Final adjustment to fit budget
        if emb_final + kv_final != self.configured_state_budget_bytes:
            kv_final = self.configured_state_budget_bytes - emb_final

        return {
            "recommended_embedding_hbm_bytes": emb_final,
            "recommended_kv_hbm_bytes": kv_final,
            "reallocation_triggered": True,
            "reallocation_ratio": abs((emb_final / self.configured_state_budget_bytes) - 0.5),
            "decision_reason": "semantic_aware_adaptive",
            "adaptive_plan": adaptive_plan,
            "semantic_plan": semantic_plan,
            "semantic_constraints_enforced": True,
        }

    def report_semantic_adherence(self) -> dict:
        """Generate report on semantic policy adherence.

        Shows how well the system is respecting semantic differences between
        embedding and KV states.
        """
        total_decisions = max(1, self._semantic_decisions)
        violation_rate = self._semantic_violations / total_decisions

        embedding_policy = self.get_semantic_policy("embedding")
        kv_policy = self.get_semantic_policy("kv_cache")

        summary = {
            "semantic_policies_active": True,
            "total_decisions": total_decisions,
            "semantic_violations": self._semantic_violations,
            "violation_rate": violation_rate,
            "embedding_policy": embedding_policy,
            "kv_policy": kv_policy,
            "embedding_regenerable": embedding_policy.get("regenerable", False),
            "embedding_staleness_tolerance": embedding_policy.get("staleness_tolerance_epochs", 0),
            "kv_session_bound": kv_policy.get("session_bound", False),
            "kv_authoritative_required": kv_policy.get("authoritative_copy_required", False),
        }

        print(
            f"\n[HotState Semantic Policy Adherence]\n"
            f"  Active: True\n"
            f"  Decisions: {total_decisions}\n"
            f"  Semantic Violations: {self._semantic_violations} ({violation_rate:.1%})\n"
            f"  Embedding:\n"
            f"    - Consistency: {embedding_policy.get('consistency_class', 'unknown')}\n"
            f"    - Regenerable: {embedding_policy.get('regenerable', False)}\n"
            f"    - Staleness Tolerance: {embedding_policy.get('staleness_tolerance_epochs', 0)} epochs\n"
            f"  KV Cache:\n"
            f"    - Consistency: {kv_policy.get('consistency_class', 'unknown')}\n"
            f"    - Session-Bound: {kv_policy.get('session_bound', False)}\n"
            f"    - Authoritative Copy Required: {kv_policy.get('authoritative_copy_required', False)}\n"
        )

        return summary
