from modules.hotstate.state_handle import Placement
from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.state_registry import StateRegistry
from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.value_engine import ValueEngine
from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter
from modules.hotstate.hot_set_manager import HotSetManager
from modules.hotstate.transfer_scheduler import TransferScheduler
import time
import os
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
    ):
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
        self.kv_adapter = KVAdapter(kv_module)
        self.registry = StateRegistry()
        self.directory = GlobalDirectory()
        self.value_engine = ValueEngine(self.configured_state_budget_bytes)
        self.hot_set = HotSetManager(
            self.configured_state_budget_bytes, self.value_engine,
            self.registry, self.directory,
            self.emb_adapter, self.kv_adapter, skip_kv_handles_for_admission_smoke=skip_kv_handles_for_admission_smoke,)
        self.scheduler = TransferScheduler(
            self.kv_adapter,
            self.directory,
            self.value_engine,
            embedding_adapter=self.emb_adapter,
            registry=self.registry,
        )
        self.epoch = 0

        # The scheduler now executes real DynamicEmb movement and stages the
        # native request-scoped KV onload consumed by model forward.
        self.enable_transfer_scheduler = True

        self.admission_smoke_max_admitted_keys = admission_smoke_max_admitted_keys

        self.trace_detail = "scalar"

        self.num_users = 8
        self.admission_batch_order_control = False
        self._pending_batch_cost_context = None

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
            "kv_physical_hbm_breakdown": self.kv_adapter.physical_hbm_breakdown(),
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
        t = time.perf_counter()

        uid = int(user_ids[0].item())
        total_history_tokens = int(total_history_lengths[0].item())
        hist_len = total_history_tokens // 2
        self.kv_adapter.record_batch_users(user_ids)
        if hasattr(user_ids, "detach"):
            protected_uids = [int(value) for value in user_ids.detach().cpu().tolist()]
        else:
            protected_uids = [int(value) for value in user_ids]
        if hasattr(total_history_lengths, "detach"):
            request_history_lengths = [
                int(value)
                for value in total_history_lengths.detach().cpu().tolist()
            ]
        else:
            request_history_lengths = [int(value) for value in total_history_lengths]

        try:
            item_values = batch.features["item_feat"].values()
            item_sequence = [int(x) for x in item_values.detach().cpu().reshape(-1).tolist()]
            item_indices = []
            seen_item_indices = set()
            for item_idx in item_sequence:
                if item_idx in seen_item_indices:
                    continue
                seen_item_indices.add(item_idx)
                item_indices.append(item_idx)
        except Exception:
            item_sequence = []
            item_indices = []

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
        hotset_ms = 1000 * (time.perf_counter() - t_hotset_start)

        planned_hbm_bytes = int(getattr(result, "planned_hbm_bytes", 0) or 0)
        planned_kv_bytes = int(getattr(result, "planned_kv_bytes", 0) or 0)
        planned_embedding_bytes = int(
            getattr(result, "planned_embedding_bytes", 0) or 0
        )

        kv_page_budget = int(getattr(result, "kv_page_budget", 0) or 0)
        kv_page_bytes = int(self.kv_adapter._page_bytes())
        kv_reserved_bytes = kv_page_budget * kv_page_bytes

        total_hbm_bytes = int(getattr(self.hot_set, "total_hbm", 0) or 0)
        residual_embedding_budget_bytes = (
            max(0, total_hbm_bytes - kv_reserved_bytes)
            if total_hbm_bytes > 0
            else planned_embedding_bytes
        )

        embedding_budget_bytes = min(
            planned_embedding_bytes,
            residual_embedding_budget_bytes,
        )

        row_size_bytes = max(1, self.emb_adapter.row_size_bytes())
        hotstate_admission_budget_keys = embedding_budget_bytes // row_size_bytes

        admit_all_control = bool(getattr(self, "admission_admit_all_control", False))
        batch_order_control = bool(getattr(self, "admission_batch_order_control", False))
        smoke_cap = getattr(self, "admission_smoke_max_admitted_keys", None)
        trace_detail = getattr(self, "trace_detail", "scalar")
        admission_trace = []

        admission_start = time.perf_counter()
        ranking_ms = 0.0
        if admit_all_control:
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

        if self.epoch <= 5 or self.epoch % 50 == 0:
            print(f"  [PROFILE epoch {self.epoch}] "
                  f"extract={1000*(t1-t):.1f}ms "
                  f"record_keys={1000*(t2-t1):.1f}ms "
                  f"run_epoch={1000*(t3-t2):.1f}ms "
                  f"scheduler={1000*(t4-t3):.1f}ms "
                  f"total={1000*(t4-t):.1f}ms"
                  f"hotset={hotset_ms:.1f}ms "
                  f"admission={admission_ms:.1f}ms "
                  f"ranking={ranking_ms:.1f}ms "
                  f"policy={policy_ms:.1f}ms ")
                  


        # 4. Track demand access only.  Residency movement is not an access;
        # feeding admissions/evictions into reuse history would teach the
        # predictor its own decisions instead of the request stream.
        self.value_engine.record_access(f"kv:uid:{uid}", self.epoch)

        if self.epoch % 50 == 0:
            self.value_engine.decay_logs(self.epoch)

        self._pending_batch_cost_context = {
            "history_tokens": total_history_tokens,
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
            "planned_kv_page_budget": result.planned_kv_page_budget,
            "hbm_bytes_used": physical_hbm["managed_state_physical_hbm_bytes"],
            "policy_planned_hbm_bytes": planned_hbm_bytes,
            "policy_planned_kv_bytes": planned_kv_bytes,
            "policy_planned_embedding_bytes": planned_embedding_bytes,
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
            "directory_pending_transfers": self.directory.pending_count(),
            "directory_state": (
                self.directory.snapshot() if trace_detail == "full" else []
            ),
            "directory_history_tail": (
                self.directory.history()[-32:] if trace_detail == "full" else []
            ),
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
            origin = int(origin_lengths[0]) if origin_lengths else 0
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
            return {
                "post_num_state_handles": 0,
                "post_state_trace": [],
                "post_directory_pending_transfers": self.directory.pending_count(),
                "consistency": self.consistency_snapshot(),
                "online_cost_model": self.value_engine.cost_model_snapshot(),
                "transfer_scheduler": scheduler_post,
            }

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

        return {
            "post_num_state_handles": len(post_state_trace),
            "post_state_trace": post_state_trace,
            "post_directory_state": self.directory.snapshot(),
            "consistency": self.consistency_snapshot(),
            "online_cost_model": self.value_engine.cost_model_snapshot(),
            "transfer_scheduler": scheduler_post,
        }

    def _check_completions(self):
        """Check subsystem-level transfer completion."""
        return self.scheduler.poll_completions(self.epoch)

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
            })
        return records
