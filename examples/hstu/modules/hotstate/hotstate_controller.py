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


class HotStateController:
    """Unified HBM control plane for generative recommendation inference."""

    def __init__(self, total_hbm_bytes: int, kv_module, embedding_module = None):
        self.emb_adapter = EmbeddingAdapter(embedding_module)
        self.kv_adapter = KVAdapter(kv_module)
        self.registry = StateRegistry()
        self.directory = GlobalDirectory()
        self.value_engine = ValueEngine(total_hbm_bytes)
        self.hot_set = HotSetManager(
            total_hbm_bytes, self.value_engine,
            self.registry, self.directory,
            self.emb_adapter, self.kv_adapter)
        self.scheduler = TransferScheduler(self.kv_adapter, self.directory, self.value_engine)
        self.epoch = 0

        self.enable_transfer_scheduler = False

        self.trace_detail = "scalar"

        # Calibrate adapter (read actual table sizes)
        self.emb_adapter.calibrate()

        # Initialize directory with current HBM placement
        for h in self.emb_adapter.export_handles():
            self.registry.register(h)
            if Placement.HBM in h.placement:
                self.directory.register(
                    h.logical_key, Placement.HBM, authoritative=False)
        for h in self.kv_adapter.export_handles():
            self.registry.register(h)
            self.directory.register(
                h.logical_key, Placement.HBM, authoritative=False)

    def set_trace_detail(self, trace_detail: str):
        if trace_detail not in ("scalar", "full"):
            raise ValueError(f"unknown trace_detail: {trace_detail}")
        self.trace_detail = trace_detail

    # Public API
    def set_embedding_module(self, embedding_module):
        """Connect the embedding adapter after both dense and sparse modules exist."""
        self.emb_adapter.set_module(embedding_module)

    def before_batch(self, batch, user_ids, total_history_lengths, batch_idx=0) -> dict:
        """Called before each inference batch. Runs control epoch + transfer planning."""
        t = time.perf_counter()

        uid = int(user_ids[0].item())
        hist_len = int(total_history_lengths[0].item()) // 2
        self.kv_adapter.record_batch_users(user_ids)

        try:
            item_values = batch.features["item_feat"].values()
            item_indices = item_values.unique().tolist()
        except Exception:
            item_indices = []

        t1 = time.perf_counter()

        self.emb_adapter.record_batch_keys(item_indices)

        self.emb_adapter.update_admission_policy(
            item_indices=item_indices,
            enabled=True,
        )

        t2 = time.perf_counter()

        demand = DemandSignal(
            current_user_id=uid, history_length=hist_len,
            num_candidates=batch.max_num_candidates or 100,
            epoch=self.epoch, item_indices=item_indices)

        self._last_demand = demand

        # 1. Check completed offloads from previous epochs
        completed = []
        if self.enable_transfer_scheduler:
            completed = self.scheduler.poll_completions(self.epoch)

        # 2. Score and decide: what to keep, what to evict
        result = self.hot_set.run_epoch(self.epoch, demand)
        t3 = time.perf_counter()

        # 3. Plan and submit transfers with priority ordering
        # Build a quick scoring lookup for reuse_imminence
        if self.enable_transfer_scheduler:
            scoring_map = {}
            for s in self.value_engine.compute_scores(
                self.registry.snapshot(), demand):
                scoring_map[s.handle.logical_key] = s

            self.scheduler.plan_and_submit(
                current_batch=batch_idx,
                evicted_keys=result.evicted_keys,
                scoring_map=scoring_map,
                epoch=self.epoch)

        t4 = time.perf_counter()

        if self.epoch <= 5 or self.epoch % 50 == 0:
            print(f"  [PROFILE epoch {self.epoch}] "
                  f"extract={1000*(t1-t):.1f}ms "
                  f"record_keys={1000*(t2-t1):.1f}ms "
                  f"run_epoch={1000*(t3-t2):.1f}ms "
                  f"scheduler={1000*(t4-t3):.1f}ms "
                  f"total={1000*(t4-t):.1f}ms")

        # 4. Track access
        for key in result.evicted_keys + result.admitted_keys:
            self.value_engine.record_access(key, self.epoch)
        self.value_engine.record_access(f"kv:uid:{uid}", self.epoch)

        if self.epoch % 50 == 0:
            self.value_engine.decay_logs(self.epoch)

        self.epoch += 1

        return {
            "kv_page_budget": result.kv_page_budget,
            "hbm_bytes_used": result.hbm_bytes_used,
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
                if self.trace_detail == "full"
                else []
            ),
            "selected_hbm_bytes": result.selected_hbm_bytes,
            "selected_kv_bytes": result.selected_kv_bytes,
            "selected_embedding_bytes": result.selected_embedding_bytes,
        }


    def after_batch(self, batch, latency_ms: float):
        """Called after inference. Updates access statistics and snapshots post-forward state."""
        for feature_name in batch.features.keys():
            self.value_engine.record_access(f"emb:{feature_name}", self.epoch)

        if self.trace_detail != "full":
            return {"post_num_state_handles": 0, "post_state_trace": []}

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
        }

    def _check_completions(self):
        """Check subsystem-level transfer completion."""
        # In V1: transfers are synchronous when issued by HotSetManager,
        # so there's nothing async to check here.
        # In V2: poll cudaEventQuery on onload/offload events.
        pass
   
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
                "reconstructability": handle.reconstructability.name,
                "consistency_class": handle.consistency_class,
                "reuse_imminence": float(handle.reuse_imminence),
                "stall_sensitivity_ms": float(handle.stall_sensitivity_ms),
                "movement_cost_ms": float(handle.movement_cost_ms),
                "score": float(scored.score),
                "benefit_density": float(scored.benefit_density),
                "occupancy_penalty": float(scored.occupancy_penalty),
                "semantic_risk": float(scored.semantic_risk),
                "gross_benefit_ms": float(getattr(scored, "gross_benefit_ms", 0.0)),
                "movement_penalty_ms": float(getattr(scored, "movement_penalty_ms", 0.0)),
                "semantic_risk_ms": float(getattr(scored, "semantic_risk_ms", 0.0)),
                "net_benefit_ms": float(getattr(scored, "net_benefit_ms", 0.0)),
                "value_density_ms_per_byte": float(
                    getattr(scored, "value_density_ms_per_byte", 0.0)
                ),
                "decision": decision_by_key.get(handle.logical_key, "unknown"),
            })
        return records
