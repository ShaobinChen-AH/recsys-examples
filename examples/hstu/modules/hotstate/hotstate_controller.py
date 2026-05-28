from modules.hotstate.state_handle import Placement
from modules.hotstate.demand_signal import DemandSignal
from modules.hotstate.state_registry import StateRegistry
from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.value_engine import ValueEngine
from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter
from modules.hotstate.hot_set_manager import HotSetManager


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
        self.epoch = 0

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

    # Public API
    def set_embedding_module(self, embedding_module):
        """Connect the embedding adapter after both dense and sparse modules exist."""
        self.emb_adapter.embedding_module = embedding_module
        self.emb_adapter.calibrate()
        # Re-initialize directory with embedding handles
        for h in self.emb_adapter.export_handles():
            self.registry.register(h)
            if Placement.HBM in h.placement:
                self.directory.register(
                    h.logical_key, Placement.HBM, authoritative=False)

    def before_batch(self, batch, user_ids, total_history_lengths) -> dict:
        """Called before each inference batch.

        Returns control decisions for logging/monitoring.
        """
        uid = int(user_ids[0].item())
        hist_len = int(total_history_lengths[0].item()) // 2

        # Extract item indices for hot-key tracking
        try:
            item_values = batch.features["item_feat"].values()
            item_indices = item_values.unique().tolist()
        except Exception:
            item_indices = []

        # Track hot keys for embedding adapter
        self.emb_adapter.record_batch_keys(item_indices)

        # Build demand signal
        demand = DemandSignal(
            current_user_id=uid,
            history_length=hist_len,
            num_candidates=batch.max_num_candidates or 100,
            epoch=self.epoch,
            item_indices=item_indices,
        )

        # Check completed transfers
        self._check_completions()

        # Run control epoch
        result = self.hot_set.run_epoch(self.epoch, demand)

        # Update access statistics in ValueEngine
        for key in result.evicted_keys + result.admitted_keys:
            self.value_engine.record_access(key, self.epoch)
        self.value_engine.record_access(
            f"kv:uid:{uid}", self.epoch)

        # Decay old access logs
        if self.epoch % 50 == 0:
            self.value_engine.decay_logs(self.epoch)

        self.epoch += 1

        return {
            "kv_page_budget": result.kv_page_budget,
            "hbm_bytes_used": result.hbm_bytes_used,
            "evicted": len(result.evicted_keys),
            "admitted": len(result.admitted_keys),
            "epoch": self.epoch - 1,
        }

    def after_batch(self, batch, latency_ms: float):
        """Called after inference. Updates access statistics."""
        for feature_name in batch.features.keys():
            self.value_engine.record_access(
                f"emb:{feature_name}", self.epoch)

    def _check_completions(self):
        """Check subsystem-level transfer completion."""
        # In V1: transfers are synchronous when issued by HotSetManager,
        # so there's nothing async to check here.
        # In V2: poll cudaEventQuery on onload/offload events.
        pass
