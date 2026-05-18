from typing import List

from modules.hotstate.state_handle import (
    StateHandle, StateType, Placement, Reconstructability
)


class KVAdapter:
    """Bridges AsyncHSTUKVCacheManager into HotState at per-user page granularity.

    Each user's allocated KV pages form a StateHandle. The controller can
    compare individual user caches against embedding row groups.
    """

    def __init__(self, async_kvcache_manager):
        self._kvcache = async_kvcache_manager
        self._gpu_mgr = async_kvcache_manager.gpu_kvcache_mgr

    @property
    def num_users(self):
        return self._kvcache.max_num_sequences

    @property
    def page_size_tokens(self):
        return self._kvcache.page_size

    @property
    def head_dim(self):
        return self._kvcache.head_dim

    @property
    def num_heads(self):
        return self._kvcache.num_heads

    @property
    def num_layers(self):
        return self._kvcache.num_layers

    def _page_bytes(self) -> int:
        """Bytes per KV page across all layers."""
        return (self.num_layers * self.page_size_tokens * 2  # K+V
                * self.num_heads * self.head_dim * 2)         # bf16

    def get_page_count_for_user(self, uid: int) -> int:
        """Number of pages currently allocated to user uid."""
        return self._gpu_mgr.get_user_page_count(uid)

    def get_empty_page_count(self) -> int:
        """Number of free pages in the pool."""
        return self._gpu_mgr.get_empty_page_count()

    def get_current_page_limit(self) -> int:
        return getattr(self._gpu_mgr, '_active_page_limit',
                       self._kvcache.num_primary_cache_pages)

    def set_page_limit(self, new_limit: int) -> None:
        """Dynamically adjust the KV page budget."""
        self._gpu_mgr.set_active_page_limit(new_limit)

    def evict_user(self, uid: int) -> None:
        """Release all pages for a user back to the empty pool."""
        try:
            self._gpu_mgr.evict(uid)
        except Exception:
            pass

    def export_handles(self) -> List[StateHandle]:
        """Export one handle per user with pages allocated to them."""
        handles = []
        page_bytes = self._page_bytes()
        for uid in range(self.num_users):
            page_count = self.get_page_count_for_user(uid)
            if page_count == 0:
                continue
            footprint = page_count * page_bytes
            handles.append(StateHandle(
                state_type=StateType.SESSION_KV_USER,
                logical_key=f"kv:uid:{uid}",
                footprint_bytes=footprint,
                placement={Placement.HBM},
                reconstructability=Reconstructability.RECOMPUTABLE,
                consistency_class="derived_recomputable",
            ))
        return handles

    def total_hbm_bytes(self) -> int:
        return self.get_current_page_limit() * self._page_bytes()
