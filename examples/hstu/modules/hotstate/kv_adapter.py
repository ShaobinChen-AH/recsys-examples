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
        self._active_page_limit = self._kvcache.num_primary_cache_pages

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
    
    def get_withheld_page_count(self) -> int:
        if hasattr(self._gpu_mgr, "get_withheld_page_count"):
            return int(self._gpu_mgr.get_withheld_page_count())
        return 0

    def get_resident_page_count(self) -> int:
        if hasattr(self._gpu_mgr, "get_resident_page_count"):
            return int(self._gpu_mgr.get_resident_page_count())
        resident = 0
        for uid in range(self.num_users):
            resident += int(self.get_page_count_for_user(uid))
        return resident

    def get_physical_page_count(self) -> int:
        return int(self._kvcache.num_primary_cache_pages)

    def get_current_page_limit(self) -> int:
        if hasattr(self._gpu_mgr, "get_active_page_limit"):
            return int(self._gpu_mgr.get_active_page_limit())
        return int(self._active_page_limit)

    def set_page_limit(self, new_limit: int) -> bool:
        """Adjust logical KV budget; physical KV tensor size is unchanged."""
        new_limit = int(new_limit)
        if hasattr(self._gpu_mgr, "set_active_page_limit"):
            self._gpu_mgr.set_active_page_limit(new_limit)
            self._active_page_limit = self.get_current_page_limit()
            return self._active_page_limit == new_limit
        self._active_page_limit = self.get_current_page_limit()
        return new_limit == self._active_page_limit

    def evict_user(self, uid: int) -> None:
        """Release all pages for a user back to the empty pool."""
        return bool(self._gpu_mgr.evict_if_present(uid))

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
    
    def logical_kv_budget_bytes(self) -> int:
        return self.get_current_page_limit() * self._page_bytes()

    def physical_kv_cache_bytes(self) -> int:
        return self.get_physical_page_count() * self._page_bytes()

    def actual_resident_kv_bytes(self) -> int:
        return self.get_resident_page_count() * self._page_bytes()

    def total_hbm_bytes(self) -> int:
        return self.logical_kv_budget_bytes()
