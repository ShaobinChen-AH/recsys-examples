import threading
from typing import Iterable, List, Set

from modules.hotstate.state_handle import (
    ConsistencyError,
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
        self._known_user_ids: Set[int] = set()
        # The native manager exposes GPU page counts but does not currently
        # expose a host-directory enumeration API.  Keep an explicit record of
        # users this adapter successfully offloaded so traces do not claim an
        # HBM replica after the physical operation completed.
        self._offloaded_user_ids: Set[int] = set()
        self._inflight_users: Set[int] = set()
        self._state_lock = threading.RLock()
        self._last_manifest: dict = {}

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

    def record_batch_users(self, user_ids) -> None:
        if hasattr(user_ids, "detach"):
            user_ids = user_ids.detach().cpu().tolist()
        for uid in user_ids:
            uid = int(uid)
            self._known_user_ids.add(uid)
            try:
                if self.get_page_count_for_user(uid) > 0:
                    self._offloaded_user_ids.discard(uid)
            except (AttributeError, RuntimeError):
                pass

    def _candidate_user_ids(self):
        if self._known_user_ids:
            return sorted(self._known_user_ids)
        if self.num_users is not None and self.num_users > 0:
            return range(int(self.num_users))
        return []

    def _page_bytes(self) -> int:
        """Bytes per KV page across all layers."""
        return (self.num_layers * self.page_size_tokens * 2  # K+V
                * self.num_heads * self.head_dim * 2)         # bf16

    def token_bytes(self) -> int:
        """Bytes for one sequence token across every KV layer."""
        return self._page_bytes() // max(1, int(self.page_size_tokens))

    def is_user_offloaded(self, uid: int) -> bool:
        uid = int(uid)
        return (
            uid in self._offloaded_user_ids
            or (self.host_token_count(uid) > 0 and self.get_page_count_for_user(uid) == 0)
        )

    def host_token_count(self, uid: int) -> int:
        """Tokens with an authoritative prefix in native host KV storage."""
        get_length = getattr(self._kvcache.host_kv_mgr, "get_kvdata_length", None)
        if not callable(get_length):
            return 0
        return max(0, int(get_length(int(uid))))

    def has_host_copy(self, uid: int) -> bool:
        return self.host_token_count(uid) > 0

    def stage_request(self, user_ids, total_history_lengths) -> dict:
        """Launch the real KV prepare/onload for later one-shot consumption."""
        normalized_users = [int(uid) for uid in user_ids]
        normalized_lengths = [int(length) for length in total_history_lengths]
        if len(normalized_users) != len(normalized_lengths):
            raise ConsistencyError("KV user IDs and history lengths must be aligned")
        if len(set(normalized_users)) != len(normalized_users):
            raise ConsistencyError("a KV request cannot contain duplicate user IDs")
        with self._state_lock:
            overlap = set(normalized_users) & self._inflight_users
            if overlap:
                raise ConsistencyError(
                    "KV request overlaps an in-flight user: "
                    + ", ".join(str(uid) for uid in sorted(overlap))
                )
            manifests = {}
            for uid, requested_length in zip(normalized_users, normalized_lengths):
                if requested_length < 0:
                    raise ConsistencyError("KV history lengths must be non-negative")
                manifest = self.user_manifest(uid)
                logical_length = int(manifest["logical_length"])
                if requested_length < logical_length:
                    raise ConsistencyError(
                        f"KV user {uid} requested length {requested_length} "
                        f"rewinds committed length {logical_length}"
                    )
                manifests[uid] = manifest
            self._inflight_users.update(normalized_users)
        try:
            generation = self._kvcache.stage_prepare_kvcache(
                normalized_users, normalized_lengths
            )
        except Exception:
            self.finish_staged_request(normalized_users, success=False)
            raise
        return {
            "generation": int(generation),
            "user_ids": normalized_users,
            "total_history_lengths": normalized_lengths,
            "manifests": manifests,
        }

    def finish_staged_request(self, user_ids, *, success: bool = True) -> None:
        """Release request ownership after forward has consumed its buffers."""
        with self._state_lock:
            for uid in user_ids:
                self._inflight_users.discard(int(uid))

    def user_manifest(self, uid: int) -> dict:
        """Describe the append-only KV segments without flattening replicas."""
        uid = int(uid)
        host_prefix = self.host_token_count(uid)
        get_total = getattr(self._gpu_mgr, "get_total_cache_length", None)
        total = host_prefix
        if callable(get_total):
            try:
                total = max(host_prefix, int(get_total([uid])[0]))
            except (IndexError, TypeError, RuntimeError):
                total = host_prefix
        page_count = int(self.get_page_count_for_user(uid))
        hbm_start = host_prefix if page_count else total
        hbm_tokens = max(0, total - hbm_start) if page_count else 0
        # Native offload may have reserved a suffix that is not durable in the
        # host map yet.  It remains protected until the completion counter says
        # the copy is done.
        pending = getattr(self._kvcache, "last_offload_submission", {}) or {}
        pending_tokens = int((pending.get("tokens_by_user", {}) or {}).get(uid, 0))
        return {
            "user_id": uid,
            "host_prefix_tokens": int(host_prefix),
            "hbm_start_token": int(hbm_start),
            "hbm_tokens": int(hbm_tokens),
            "logical_length": int(max(total, host_prefix + hbm_tokens)),
            "page_count": page_count,
            "pending_offload_tokens": max(0, pending_tokens),
            "append_only": True,
            "version": int(max(total, host_prefix + hbm_tokens)),
        }

    def native_offload_snapshot(self) -> dict:
        return dict(getattr(self._kvcache, "last_offload_submission", {}) or {})

    def completed_offload_count(self) -> int:
        get_count = getattr(self._gpu_mgr, "get_completed_offload_count", None)
        if callable(get_count):
            return int(get_count())
        return int(not self.is_busy_offloading())

    def is_busy_offloading(self) -> bool:
        busy = getattr(self._gpu_mgr, "is_busy_offloading", None)
        return bool(busy()) if callable(busy) else False

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
        for uid in self._candidate_user_ids():
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

    def evict_user(self, uid: int) -> bool:
        """Release all pages for a user back to the empty pool."""
        uid = int(uid)
        with self._state_lock:
            if uid in self._inflight_users:
                return False
        is_frozen = getattr(self._gpu_mgr, "is_user_offload_frozen", None)
        if callable(is_frozen) and bool(is_frozen(uid)):
            return False
        moved = bool(self._gpu_mgr.evict_if_present(uid))
        if moved:
            self._offloaded_user_ids.add(uid)
        return moved

    def apply_eviction_plan(self, logical_keys: Iterable[str], protected_uids=()) -> List[str]:
        """Physically evict planned users and return only successful moves."""
        protected = {int(uid) for uid in protected_uids}
        applied: List[str] = []
        for logical_key in logical_keys:
            if not logical_key.startswith("kv:uid:"):
                continue
            try:
                uid = int(logical_key.rsplit(":", 1)[-1])
            except ValueError:
                continue
            if uid in protected:
                continue
            if self.evict_user(uid):
                applied.append(logical_key)
        return applied

    def export_handles(self) -> List[StateHandle]:
        """Export one handle per user with observed HBM/host placement."""
        handles = []
        page_bytes = self._page_bytes()
        candidate_ids = set(self._candidate_user_ids()) | self._offloaded_user_ids
        for uid in sorted(candidate_ids):
            page_count = self.get_page_count_for_user(uid)
            host_tokens = self.host_token_count(uid)
            if page_count == 0 and host_tokens == 0 and uid not in self._offloaded_user_ids:
                continue
            footprint = max(1, page_count) * page_bytes
            manifest = self.user_manifest(uid)
            hbm_end = int(manifest["hbm_start_token"]) + int(
                manifest["hbm_tokens"]
            )
            placement = set()
            if page_count > 0:
                placement.add(Placement.HBM)
            if host_tokens > 0 or uid in self._offloaded_user_ids:
                placement.add(Placement.HOST_DRAM)
            owner_device = None
            try:
                owner_device = str(self._kvcache.cache_table.device)
            except AttributeError:
                pass
            handles.append(StateHandle(
                state_type=StateType.SESSION_KV_USER,
                logical_key=f"kv:uid:{uid}",
                footprint_bytes=footprint,
                placement=placement,
                reconstructability=Reconstructability.RECOMPUTABLE,
                consistency_class="append_only_session",
                # Host prefix and HBM suffix are jointly authoritative.  Do
                # not pretend one tier is a complete replica.
                authoritative_placement=(
                    None if page_count > 0 and host_tokens > 0
                    else Placement.HBM if page_count > 0 else Placement.HOST_DRAM
                ),
                version=int(manifest["version"]),
                freshness_epoch=int(manifest["version"]),
                replica_versions={
                    Placement.HOST_DRAM: int(manifest["host_prefix_tokens"]),
                    Placement.HBM: int(hbm_end),
                },
                replica_freshness_epochs={
                    Placement.HOST_DRAM: int(manifest["host_prefix_tokens"]),
                    Placement.HBM: int(hbm_end),
                },
                staleness_tolerance_epochs=0,
                writeback_required=False,
                owner_device=(owner_device if page_count > 0 else None),
                transfer_cost_ms=footprint / 25_000_000.0,
                reconstruction_cost_ms=footprint / 25_000_000.0,
                expected_reuse_window=1,
                dependencies=(f"session:{uid}", "model:hstu"),
                metadata={
                    "page_count": int(page_count),
                    "page_bytes": int(page_bytes),
                    "page_size_tokens": int(self.page_size_tokens),
                    "host_token_count": int(host_tokens),
                    "segments": manifest,
                    "source": "gpu_kv_cache_manager",
                },
            ))
        return handles
    
    def logical_kv_budget_bytes(self) -> int:
        return self.get_current_page_limit() * self._page_bytes()

    def physical_kv_cache_bytes(self) -> int:
        """Primary-page capacity only; retained for compatibility.

        Use :meth:`physical_hbm_bytes` for the complete device allocation.
        """
        return self.get_physical_page_count() * self._page_bytes()

    def physical_hbm_bytes(self) -> int:
        """Complete KV HBM allocation, including buffers and metadata."""
        measure = getattr(self._kvcache, "physical_hbm_bytes", None)
        if measure is None:
            raise RuntimeError(
                "AsyncHSTUKVCacheManager does not expose physical_hbm_bytes(); "
                "cannot validate a physical HotState budget"
            )
        return int(measure())

    def physical_hbm_breakdown(self) -> dict:
        measure = getattr(self._kvcache, "physical_hbm_breakdown", None)
        if measure is None:
            return {"physical_hbm_bytes": self.physical_hbm_bytes()}
        return dict(measure())

    def actual_resident_kv_bytes(self) -> int:
        return self.get_resident_page_count() * self._page_bytes()

    def total_hbm_bytes(self) -> int:
        # Historical callers use this as the policy budget.  Physical HBM is
        # exposed separately so a logical page limit cannot be mistaken for
        # the allocated tensor footprint.
        return self.logical_kv_budget_bytes()
