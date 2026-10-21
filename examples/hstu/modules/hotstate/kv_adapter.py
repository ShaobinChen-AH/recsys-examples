import threading
from typing import Iterable, List, Optional, Set

from modules.hotstate.state_handle import (
    ConsistencyError,
    StateHandle, StateType, Placement, Reconstructability
)


class KVAdapter:
    """Expose append-only host prefixes and HBM suffix segments to HotState.

    The native runtime evicts a user's contiguous HBM suffix; segment handles
    mirror that physical granularity and never claim arbitrary-page eviction.
    """

    def __init__(self, async_kvcache_manager, granularity="segment"):
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
        self.granularity = granularity
        self._exported_segments = {}

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
        interval = getattr(self._gpu_mgr, "get_user_cache_interval", None)
        if callable(interval):
            start, end, page_count = (int(v) for v in interval(uid))
            hbm_start = start if page_count else host_prefix
            hbm_tokens = max(0, end - start) if page_count else 0
            total = max(host_prefix, end)
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
            "logical_length": int(max(total, hbm_start + hbm_tokens, host_prefix)),
            "page_count": page_count,
            "pending_offload_tokens": max(0, pending_tokens),
            "append_only": True,
            "version": int(max(total, hbm_start + hbm_tokens, host_prefix)),
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

    def capacity_snapshot(self) -> dict:
        page_bytes = int(self._page_bytes())
        physical_pages = int(self.get_physical_page_count())
        active_pages = int(self.get_current_page_limit())
        return {
            "physical_page_capacity_bytes": physical_pages * page_bytes,
            "physical_pages": physical_pages,
            "active_page_limit_bytes": active_pages * page_bytes,
            "active_page_limit_pages": active_pages,
            "resident_page_bytes": int(self.actual_resident_kv_bytes()),
            "physical_pool_resize_supported": False,
            "logical_limit_supported": hasattr(self._gpu_mgr, "set_active_page_limit"),
            "segment_release_supported": callable(getattr(self._gpu_mgr, "evict_user_suffix_pages", None)),
            "requested_granularity": self.granularity,
        }

    def get_page_pressure(self) -> dict:
        """Return actual vs allocated KV page metrics for constraint binding analysis."""
        resident = self.get_resident_page_count()
        limit = self.get_current_page_limit()
        physical = self.get_physical_page_count()
        pressure = (resident / limit * 100) if limit > 0 else 0
        return {
            "resident_pages": resident,
            "page_limit": limit,
            "physical_pages": physical,
            "pressure_percent": pressure,
            "is_constrained": resident >= (limit * 0.95),  # 95% threshold
        }

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

    @staticmethod
    def user_id_from_key(logical_key: str):
        """Parse both legacy user keys and explicit KV segment keys."""
        parts = str(logical_key).split(":")
        if len(parts) < 3 or parts[0] != "kv" or parts[1] != "uid":
            return None
        try:
            return int(parts[2])
        except ValueError:
            return None

    def evict_hbm_suffix(self, uid: int, num_pages: Optional[int] = None, expected=None) -> bool:
        """Release the newest contiguous KV pages, keeping the older prefix."""
        uid = int(uid)
        with self._state_lock:
            if uid in self._inflight_users:
                return False
        frozen = getattr(self._gpu_mgr, "is_user_offload_frozen", None)
        if callable(frozen) and bool(frozen(uid)):
            return False
        current_pages = int(self.get_page_count_for_user(uid))
        if current_pages <= 0:
            return False
        pages = current_pages if num_pages is None else int(num_pages)
        if pages <= 0 or pages > current_pages:
            return False
        release_suffix = getattr(self._gpu_mgr, "evict_user_suffix_pages", None)
        if callable(release_suffix):
            manifest = expected or self.user_manifest(uid)
            start = int(manifest["hbm_start_token"])
            end = start + int(manifest["hbm_tokens"])
            released = bool(release_suffix(uid, pages, start, end))
        elif pages == current_pages:
            released = bool(self._gpu_mgr.evict_if_present(uid))
        else:
            # Older native extensions only support whole-user release.
            return False
        if released:
            if self.host_token_count(uid) > 0:
                self._offloaded_user_ids.add(uid)
            else:
                self._offloaded_user_ids.discard(uid)
        return released

    def apply_eviction_plan(self, logical_keys: Iterable[str], protected_uids=()) -> List[str]:
        """Apply planned segment releases and return confirmed logical keys."""
        protected = {int(uid) for uid in protected_uids}
        applied: List[str] = []
        page_releases = {}
        whole_user_releases = []
        for logical_key in logical_keys:
            if not logical_key.startswith("kv:uid:"):
                continue
            uid = self.user_id_from_key(logical_key)
            if uid is None:
                continue
            if uid in protected:
                continue
            parts = str(logical_key).split(":")
            if len(parts) == 6 and parts[3] == "hbm_page":
                handle = self._exported_segments.get(logical_key)
                if handle is not None:
                    page_releases.setdefault(uid, []).append((handle.metadata["page_index"], logical_key))
            elif logical_key.endswith(":hbm_suffix"):
                whole_user_releases.append((uid, logical_key))
            elif len(parts) == 3:
                whole_user_releases.append((uid, logical_key))
        for uid, entries in page_releases.items():
            page_count = int(self.get_page_count_for_user(uid))
            indices = sorted({index for index, _key in entries})
            first = indices[0] if indices else page_count
            if indices != list(range(first, page_count)):
                continue
            expected = self._exported_segments[entries[0][1]].metadata["segments"]
            if self.evict_hbm_suffix(uid, page_count - first, expected=expected):
                applied.extend(key for _index, key in sorted(entries))
        for uid, logical_key in whole_user_releases:
            moved = self.evict_hbm_suffix(uid) if logical_key.endswith(":hbm_suffix") else self.evict_user(uid)
            if moved:
                applied.append(logical_key)
        return applied

    def export_handles(self) -> List[StateHandle]:
        """Export page-aligned contiguous host-prefix and HBM-suffix segments."""
        handles = []
        page_bytes = self._page_bytes()
        self._exported_segments = {}
        candidate_ids = set(self._candidate_user_ids()) | self._offloaded_user_ids
        for uid in sorted(candidate_ids):
            page_count = self.get_page_count_for_user(uid)
            host_tokens = self.host_token_count(uid)
            if page_count == 0 and host_tokens == 0 and uid not in self._offloaded_user_ids:
                continue
            manifest = self.user_manifest(uid)
            page_count = int(manifest["page_count"])
            hbm_end = int(manifest["hbm_start_token"]) + int(
                manifest["hbm_tokens"]
            )
            owner_device = None
            try:
                owner_device = str(self._kvcache.cache_table.device)
            except AttributeError:
                pass
            base_metadata = {
                "user_id": int(uid),
                "access_key": f"kv:uid:{uid}",
                "page_bytes": int(page_bytes),
                "page_size_tokens": int(self.page_size_tokens),
                "host_token_count": int(host_tokens),
                "segments": manifest,
                "source": "gpu_kv_cache_manager",
            }
            if host_tokens > 0:
                host_bytes = int(host_tokens) * int(self.token_bytes())
                handles.append(StateHandle(
                    state_type=StateType.SESSION_KV_SEGMENT,
                    logical_key=f"kv:uid:{uid}:host_prefix",
                    footprint_bytes=max(1, host_bytes),
                    placement={Placement.HOST_DRAM},
                    reconstructability=Reconstructability.RECOMPUTABLE,
                    consistency_class="append_only_session_host_prefix",
                    authoritative_placement=Placement.HOST_DRAM,
                    version=int(host_tokens),
                    freshness_epoch=int(host_tokens),
                    replica_versions={Placement.HOST_DRAM: int(host_tokens)},
                    replica_freshness_epochs={Placement.HOST_DRAM: int(host_tokens)},
                    staleness_tolerance_epochs=0,
                    transfer_cost_ms=host_bytes / 25_000_000.0,
                    reconstruction_cost_ms=host_bytes / 25_000_000.0,
                    expected_reuse_window=1,
                    dependencies=(f"session:{uid}", "model:hstu"),
                    metadata={
                        **base_metadata,
                        "segment_tier": "host",
                        "segment_start_token": 0,
                        "segment_end_token": int(host_tokens),
                        "token_count": int(host_tokens),
                        "policy_candidate": False,
                    },
                ))
            if page_count > 0:
                hbm_tokens = int(manifest["hbm_tokens"])
                hbm_start = int(manifest["hbm_start_token"])
                previous_key = None
                segmented = self.granularity == "segment" and callable(
                    getattr(self._gpu_mgr, "evict_user_suffix_pages", None)
                )
                for local_page_index in (range(int(page_count)) if segmented else (0,)):
                    segment_start = hbm_start + local_page_index * int(self.page_size_tokens)
                    segment_end = (min(hbm_start + hbm_tokens,
                                       segment_start + int(self.page_size_tokens))
                                   if segmented else hbm_start + hbm_tokens)
                    logical_key = (f"kv:uid:{uid}:hbm_page:{segment_start}:{segment_end}"
                                   if segmented else f"kv:uid:{uid}:hbm_suffix")
                    segment_tokens = max(0, segment_end - segment_start)
                    handle = StateHandle(
                        state_type=StateType.SESSION_KV_SEGMENT,
                        logical_key=logical_key,
                        footprint_bytes=int(page_bytes) * (1 if segmented else page_count),
                        placement={Placement.HBM},
                        reconstructability=Reconstructability.RECOMPUTABLE,
                        consistency_class="append_only_session_hbm_page",
                        authoritative_placement=Placement.HBM,
                        version=int(segment_end),
                        freshness_epoch=int(segment_end),
                        replica_versions={Placement.HBM: int(segment_end)},
                        replica_freshness_epochs={Placement.HBM: int(segment_end)},
                        staleness_tolerance_epochs=0,
                        owner_device=owner_device,
                        transfer_cost_ms=page_bytes / 25_000_000.0,
                        reconstruction_cost_ms=segment_tokens * self.token_bytes() / 25_000_000.0,
                        expected_reuse_window=1,
                        dependencies=(f"session:{uid}", "model:hstu"),
                        metadata={
                            **base_metadata,
                            "segment_tier": "hbm",
                            "segment_start_token": segment_start,
                            "segment_end_token": segment_end,
                            "token_count": segment_tokens,
                            "page_index": local_page_index,
                            "page_count": 1 if segmented else page_count,
                            "pinned": bool(getattr(self._gpu_mgr, "is_user_offload_frozen", lambda _uid: False)(uid)),
                            "policy_candidate": True,
                            "retain_after": previous_key,
                        },
                    )
                    handles.append(handle)
                    previous_key = logical_key
        self._exported_segments = {h.logical_key: h for h in handles}
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
