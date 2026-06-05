import math
from collections import defaultdict
from modules.hotstate.state_handle import StateType, Placement


class TransferScheduler:
    """Dependency-aware transfer execution with priority ordering.

    Inlines the dataset schedule prediction (no separate TransferPredictor).
    Submits onload/offload commands sorted by reuse urgency so the most
    critical transfers execute first within each batch epoch.

    V3-ready: when C++ stream-group API arrives, replace _submit_offload
    and _submit_onload with submit_transfer() calls using group/priority args.
    """

    HORIZON = 16              # batches to look ahead for pre-planning
    NUM_USERS = 8
    INCREMENT = 64            # history step size per seqlen group
    PAGE_SIZE = 32            # tokens per KV page
    BATCH_MS = 4.0            # estimated mean per-batch latency
    PCIE_BYTES_PER_MS = 25_000_000
    # Per-page transfer size: page_size × 2(K+V) × heads × head_dim × bytes
    BYTES_PER_PAGE = 32 * 2 * 4 * 128 * 2  # = 65,536 (2 bytes bf16)

    def __init__(self, kv_adapter, directory, value_engine):
        self.kv = kv_adapter
        self.dir = directory
        self.ve = value_engine
        self._pending = {}          # transfer_id → (uid, direction, epoch)
        self._next_id = 0
        self._bytes_per_epoch = defaultdict(int)  # epoch → scheduled bytes

    # Public API

    def plan_and_submit(self, current_batch, evicted_keys, scoring_map, epoch):
        """Plan onloads for upcoming users + submit offloads for evicted users."""
        # Phase 1: Schedule onloads for upcoming users
        for offset in range(self.HORIZON):
            uid, hist_len = self._predict_user_at(current_batch, offset)
            if uid in evicted_keys:
                continue   # user being evicted — don't load pages for them

            pages_needed = self._pages_for_history(hist_len)
            current_pages = self.kv.get_page_count_for_user(uid)
            shortfall = pages_needed - current_pages
            if shortfall <= 0:
                continue

            transfer_bytes = shortfall * self.BYTES_PER_PAGE * self.kv.num_layers
            if self._bytes_per_epoch[epoch] + transfer_bytes > self._max_per_epoch():
                continue   # bandwidth saturated this epoch

            priority = math.exp(-0.5 * offset)
            stream_group = 0 if offset <= 1 else 1

            self._submit_onload(uid, shortfall, priority, stream_group, epoch)
            self._bytes_per_epoch[epoch] += transfer_bytes

        # Phase 2: Submit offloads for evicted users
        for key in evicted_keys:
            if not key.startswith("kv:uid:"):
                continue
            uid = int(key.split(":")[-1])
            pages = self.kv.get_page_count_for_user(uid)
            if pages == 0:
                continue
            priority = 0.05   # background priority
            self._submit_offload(uid, priority, epoch)

        # Phase 3: Cancel stale onloads for evicted users
        self.cancel_stale(evicted_keys)

    def poll_completions(self, epoch):
        """Check in-progress offloads for completion. Returns list of completed keys."""
        completed = []
        stale = []
        for tid, (uid, direction, start_epoch) in list(self._pending.items()):
            if direction != "offload":
                continue   # onloads complete synchronously (prepare_kvcache)
            if not self.kv._gpu_mgr.is_busy_offloading():
                self.dir.mark_complete(f"kv:uid:{uid}")
                completed.append((tid, f"kv:uid:{uid}"))
                stale.append(tid)
        for tid in stale:
            del self._pending[tid]
        return [key for _, key in completed]

    def cancel_stale(self, evicted_keys):
        """Cancel pending onloads for users that are now evicted."""
        for tid, (uid, direction, _) in list(self._pending.items()):
            key = f"kv:uid:{uid}"
            if direction == "onload" and key in evicted_keys:
                del self._pending[tid]
                self.dir.mark_complete(key)

    # Internal: prediction

    def _predict_user_at(self, current_batch, offset):
        batch = current_batch + offset
        uid = batch % self.NUM_USERS
        seqlen_step = batch // self.NUM_USERS
        hist_len = self.INCREMENT + seqlen_step * self.INCREMENT
        return uid, hist_len

    def _pages_for_history(self, hist_len):
        return math.ceil(hist_len * 2 / self.PAGE_SIZE)

    def _max_per_epoch(self):
        return self.BATCH_MS * self.PCIE_BYTES_PER_MS

    # Internal: submission (P0 API today, V3 API later)

    def _submit_onload(self, uid, num_pages, priority, stream_group, epoch):
        """V1/V2: grow budget so pages are allocatable. 
           V3: submit_transfer(uid, ONLOAD, group, priority, num_pages)."""
        current_limit = self.kv.get_current_page_limit()
        new_limit = min(current_limit + num_pages,
                        self.kv._kvcache.num_primary_cache_pages)
        if new_limit > current_limit:
            self.kv.set_page_limit(new_limit)
        tid = self._next_id
        self._next_id += 1
        self._pending[tid] = (uid, "onload", epoch)
        self.dir.mark_in_flight(f"kv:uid:{uid}", "onload", Placement.HBM, epoch)

    def _submit_offload(self, uid, priority, epoch):
        """V1/V2: call evict_user. 
           V3: submit_transfer(uid, OFFLOAD, group=2, priority, num_pages)."""
        self.kv.evict_user(uid)
        tid = self._next_id
        self._next_id += 1
        self._pending[tid] = (uid, "offload", epoch)
        self.dir.mark_in_flight(f"kv:uid:{uid}", "offload", Placement.HOST_DRAM, epoch)
