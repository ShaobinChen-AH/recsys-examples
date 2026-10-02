"""CPU contract tests for HotState's inference consistency semantics."""

import sys
from pathlib import Path


HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))
EXAMPLES_ROOT = HSTU_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import pytest

from modules.hotstate.global_directory import GlobalDirectory
from modules.hotstate.kv_adapter import KVAdapter
from modules.hotstate.state_handle import (
    ConsistencyError,
    Placement,
    StateHandle,
    StateType,
)


def test_directory_rejects_stale_reads_and_preserves_inflight_replica():
    directory = GlobalDirectory()
    handle = StateHandle(
        state_type=StateType.EMBEDDING_HOT_ROWS,
        logical_key="emb:item:1",
        footprint_bytes=4,
        placement={Placement.HOST_DRAM},
        authoritative_placement=Placement.HOST_DRAM,
        version=3,
        consistency_class="immutable_checkpoint",
    )
    directory.register_handle(handle)
    directory.begin_transfer(
        handle.logical_key,
        "admission",
        Placement.HBM,
        1,
        source=Placement.HOST_DRAM,
        transfer_id="t1",
    )
    # A stale observation cannot erase the source while the copy is active.
    observed = StateHandle(
        state_type=handle.state_type,
        logical_key=handle.logical_key,
        footprint_bytes=4,
        placement=set(),
        version=3,
        consistency_class="immutable_checkpoint",
    )
    entry = directory.observe_handle(observed, epoch=2)
    assert Placement.HOST_DRAM in entry.placements
    with pytest.raises(ConsistencyError):
        directory.validate_read(handle.logical_key, Placement.HOST_DRAM)

    directory.complete_transfer(
        handle.logical_key,
        target=Placement.HBM,
        epoch=3,
        transfer_id="t1",
    )
    directory.validate_read(handle.logical_key, Placement.HBM)


def test_split_append_only_kv_requires_a_contiguous_segment():
    directory = GlobalDirectory()
    entry = directory.register(
        "kv:uid:4",
        Placement.HBM,
        placements={Placement.HOST_DRAM, Placement.HBM},
        authoritative=False,
        version=128,
        metadata={
            "segments": {
                "append_only": True,
                "host_prefix_tokens": 64,
                "hbm_start_token": 64,
                "hbm_tokens": 64,
                "logical_length": 128,
            }
        },
    )
    directory.validate_read("kv:uid:4", Placement.HBM, required_version=128)
    with pytest.raises(ConsistencyError):
        directory.validate_read("kv:uid:4", Placement.HOST_DRAM, required_version=128)

    handle = StateHandle(
        state_type=StateType.SESSION_KV_USER,
        logical_key="kv:uid:4",
        footprint_bytes=1,
        placement={Placement.HOST_DRAM, Placement.HBM},
        version=128,
        replica_versions={Placement.HOST_DRAM: 64, Placement.HBM: 128},
        consistency_class="append_only_session",
    )
    with pytest.raises(ConsistencyError):
        handle.validate_read(Placement.HOST_DRAM, required_version=128)


class _FrozenGPU:
    def __init__(self):
        self.frozen = set()
        self.pages = {7: 2}

    def get_user_page_count(self, uid):
        return self.pages.get(int(uid), 0)

    def is_user_offload_frozen(self, uid):
        return int(uid) in self.frozen

    def evict_if_present(self, uid):
        if int(uid) in self.frozen:
            return False
        return self.pages.pop(int(uid), 0) > 0


class _KV:
    num_primary_cache_pages = 8
    max_num_sequences = 8
    page_size = 32
    num_layers = 1
    num_heads = 1
    head_dim = 1
    gpu_kvcache_mgr = _FrozenGPU()

    class _Host:
        def get_kvdata_length(self, uid):
            return 64

    host_kv_mgr = _Host()


def test_kv_eviction_is_blocked_for_inflight_native_owner():
    adapter = KVAdapter(_KV())
    adapter._inflight_users.add(7)
    assert not adapter.evict_user(7)
    adapter._inflight_users.clear()
    adapter._gpu_mgr.frozen.add(7)
    assert not adapter.evict_user(7)
    adapter._gpu_mgr.frozen.clear()
    assert adapter.evict_user(7)
