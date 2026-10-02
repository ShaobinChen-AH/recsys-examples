import sys
from pathlib import Path


HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))
EXAMPLES_ROOT = HSTU_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from modules.hotstate.demand_signal import DemandSignal  # noqa: E402
from modules.hotstate.hot_set_manager import HotSetManager  # noqa: E402
from modules.hotstate.kv_adapter import KVAdapter  # noqa: E402
from modules.hotstate.state_handle import (  # noqa: E402
    Placement,
    Reconstructability,
    StateHandle,
    StateType,
)
from modules.hotstate.state_registry import StateRegistry  # noqa: E402
from modules.hotstate.global_directory import GlobalDirectory  # noqa: E402
from modules.hotstate.value_engine import ValueEngine  # noqa: E402


class _PlanningEmbedding:
    def row_size_bytes(self):
        return 4

    def hbm_capacity_keys(self):
        return 2

    def export_handles(self):
        return [
            StateHandle(
                state_type=StateType.EMBEDDING_HOT_ROWS,
                logical_key="emb:item:hot_rows",
                footprint_bytes=12,
                placement={Placement.HOST_DRAM},
                reconstructability=Reconstructability.REFETCHABLE,
            )
        ]


class _PlanningKV:
    page_size_tokens = 32

    def _page_bytes(self):
        return 4

    def get_current_page_limit(self):
        return 8

    def get_resident_page_count(self):
        return 2

    def get_physical_page_count(self):
        return 8

    def export_handles(self):
        return [
            StateHandle(
                state_type=StateType.SESSION_KV_USER,
                logical_key="kv:uid:1",
                footprint_bytes=4,
                placement={Placement.HBM},
                reconstructability=Reconstructability.RECOMPUTABLE,
            ),
            StateHandle(
                state_type=StateType.SESSION_KV_USER,
                logical_key="kv:uid:2",
                footprint_bytes=4,
                placement={Placement.HBM},
                reconstructability=Reconstructability.RECOMPUTABLE,
            ),
        ]


def test_hotset_emits_concrete_embedding_and_kv_actions():
    manager = HotSetManager(
        total_hbm_bytes=16,
        value_engine=ValueEngine(16),
        registry=StateRegistry(),
        directory=GlobalDirectory(),
        emb_adapter=_PlanningEmbedding(),
        kv_adapter=_PlanningKV(),
    )
    result = manager.run_epoch(
        0,
        DemandSignal(
            current_user_id=0,
            history_length=32,
            num_candidates=10,
            epoch=0,
            item_indices=[10, 11, 12],
            item_sequence=[10, 11, 12],
        ),
    )

    assert result.evicted_keys == []
    assert result.admitted_keys == []
    assert len(result.planned_embedding_keys) == 2
    assert "kv:uid:2" in result.planned_kv_eviction_keys


class _FakeGpuManager:
    def evict_if_present(self, uid):
        return uid == 3


class _FakeKVManager:
    num_primary_cache_pages = 8
    max_num_sequences = 8
    page_size = 32
    num_layers = 1
    num_heads = 1
    head_dim = 1
    gpu_kvcache_mgr = _FakeGpuManager()


def test_kv_adapter_reports_only_successful_evictions():
    adapter = KVAdapter(_FakeKVManager())
    assert adapter.apply_eviction_plan(
        ["kv:uid:2", "kv:uid:3", "bad-key"], protected_uids=[2]
    ) == ["kv:uid:3"]
