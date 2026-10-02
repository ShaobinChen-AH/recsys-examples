import sys
from pathlib import Path


HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))
EXAMPLES_ROOT = HSTU_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from modules.hotstate.global_directory import GlobalDirectory  # noqa: E402
from modules.hotstate.state_handle import (  # noqa: E402
    Placement,
    Reconstructability,
    ScoredHandle,
    StateHandle,
    StateType,
)
from modules.hotstate.state_registry import StateRegistry  # noqa: E402
from modules.hotstate.transfer_scheduler import (  # noqa: E402
    TransferResource,
    TransferScheduler,
)
from modules.hotstate.value_engine import ValueEngine  # noqa: E402


class _EmbeddingBackend:
    def __init__(self):
        self.calls = []

    def apply_residency_plan(self, admitted, evicted):
        self.calls.append((list(admitted), list(evicted)))
        return list(admitted), list(evicted)


class _KVBackend:
    def __init__(self):
        self.pages = {7: 2}
        self.host_tokens = {7: 64, 9: 96}
        self.eviction_calls = []
        self.stage_calls = []
        self.offload = {
            "generation": 0,
            "accepted": False,
            "tokens_by_user": {},
            "submitted_at": None,
            "completion_target": None,
        }
        self.completed = 0

    def _page_bytes(self):
        return 128

    def token_bytes(self):
        return 4

    def get_page_count_for_user(self, uid):
        return self.pages.get(int(uid), 0)

    def host_token_count(self, uid):
        return self.host_tokens.get(int(uid), 0)

    def has_host_copy(self, uid):
        return self.host_token_count(uid) > 0

    def apply_eviction_plan(self, keys, protected_uids=()):
        protected = {int(uid) for uid in protected_uids}
        self.eviction_calls.append((list(keys), protected))
        applied = []
        for key in keys:
            uid = int(key.rsplit(":", 1)[-1])
            if uid not in protected and self.pages.pop(uid, 0) > 0:
                applied.append(key)
        return applied

    def stage_request(self, user_ids, history_lengths):
        self.stage_calls.append((list(user_ids), list(history_lengths)))
        return {"generation": len(self.stage_calls)}

    def native_offload_snapshot(self):
        return dict(self.offload)

    def completed_offload_count(self):
        return self.completed

    def is_busy_offloading(self):
        return self.completed < int(self.offload.get("completion_target") or 0)


def _handle(key, state_type, placement, footprint=128):
    return StateHandle(
        state_type=state_type,
        logical_key=key,
        footprint_bytes=footprint,
        placement={placement},
        reconstructability=(
            Reconstructability.REFETCHABLE
            if state_type != StateType.SESSION_KV_USER
            else Reconstructability.RECOMPUTABLE
        ),
        authoritative_placement=placement,
    )


def _scored(handle, density, net):
    return ScoredHandle(
        handle=handle,
        score=density,
        value_density_ms_per_byte=density,
        net_benefit_ms=net,
    )


def _scheduler():
    directory = GlobalDirectory()
    registry = StateRegistry()
    embedding = _EmbeddingBackend()
    kv = _KVBackend()
    handles = [
        _handle("emb:item:1", StateType.EMBEDDING_HOT_ROWS, Placement.HOST_DRAM),
        _handle("emb:item:2", StateType.EMBEDDING_HOT_ROWS, Placement.HBM),
        _handle("kv:uid:7", StateType.SESSION_KV_USER, Placement.HBM, 256),
        _handle("kv:uid:9", StateType.SESSION_KV_USER, Placement.HOST_DRAM, 384),
    ]
    for handle in handles:
        registry.register(handle)
        directory.register_handle(handle)
    scheduler = TransferScheduler(
        kv,
        directory,
        ValueEngine(4096),
        embedding_adapter=embedding,
        registry=registry,
    )
    return scheduler, embedding, kv, directory, registry, handles


def test_scheduler_executes_real_backends_and_closes_lifecycle():
    scheduler, embedding, kv, directory, _registry, handles = _scheduler()
    scores = {
        handle.logical_key: _scored(handle, density=0.01, net=2.0)
        for handle in handles
    }
    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[1],
        evicted_embedding_keys=[2],
        evicted_kv_keys=["kv:uid:7"],
        scoring_map=scores,
        protected_uids=[],
        epoch=3,
        row_size_bytes=128,
    )

    assert embedding.calls == [([], [2]), ([1], [])]
    assert kv.eviction_calls == [(["kv:uid:7"], set())]
    assert result.admitted_embedding_keys == [1]
    assert result.evicted_embedding_keys == [2]
    assert result.released_kv_keys == ["kv:uid:7"]
    assert Placement.HBM in directory.get("emb:item:1").placements
    assert Placement.HOST_DRAM in directory.get("emb:item:2").placements
    assert Placement.HBM not in directory.get("kv:uid:7").placements
    assert scheduler.pending_count() == 0
    assert directory.pending_count() == 0
    snapshot = scheduler.snapshot()
    assert snapshot["counters"]["completed"] == 3
    assert snapshot["counters"]["failed"] == 0
    release = next(
        record
        for record in snapshot["history_tail"]
        if record["logical_key"] == "kv:uid:7"
    )
    assert release["bytes"] == 0
    assert release["metadata"]["released_footprint_bytes"] == 256


def test_scheduler_marks_unconfirmed_backend_actions_failed():
    scheduler, embedding, _kv, _directory, _registry, handles = _scheduler()
    embedding.apply_residency_plan = lambda admitted, evicted: ([], [])
    scores = {handles[0].logical_key: _scored(handles[0], 0.01, 1.0)}
    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[1],
        evicted_embedding_keys=[],
        evicted_kv_keys=[],
        scoring_map=scores,
        protected_uids=[],
        epoch=1,
        row_size_bytes=128,
    )
    assert len(result.failed_ids) == 1
    assert scheduler.snapshot()["counters"]["failed"] == 1


def test_failed_prerequisite_defers_dependent_admission():
    scheduler, embedding, _kv, _directory, _registry, handles = _scheduler()
    embedding.apply_residency_plan = lambda admitted, evicted: ([], [])
    scores = {handle.logical_key: _scored(handle, 0.01, 1.0) for handle in handles}
    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[1],
        evicted_embedding_keys=[2],
        evicted_kv_keys=[],
        scoring_map=scores,
        protected_uids=[],
        epoch=1,
        row_size_bytes=128,
    )
    assert len(result.failed_ids) == 1
    assert len(result.deferred_ids) == 1
    assert result.admitted_embedding_keys == []
    deferred = next(
        record
        for record in scheduler.snapshot()["history_tail"]
        if record["status"] == "DEFERRED"
    )
    assert deferred["metadata"]["incomplete_dependencies"] == result.failed_ids


def test_backend_failure_isolated_from_other_resource_domain():
    scheduler, embedding, kv, _directory, _registry, handles = _scheduler()

    def fail_embedding(_admitted, _evicted):
        raise RuntimeError("embedding copy failed")

    embedding.apply_residency_plan = fail_embedding
    scores = {handle.logical_key: _scored(handle, 0.01, 1.0) for handle in handles}
    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[],
        evicted_embedding_keys=[2],
        evicted_kv_keys=["kv:uid:7"],
        scoring_map=scores,
        protected_uids=[],
        epoch=1,
        row_size_bytes=128,
    )
    assert result.backend_errors == {"embedding_eviction": "embedding copy failed"}
    assert result.released_kv_keys == ["kv:uid:7"]
    assert kv.eviction_calls == [(["kv:uid:7"], set())]
    assert len(result.failed_ids) == 1


def test_dependency_outcomes_outlive_bounded_trace_history():
    scheduler, embedding, _kv, _directory, _registry, handles = _scheduler()
    scheduler.max_history = 32
    scheduler._max_bytes_for_epoch = lambda: 1 << 30
    evictions = list(range(1000, 1064))
    scores = {handles[0].logical_key: _scored(handles[0], 0.01, 2.0)}
    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[1],
        evicted_embedding_keys=evictions,
        evicted_kv_keys=[],
        scoring_map=scores,
        protected_uids=[],
        epoch=1,
        row_size_bytes=128,
    )
    assert result.admitted_embedding_keys == [1]
    assert result.deferred_ids == []
    assert embedding.calls[-1] == ([1], [])


def test_directory_distinguishes_overlapping_same_key_transfers():
    scheduler, _embedding, _kv, directory, registry, _handles = _scheduler()
    first = scheduler._enqueue(
        logical_key="kv:uid:9",
        state_class="kv",
        direction="onload",
        source=Placement.HOST_DRAM,
        target=Placement.HBM,
        bytes_moved=64,
        priority=1.0,
        deadline_epoch=2,
        epoch=1,
        resource=TransferResource.PCIE_H2D,
    )
    second = scheduler._enqueue(
        logical_key="kv:uid:9",
        state_class="kv",
        direction="offload",
        source=Placement.HBM,
        target=Placement.HOST_DRAM,
        bytes_moved=64,
        priority=1.0,
        deadline_epoch=2,
        epoch=1,
        resource=TransferResource.PCIE_D2H,
    )
    scheduler._start(first, 1)
    scheduler._start(second, 1)
    assert directory.pending_count() == 2
    scheduler._complete(first, 2, keep_source_replica=True)
    assert directory.pending_count() == 1
    assert registry.get("kv:uid:9").transfer_id == second.transfer_id
    scheduler._complete(second, 2, keep_source_replica=True)
    assert directory.pending_count() == 0


def test_scheduler_defers_background_copy_under_learned_bandwidth_budget():
    scheduler, embedding, _kv, _directory, _registry, handles = _scheduler()
    scheduler._max_bytes_for_epoch = lambda: 128
    scores = {
        handle.logical_key: _scored(handle, density=0.01, net=2.0)
        for handle in handles
    }
    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[1],
        evicted_embedding_keys=[2],
        evicted_kv_keys=[],
        scoring_map=scores,
        protected_uids=[],
        epoch=2,
        row_size_bytes=128,
    )
    assert len(result.deferred_ids) == 1
    assert embedding.calls == [([1], [])]
    snapshot = scheduler.snapshot()
    assert snapshot["counters"]["deferred"] == 1
    assert snapshot["counters"]["completed"] == 1


def test_request_scoped_kv_stage_has_real_completion_lifecycle():
    scheduler, _embedding, kv, directory, _registry, handles = _scheduler()
    scores = {handle.logical_key: _scored(handle, 0.01, 2.0) for handle in handles}
    staged = scheduler.stage_kv_request(
        [9], [128], scoring_map=scores, epoch=4
    )
    assert kv.stage_calls == [([9], [128])]
    assert staged["onload_bytes"] == 96 * kv.token_bytes()
    assert scheduler.pending_count() == 1
    completed = scheduler.complete_staged_kv_request(
        staged["prepare_generation"], epoch=5
    )
    assert completed == ["kv:uid:9"]
    assert scheduler.pending_count() == 0
    assert Placement.HBM in directory.get("kv:uid:9").placements
    assert Placement.HOST_DRAM in directory.get("kv:uid:9").placements
    assert Placement.HBM in _registry.get("kv:uid:9").placement
    assert Placement.HOST_DRAM in _registry.get("kv:uid:9").placement


def test_value_scores_order_ready_transfers():
    scheduler, embedding, _kv, directory, registry, handles = _scheduler()
    second = _handle(
        "emb:item:3", StateType.EMBEDDING_HOT_ROWS, Placement.HOST_DRAM
    )
    registry.register(second)
    directory.register_handle(second)
    scores = {
        "emb:item:1": _scored(handles[0], density=0.01, net=1.0),
        "emb:item:3": _scored(second, density=0.01, net=10.0),
    }

    result = scheduler.execute_residency_plan(
        admitted_embedding_keys=[1, 3],
        evicted_embedding_keys=[],
        evicted_kv_keys=[],
        scoring_map=scores,
        protected_uids=[],
        epoch=3,
        row_size_bytes=128,
    )

    assert embedding.calls == [([3, 1], [])]
    assert result.admitted_embedding_keys == [3, 1]


def test_native_offload_waits_for_its_completion_sequence():
    scheduler, _embedding, kv, _directory, _registry, _handles = _scheduler()
    kv.offload = {
        "generation": 1,
        "accepted": True,
        "tokens_by_user": {7: 64},
        "submitted_at": 1.0,
        "completion_target": 3,
    }
    submitted = scheduler.observe_native_kv_activity(epoch=6)
    assert len(submitted["submitted_ids"]) == 1
    kv.completed = 2
    assert scheduler.poll_completions(epoch=7) == []
    assert scheduler.pending_count() == 1
    resource = scheduler.snapshot()["resources"]["PCIE_D2H"]
    assert resource["in_flight_count"] == 1
    assert resource["in_flight_bytes"] == 64 * kv.token_bytes()
    kv.completed = 3
    assert scheduler.poll_completions(epoch=8) == ["kv:uid:7"]
    assert scheduler.pending_count() == 0
    assert scheduler.snapshot()["counters"]["deadline_misses"] == 1


def test_scheduler_never_uses_page_limit_as_transfer():
    source = Path(
        HSTU_ROOT / "modules" / "hotstate" / "transfer_scheduler.py"
    ).read_text()
    assert "set_page_limit(" not in source
    assert "_predict_user_at" not in source
