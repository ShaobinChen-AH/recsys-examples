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
    StateHandle,
    StateType,
    TransferState,
)
from modules.hotstate.state_registry import StateRegistry  # noqa: E402


def _handle():
    return StateHandle(
        state_type=StateType.EMBEDDING_HOT_ROWS,
        logical_key="emb:item:7",
        footprint_bytes=128,
        placement={Placement.HBM, Placement.HOST_DRAM},
        reconstructability=Reconstructability.REFETCHABLE,
        version=3,
        freshness_epoch=10,
        authoritative_placement=Placement.HOST_DRAM,
        staleness_tolerance_epochs=1,
        ttl_epochs=4,
        writeback_required=True,
        transfer_cost_ms=0.5,
        reconstruction_cost_ms=2.0,
        expected_reuse_window=8,
        dependencies=("table:0",),
    )


def test_state_handle_tracks_replica_versions_staleness_and_expiry():
    handle = _handle()
    handle.add_replica(Placement.HBM, version=2, freshness_epoch=8)
    assert handle.is_stale(Placement.HBM)
    handle.mark_access(10)
    assert not handle.is_expired(14)
    assert handle.is_expired(15)
    handle.set_authoritative(Placement.HBM, version=4, freshness_epoch=11)
    assert handle.authoritative_placement == Placement.HBM
    assert handle.version == 4


def test_registry_records_failed_and_successful_transitions():
    registry = StateRegistry()
    handle = _handle()
    registry.register(handle)
    registry.record_transition(
        handle.logical_key,
        action="admit",
        source=Placement.HOST_DRAM,
        target=Placement.HBM,
        transfer_id="t1",
        success=True,
        authoritative=True,
        version=4,
        freshness_epoch=11,
    )
    assert handle.transfer_state == TransferState.COMPLETE
    assert handle.authoritative_placement == Placement.HBM
    registry.record_transition(
        handle.logical_key,
        action="evict",
        source=Placement.HBM,
        target=Placement.HOST_DRAM,
        transfer_id="t2",
        success=False,
        error="copy engine unavailable",
    )
    assert Placement.HBM in handle.placement
    assert handle.transfer_state == TransferState.FAILED
    assert registry.history()[-1]["error"] == "copy engine unavailable"


def test_directory_tracks_replicas_authority_and_recovery():
    directory = GlobalDirectory()
    handle = _handle()
    directory.register_handle(handle)
    transfer_id = directory.begin_transfer(
        handle.logical_key,
        "admission",
        Placement.HBM,
        epoch=11,
        source=Placement.HOST_DRAM,
        bytes=handle.footprint_bytes,
    )
    assert directory.is_in_flight(handle.logical_key)
    assert directory.get(handle.logical_key).transfer_id == transfer_id
    directory.complete_transfer(
        handle.logical_key,
        target=Placement.HBM,
        epoch=12,
        authoritative=True,
        version=4,
        freshness_epoch=12,
    )
    entry = directory.get(handle.logical_key)
    assert entry.authoritative_placement == Placement.HBM
    assert Placement.HBM in entry.placements
    assert Placement.HOST_DRAM not in entry.placements

    directory.begin_transfer(
        handle.logical_key,
        "eviction",
        Placement.HOST_DRAM,
        epoch=13,
        source=Placement.HBM,
    )
    directory.fail_transfer(handle.logical_key, "host write failed", epoch=13)
    entry = directory.get(handle.logical_key)
    assert entry.transfer_state == TransferState.FAILED
    assert entry.last_error == "host write failed"
    assert entry.recovery_source == Placement.HBM
    snapshot = directory.snapshot()[0]
    assert snapshot["authoritative_placement"] == "HBM"
    assert directory.history()[-1]["event"] == "transfer_failed"


def test_directory_rejects_unknown_exact_transfer_completion():
    directory = GlobalDirectory()
    handle = _handle()
    directory.register_handle(handle)
    try:
        directory.complete_transfer(
            handle.logical_key,
            target=Placement.HBM,
            transfer_id="not-running",
        )
    except KeyError:
        pass
    else:
        raise AssertionError("unknown exact transfer completion must fail closed")
