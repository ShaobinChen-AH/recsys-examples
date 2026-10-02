import sys
from pathlib import Path


HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))
EXAMPLES_ROOT = HSTU_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from modules.hotstate.demand_signal import DemandSignal  # noqa: E402
from modules.hotstate.online_cost_model import (  # noqa: E402
    BatchCostObservation,
    OnlineSystemCostModel,
)
from modules.hotstate.state_handle import (  # noqa: E402
    Placement,
    Reconstructability,
    StateHandle,
    StateType,
)
from modules.hotstate.value_engine import ValueEngine  # noqa: E402


def _observation(latency_ms, embedding_misses=0, kv_tokens=0):
    return BatchCostObservation(
        latency_ms=latency_ms,
        history_tokens=1000,
        num_candidates=100,
        embedding_requests=100,
        embedding_misses=embedding_misses,
        kv_miss_tokens=kv_tokens,
        concurrent_transfer_bytes=0,
        transfer_queue_depth=0,
        hbm_pressure=0.5,
        host_memory_pressure=0.2,
    )


def test_batch_observations_update_measured_miss_costs():
    model = OnlineSystemCostModel()
    initial_embedding_cost = model.embedding_miss_ms(100)
    for _ in range(24):
        model.observe_batch(_observation(5.0, embedding_misses=0))
        model.observe_batch(_observation(25.0, embedding_misses=100))

    assert model.batch_model.samples == 48
    assert model.confidence == 1.0
    assert model.embedding_miss_ms(100) != initial_embedding_cost
    assert model.embedding_miss_ms(100) > 0.0
    snapshot = model.snapshot()
    assert snapshot["calibration_state"] == "online"
    assert snapshot["last_absolute_error_ms"] >= 0.0


def test_transfer_observations_replace_fixed_bandwidth_assumption():
    model = OnlineSystemCostModel()
    one_mib = 1024 * 1024
    initial = model.estimate_transfer_ms(
        state_class="embedding",
        direction="admission",
        bytes_moved=one_mib,
        queue_depth=0,
    )
    for _ in range(20):
        model.observe_transfer(
            state_class="embedding",
            direction="admission",
            bytes_moved=one_mib,
            elapsed_ms=0.20,
            queue_depth=0,
        )
    learned = model.estimate_transfer_ms(
        state_class="embedding",
        direction="admission",
        bytes_moved=one_mib,
        queue_depth=0,
    )
    assert learned > initial
    assert abs(learned - 0.20) < 0.05
    assert model.snapshot()["transfer_models"]["embedding:admission"]["samples"] == 20


def test_allocation_frontier_creates_real_opportunity_cost():
    model = OnlineSystemCostModel()
    model.observe_allocation(
        candidate_densities_ms_per_byte=[0.010, 0.008, 0.004],
        selected_densities_ms_per_byte=[0.010, 0.008],
        used_bytes=100,
        budget_bytes=100,
    )
    assert model.shadow_price_ms_per_byte > 0.0
    assert model.opportunity_cost_ms(10) > 0.0


def test_value_engine_uses_observed_reuse_not_fixed_user_cycle():
    engine = ValueEngine(total_hbm_bytes=1024)
    for epoch in (1, 3, 5, 7):
        engine.record_access("kv:uid:7", epoch, StateType.SESSION_KV_USER)
    handle = StateHandle(
        state_type=StateType.SESSION_KV_USER,
        logical_key="kv:uid:7",
        footprint_bytes=128,
        placement={Placement.HBM},
        reconstructability=Reconstructability.RECOMPUTABLE,
        metadata={"page_bytes": 128, "page_count": 1, "page_size_tokens": 32},
    )
    demand = DemandSignal(
        current_user_id=0,
        history_length=64,
        num_candidates=100,
        epoch=8,
        item_indices=[],
        item_sequence=[],
    )
    scored = engine.compute_scores([handle], demand)[0]
    assert scored.reuse_probability > 0.5
    assert scored.decision_reason.startswith("online_cost(")


def test_value_engine_reports_learned_cost_snapshot():
    engine = ValueEngine(total_hbm_bytes=1024)
    engine.observe_batch_cost(
        latency_ms=8.0,
        history_tokens=64,
        num_candidates=10,
        embedding_requests=4,
        embedding_misses=2,
        kv_miss_tokens=32,
        concurrent_transfer_bytes=0,
        transfer_queue_depth=0,
        hbm_pressure=0.5,
        host_memory_pressure=0.1,
    )
    snapshot = engine.cost_model_snapshot()
    assert snapshot["batch_samples"] == 1
    assert "embedding_miss_hundreds" in snapshot["batch_coefficients"]
    assert "kv_uncached_k_tokens" in snapshot["batch_coefficients"]
