"""CPU checks for configurable HotState policy contracts."""

import sys
from pathlib import Path

HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))
EXAMPLES_ROOT = HSTU_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

import pytest

from modules.hotstate.policy import (
    InferenceBudget,
    InferencePolicy,
    PolicyConfig,
    ValueInputs,
    WeightedValue,
)
from modules.hotstate.state_handle import (
    Placement,
    ScoredHandle,
    StateHandle,
    StateType,
)


def _scored(key, state_type, footprint, score, *, user_id=None, parent=None):
    metadata = {"policy_candidate": True}
    if user_id is not None:
        metadata["user_id"] = user_id
    if parent is not None:
        metadata["retain_after"] = parent
    handle = StateHandle(
        state_type=state_type,
        logical_key=key,
        footprint_bytes=footprint,
        placement={Placement.HBM},
        metadata=metadata,
    )
    return ScoredHandle(
        handle=handle,
        score=score,
        net_benefit_ms=score * footprint,
        value_density_ms_per_byte=score,
    )


def test_policy_configuration_is_validated_and_reproducible():
    config = PolicyConfig(benefit_weight=2.0, kv_granularity="user")
    assert config.snapshot()["config_id"] == PolicyConfig(
        benefit_weight=2.0, kv_granularity="user"
    ).snapshot()["config_id"]
    assert WeightedValue(config)(ValueInputs(3.0, 1.0, 0.5, 0.5)) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="initial_embedding_ratio"):
        PolicyConfig(initial_embedding_ratio=1.5)


def test_policy_reserves_current_user_and_only_selects_contiguous_kv_prefix():
    config = PolicyConfig()
    policy = InferencePolicy(config)
    rows = [
        _scored("kv:uid:7:hbm_page:0:32", StateType.SESSION_KV_SEGMENT, 4,
                0.001, user_id=7),
        _scored("kv:uid:7:hbm_page:32:64", StateType.SESSION_KV_SEGMENT, 4,
                0.10, user_id=7, parent="kv:uid:7:hbm_page:0:32"),
        _scored("emb:item:4", StateType.EMBEDDING_HOT_ROWS, 4, 0.02),
    ]
    budget = InferenceBudget(12, 8, 8, 12)

    selected = policy.select(rows, budget, protected_users=[7])

    selected_keys = {handle.logical_key for handle in selected}
    assert "kv:uid:7:hbm_page:0:32" in selected_keys
    assert "kv:uid:7:hbm_page:32:64" in selected_keys


def test_static_partition_uses_configurable_ratio():
    policy = InferencePolicy(PolicyConfig(selection="static_partition", static_embedding_ratio=0.25))
    rows = [
        _scored("emb:item:1", StateType.EMBEDDING_HOT_ROWS, 4, 1.0),
        _scored("kv:uid:1:hbm_page:0:32", StateType.SESSION_KV_SEGMENT, 4, 1.0,
                user_id=1),
    ]
    selected = policy.select(rows, InferenceBudget(8, 8, 8, 8))
    assert {handle.logical_key for handle in selected} == {"kv:uid:1:hbm_page:0:32"}
