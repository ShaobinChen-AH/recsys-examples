import json
import sys
from pathlib import Path

import pytest


HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))

from benchmark.evidence import (  # noqa: E402
    EvidenceValidationError,
    compare_static_baseline,
    evidence_summary,
    validate_static_baseline_trace,
    validate_hotstate_trace,
)


def _hot_row(index, latency, embedding=20, kv=30, forward_latency=None):
    forward_latency = latency if forward_latency is None else forward_latency
    return {
        "record_type": "hotstate",
        "implementation_scope": "fixed_shared_hbm_inference",
        "latency_semantics": "inference_only",
        "batch_fingerprint": f"batch-{index}",
        "latency_ms": latency,
        "forward_latency_ms": forward_latency,
        "end_to_end_latency_ms": latency,
        "configured_state_budget_bytes": 100,
        "embedding_physical_hbm_bytes": embedding,
        "kv_physical_hbm_bytes": kv,
        "managed_state_physical_hbm_bytes": embedding + kv,
        "physical_admitted_keys": [],
        "physical_evicted_keys": [],
        "admission_strategy_enabled": False,
        "model_rebuild_count": 0,
        "online_pool_resize_observed": False,
        "benchmark_variant": "physical_envelope",
        "physical_pool_resize_applied": False,
    }


def _with_latency_fields(records):
    """Give static fixture rows explicit measurements for both intervals."""
    rows = []
    for record in records:
        row = dict(record)
        row.setdefault("forward_latency_ms", row["latency_ms"])
        row.setdefault("end_to_end_latency_ms", row["latency_ms"])
        rows.append(row)
    return rows


def test_fixed_envelope_trace_is_measured_without_online_pool_resize():
    report = validate_hotstate_trace([_hot_row(0, 4.0), _hot_row(1, 5.0)])

    assert report["physical_rows_validated"] == 2
    assert report["claims"]["fixed_shared_hbm_envelope"] == "measured"
    assert report["claims"]["online_cross_pool_resize"] == "not_demonstrated"


def test_self_reported_pool_resize_flag_is_not_evidence():
    row = _hot_row(0, 4.0)
    row["online_pool_resize_observed"] = True
    report = validate_hotstate_trace([row])

    assert report["physical_pool_resize_events"] == 0
    assert report["claims"]["online_cross_pool_resize"] == "not_demonstrated"


def test_trace_rejects_budget_or_accounting_violation():
    row = _hot_row(0, 4.0, embedding=80, kv=30)
    with pytest.raises(EvidenceValidationError, match="physical state budget"):
        validate_hotstate_trace([row])

    row = _hot_row(0, 4.0)
    row["managed_state_physical_hbm_bytes"] = 49
    with pytest.raises(EvidenceValidationError, match="inconsistent physical byte sum"):
        validate_hotstate_trace([row])


def test_evidence_requires_both_timings_and_end_to_end_not_below_forward():
    row = _hot_row(0, 4.0)
    del row["forward_latency_ms"]
    with pytest.raises(EvidenceValidationError, match="forward_latency_ms"):
        validate_hotstate_trace([row])

    row = _hot_row(0, 4.0, forward_latency=5.0)
    with pytest.raises(EvidenceValidationError, match="cannot be shorter"):
        validate_hotstate_trace([row])


def test_paired_static_comparison_uses_common_fingerprints():
    hot = [_hot_row(0, 4.0), _hot_row(1, 6.0)]
    static = [
        {
            "record_type": "static", "split": "static_30_70",
            "batch_fingerprint": "batch-0", "latency_ms": 5.0,
            "latency_semantics": "inference_only",
            "implementation_scope": "physical_static_shared_hbm_baseline",
            "state_budget_bytes": 100,
            "embedding_physical_hbm_bytes": 20,
            "kv_physical_hbm_bytes": 30,
            "managed_state_physical_hbm_bytes": 50,
        },
        {
            "record_type": "static", "split": "static_30_70",
            "batch_fingerprint": "batch-1", "latency_ms": 7.0,
            "latency_semantics": "inference_only",
            "implementation_scope": "physical_static_shared_hbm_baseline",
            "state_budget_bytes": 100,
            "embedding_physical_hbm_bytes": 20,
            "kv_physical_hbm_bytes": 30,
            "managed_state_physical_hbm_bytes": 50,
        },
        {
            "record_type": "static", "split": "static_70_30",
            "batch_fingerprint": "batch-0", "latency_ms": 7.0,
            "latency_semantics": "inference_only",
            "implementation_scope": "physical_static_shared_hbm_baseline",
            "state_budget_bytes": 100,
            "embedding_physical_hbm_bytes": 20,
            "kv_physical_hbm_bytes": 30,
            "managed_state_physical_hbm_bytes": 50,
        },
        {
            "record_type": "static", "split": "static_70_30",
            "batch_fingerprint": "batch-1", "latency_ms": 8.0,
            "latency_semantics": "inference_only",
            "implementation_scope": "physical_static_shared_hbm_baseline",
            "state_budget_bytes": 100,
            "embedding_physical_hbm_bytes": 20,
            "kv_physical_hbm_bytes": 30,
            "managed_state_physical_hbm_bytes": 50,
        },
    ]
    comparison = compare_static_baseline(hot, _with_latency_fields(static))

    assert comparison["paired"] is True
    assert comparison["best_static_split"] == "static_30_70"
    assert comparison["gain_vs_best_static_pct"] == pytest.approx(16.6666667)


def test_control_and_forward_latency_are_separate_but_pair_on_end_to_end():
    hot = [_hot_row(0, 11.0, forward_latency=2.0)]
    hot[0]["batch_fingerprint"] = "same-request"
    hot[0]["batch_idx"] = 0
    hot[0]["hotstate_control_ms"] = 9.0
    static = [{
        "record_type": "static",
        "split": "static_50_50",
        "batch_fingerprint": "same-request",
        "batch_idx": 0,
        "latency_ms": 10.0,
        "forward_latency_ms": 10.0,
        "end_to_end_latency_ms": 10.0,
        "latency_semantics": "inference_only",
        "implementation_scope": "physical_static_shared_hbm_baseline",
        "state_budget_bytes": 100,
        "embedding_physical_hbm_bytes": 20,
        "kv_physical_hbm_bytes": 30,
        "managed_state_physical_hbm_bytes": 50,
    }]

    comparison = compare_static_baseline(hot, static)

    assert comparison["paired"] is True
    assert comparison["static_splits"][0]["common_samples"] == 1
    assert comparison["comparison_metric"] == "end_to_end_latency_ms"
    assert comparison["gain_vs_best_static_pct"] == pytest.approx(-10.0)
    assert comparison["hotstate_forward_attribution_on_common_samples"][
        "mean_ms"
    ] == pytest.approx(2.0)
    assert comparison["best_static_forward_attribution_on_common_samples"][
        "mean_ms"
    ] == pytest.approx(10.0)


def test_evidence_summary_is_json_serializable():
    report = validate_hotstate_trace([_hot_row(0, 4.0)])
    payload = evidence_summary(report)
    json.dumps(payload)
    assert payload["record_type"] == "evidence_summary"
    assert "split fixed at model construction" in payload["claim_boundary"]
    assert "residency managed within those pools" in payload["claim_boundary"]
    assert payload["comparison_latency_metric"] == "end_to_end_latency_ms"
    assert payload["attribution_only_latency_metric"] == "forward_latency_ms"


def test_static_baseline_report_requires_physical_scope():
    row = {
        "record_type": "static",
        "split": "static_50_50",
        "batch_fingerprint": "batch-0",
        "latency_ms": 5.0,
        "forward_latency_ms": 5.0,
        "end_to_end_latency_ms": 5.0,
        "latency_semantics": "inference_only",
        "implementation_scope": "policy_estimate_only",
        "state_budget_bytes": 100,
        "embedding_physical_hbm_bytes": 20,
        "kv_physical_hbm_bytes": 30,
        "managed_state_physical_hbm_bytes": 50,
    }
    with pytest.raises(EvidenceValidationError, match="physical_static_shared_hbm_baseline"):
        validate_static_baseline_trace([row])


def test_admission_smoke_is_explicitly_non_physical():
    row = _hot_row(0, 4.0)
    row["benchmark_variant"] = "admission_smoke"
    row["admission_strategy_enabled"] = True
    report = validate_hotstate_trace([row], require_physical=False)

    assert report["implementation_scope"] == "admission_smoke_diagnostic"
    assert report["physical_rows_validated"] == 0
    assert report["claims"]["fixed_shared_hbm_envelope"] == "not_measured"


def test_physical_validator_cannot_be_downgraded_for_arbitrary_trace():
    row = _hot_row(0, 4.0)
    with pytest.raises(EvidenceValidationError, match="reserved for admission_smoke"):
        validate_hotstate_trace([row], require_physical=False)


def test_static_baseline_must_match_state_budget():
    hot = [_hot_row(0, 4.0)]
    baseline = [{
        "record_type": "static",
        "split": "static_50_50",
        "batch_fingerprint": "batch-0",
        "latency_ms": 5.0,
        "latency_semantics": "inference_only",
        "implementation_scope": "physical_static_shared_hbm_baseline",
        "state_budget_bytes": 101,
        "embedding_physical_hbm_bytes": 20,
        "kv_physical_hbm_bytes": 30,
        "managed_state_physical_hbm_bytes": 50,
    }]
    with pytest.raises(EvidenceValidationError, match="different state budgets"):
        compare_static_baseline(hot, _with_latency_fields(baseline))


def test_static_baseline_requires_complete_pairing():
    hot = [_hot_row(0, 4.0), _hot_row(1, 6.0)]
    baseline = [{
        "record_type": "static",
        "split": "static_50_50",
        "batch_fingerprint": "batch-0",
        "latency_ms": 5.0,
        "latency_semantics": "inference_only",
        "implementation_scope": "physical_static_shared_hbm_baseline",
        "state_budget_bytes": 100,
        "embedding_physical_hbm_bytes": 20,
        "kv_physical_hbm_bytes": 30,
        "managed_state_physical_hbm_bytes": 50,
    }]
    with pytest.raises(EvidenceValidationError, match="incomplete pairing"):
        compare_static_baseline(hot, _with_latency_fields(baseline))
