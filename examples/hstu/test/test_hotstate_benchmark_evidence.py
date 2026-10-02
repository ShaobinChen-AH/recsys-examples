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


def _hot_row(index, latency, embedding=20, kv=30):
    return {
        "record_type": "hotstate",
        "implementation_scope": "fixed_shared_hbm_inference",
        "latency_semantics": "inference_only",
        "batch_fingerprint": f"batch-{index}",
        "latency_ms": latency,
        "configured_state_budget_bytes": 100,
        "embedding_physical_hbm_bytes": embedding,
        "kv_physical_hbm_bytes": kv,
        "managed_state_physical_hbm_bytes": embedding + kv,
        "physical_admitted_keys": [],
        "physical_evicted_keys": [],
        "admission_strategy_enabled": False,
        "model_rebuild_count": 0,
        "online_reallocation_observed": False,
        "benchmark_variant": "physical_envelope",
    }


def test_fixed_envelope_trace_is_measured_but_not_online_reallocation():
    report = validate_hotstate_trace([_hot_row(0, 4.0), _hot_row(1, 5.0)])

    assert report["physical_rows_validated"] == 2
    assert report["claims"]["fixed_shared_hbm_envelope"] == "measured"
    assert report["claims"]["online_cross_type_reallocation"] == "not_demonstrated"


def test_self_reported_reallocation_flag_is_not_evidence():
    row = _hot_row(0, 4.0)
    row["online_reallocation_observed"] = True
    report = validate_hotstate_trace([row])

    assert report["physical_reallocation_events"] == 0
    assert report["claims"]["online_cross_type_reallocation"] == "not_demonstrated"


def test_trace_rejects_budget_or_accounting_violation():
    row = _hot_row(0, 4.0, embedding=80, kv=30)
    with pytest.raises(EvidenceValidationError, match="physical state budget"):
        validate_hotstate_trace([row])

    row = _hot_row(0, 4.0)
    row["managed_state_physical_hbm_bytes"] = 49
    with pytest.raises(EvidenceValidationError, match="inconsistent physical byte sum"):
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
    comparison = compare_static_baseline(hot, static)

    assert comparison["paired"] is True
    assert comparison["best_static_split"] == "static_30_70"
    assert comparison["gain_vs_best_static_pct"] == pytest.approx(16.6666667)


def test_evidence_summary_is_json_serializable():
    report = validate_hotstate_trace([_hot_row(0, 4.0)])
    payload = evidence_summary(report)
    json.dumps(payload)
    assert payload["record_type"] == "evidence_summary"
    assert "no online model resize" in payload["claim_boundary"]


def test_static_baseline_report_requires_physical_scope():
    row = {
        "record_type": "static",
        "split": "static_50_50",
        "batch_fingerprint": "batch-0",
        "latency_ms": 5.0,
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
        compare_static_baseline(hot, baseline)


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
        compare_static_baseline(hot, baseline)
