"""CPU checks for paired HotState calibration analysis."""

import json
import random
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import numpy as np
import torch

HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))

from benchmark.evidence import validate_hotstate_trace  # noqa: E402
from benchmark.hotstate_policy_sweep import (  # noqa: E402
    _load_variants,
    _metrics,
    analyze_sweep,
)
from modules.hotstate.inference_runtime import run_inference_request  # noqa: E402
from modules.hotstate.policy import PolicyConfig  # noqa: E402
from modules.hotstate.sweep_rng import (  # noqa: E402
    derive_sweep_batch_seed,
    seed_sweep_rngs,
)
from modules.hotstate.trace_writer import (  # noqa: E402
    close_jsonl_writers,
    write_jsonl_record,
)


def _row(index, latency, *, hotstate, config_id=None):
    record = {
        "record_type": "hotstate" if hotstate else "static",
        "implementation_scope": (
            "fixed_shared_hbm_inference"
            if hotstate
            else "physical_static_shared_hbm_baseline"
        ),
        "latency_semantics": "inference_only",
        "benchmark_variant": "physical_envelope",
        "batch_fingerprint": f"request-{index}",
        "batch_idx": index,
        "latency_ms": latency,
        "forward_latency_ms": latency - 1.0 if hotstate else latency,
        "end_to_end_latency_ms": latency,
        "configured_state_budget_bytes": 100,
        "embedding_physical_hbm_bytes": 20,
        "kv_physical_hbm_bytes": 30,
        "managed_state_physical_hbm_bytes": 50,
        "split": "hotstate" if hotstate else "static_50_50",
    }
    if hotstate:
        record.update(
            {
                "hotstate_control_ms": 1.0,
                "hotstate_post_control_ms": 0.1,
                "policy": {"config_id": config_id},
                "online_cost_model_after": {
                    "calibration_state": "online",
                    "confidence": 1.0,
                    "batch_samples": 32,
                    "batch_error_ewma_ms": 0.05,
                    "batch_coefficients": {"embedding_miss_hundreds": 1.0},
                    "transfer_models": {},
                },
                "physical_admitted_keys": [],
                "physical_evicted_keys": [],
            }
        )
    return record


def _write_trace(path, records):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )


def test_trace_writer_appends_buffered_compact_jsonl(tmp_path):
    trace_path = tmp_path / "buffered.jsonl"
    write_jsonl_record(trace_path, {"batch_idx": 0, "latency_ms": 1.25})
    write_jsonl_record(trace_path, {"batch_idx": 1, "latency_ms": 1.5})
    close_jsonl_writers()

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows == [
        {"batch_idx": 0, "latency_ms": 1.25},
        {"batch_idx": 1, "latency_ms": 1.5},
    ]


def test_default_policy_sweep_manifest_has_valid_staged_variants():
    variants = _load_variants(HSTU_ROOT / "benchmark" / "hotstate_policy_sweep.json")
    stages = {variant["stage"] for variant in variants}

    assert len(variants) == 49
    assert {"pool_split", "value_model", "weights", "reuse", "scheduler",
            "admission", "online_model"}.issubset(stages)
    assert sum(not variant["hotstate_enabled"] for variant in variants) == 3
    assert sum(variant["reference"] for variant in variants) == 1


def test_sweep_rngs_repeat_for_identical_seed_repeat_and_batch():
    first_seed = seed_sweep_rngs(2026, 2, 7)
    first_draws = (
        first_seed,
        random.randint(0, 10_000_000),
        int(np.random.randint(0, 10_000_000)),
        torch.randint(10_000_000, (4,)).tolist(),
    )
    second_seed = seed_sweep_rngs(2026, 2, 7)
    second_draws = (
        second_seed,
        random.randint(0, 10_000_000),
        int(np.random.randint(0, 10_000_000)),
        torch.randint(10_000_000, (4,)).tolist(),
    )

    assert first_draws == second_draws
    assert derive_sweep_batch_seed(2026, 2, 7) != derive_sweep_batch_seed(
        2026, 2, 8
    )
    assert derive_sweep_batch_seed(2026, 2, 7) != derive_sweep_batch_seed(
        2026, 3, 7
    )


def test_analysis_ranks_on_discovery_and_reports_paired_holdout(tmp_path):
    config = PolicyConfig().snapshot()
    _write_trace(
        tmp_path / "hot.jsonl",
        [_row(0, 4.0, hotstate=True, config_id=config["config_id"]),
         _row(1, 7.0, hotstate=True, config_id=config["config_id"])],
    )
    _write_trace(
        tmp_path / "static.jsonl",
        [_row(0, 5.0, hotstate=False), _row(1, 8.0, hotstate=False)],
    )
    manifest = {
        "variants": [
            {
                "name": "hot",
                "hotstate_enabled": True,
                "policy_config": config,
                "construction_embedding_hbm_ratio": 0.5,
                "runs": [{"repeat": 0, "return_code": 0, "trace_path": "hot.jsonl"}],
            },
            {
                "name": "static",
                "hotstate_enabled": False,
                "policy_config": config,
                "construction_embedding_hbm_ratio": 0.5,
                "runs": [{"repeat": 0, "return_code": 0, "trace_path": "static.jsonl"}],
            },
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    analyze_sweep(
        Namespace(
            manifest=str(manifest_path),
            out_json="",
            discovery_fraction=0.5,
        )
    )

    report = json.loads(
        (tmp_path / "policy_sweep_analysis.json").read_text(encoding="utf-8")
    )
    assert report["best_hotstate_on_discovery"] == "hot"
    assert report["best_static_on_discovery"] == "static"
    assert report["heldout_comparison"]["paired_requests"] == 1
    assert report["heldout_comparison"]["gain_vs_best_static_pct"] == pytest.approx(
        12.5
    )


def test_analysis_rejects_unpaired_request_traces(tmp_path):
    config = PolicyConfig().snapshot()
    _write_trace(
        tmp_path / "hot.jsonl",
        [_row(0, 4.0, hotstate=True, config_id=config["config_id"]),
         _row(1, 7.0, hotstate=True, config_id=config["config_id"])],
    )
    _write_trace(
        tmp_path / "static.jsonl",
        [_row(0, 5.0, hotstate=False), _row(2, 8.0, hotstate=False)],
    )
    manifest = {
        "variants": [
            {"name": "hot", "hotstate_enabled": True, "policy_config": config,
             "runs": [{"repeat": 0, "return_code": 0, "trace_path": "hot.jsonl"}]},
            {"name": "static", "hotstate_enabled": False, "policy_config": config,
             "runs": [{"repeat": 0, "return_code": 0, "trace_path": "static.jsonl"}]},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="request pairing differs"):
        analyze_sweep(
            Namespace(manifest=str(manifest_path), out_json="", discovery_fraction=0.5)
        )


def test_inference_trace_keeps_scalar_measurements_without_per_key_dumps(
    monkeypatch,
):
    monkeypatch.setenv("HOTSTATE_TRACE_DETAIL", "scalar")

    class Features:
        @staticmethod
        def keys():
            return ["item_feat"]

        @staticmethod
        def values():
            return torch.tensor([1, 2])

        @staticmethod
        def lengths():
            return torch.tensor([2])

    class Batch:
        features = Features()
        max_num_candidates = 10

    class Cache:
        last_wait_timings = {"prepare_wait_ms": 0.1}

    class Dense:
        async_kvcache = Cache()

    class Model:
        dense_module = Dense()

        @staticmethod
        def forward_with_kvcache(_batch, _uids, _history_lengths):
            return torch.tensor([1.0])

    class Controller:
        policy_config = PolicyConfig()
        configured_state_budget_bytes = 100

        def __init__(self):
            self.record = None

        @staticmethod
        def before_batch(_batch, _uids, _history_lengths, _batch_idx):
            return {
                "policy_budget": {"residency_budget_bytes": 50},
                "request_metrics": {"embedding_requests": 2},
                "profile_ms": {"run_epoch": 0.2},
                "online_cost_model": {"confidence": 0.25},
                "physical_admitted_keys": ["emb:item:11"],
                "physical_evicted_keys": [],
                "state_trace": [{"logical_key": "per-key-state"}],
                "policy_inputs": [{"logical_key": "per-key-policy-input"}],
                "transfer_execution": {
                    "submitted_ids": ["transfer-1"],
                    "submitted_bytes": 256,
                },
                "transfer_scheduler": {
                    "queued": [{"logical_key": "queued-key"}],
                },
                "consistency": {"entries": [{"logical_key": "directory-key"}]},
            }

        @staticmethod
        def after_batch(_batch, _latency_ms):
            return {
                "observed_cost_inputs": {"embedding_misses": 1},
                "online_cost_model": {"confidence": 0.5},
                "consistency": {"entries": [{"logical_key": "post-key"}]},
                "transfer_scheduler": {
                    "queued": [{"logical_key": "post-queued-key"}],
                },
            }

        @staticmethod
        def physical_hbm_snapshot():
            return {
                "embedding_physical_hbm_bytes": 20,
                "kv_physical_hbm_bytes": 30,
                "managed_state_physical_hbm_bytes": 50,
            }

        def record_inference_trace(self, record):
            self.record = record

        @staticmethod
        def abort_batch(_error):
            raise AssertionError("inference request should not fail")

    controller = Controller()
    run_inference_request(
        Model(),
        Batch(),
        torch.tensor([7]),
        torch.tensor([128]),
        controller=controller,
        batch_idx=3,
    )

    assert controller.record["request_metrics"]["embedding_requests"] == 2
    assert controller.record["control_profile_ms"]["run_epoch"] == 0.2
    assert controller.record["online_cost_model_before"]["confidence"] == 0.25
    assert controller.record["observed_cost_inputs"]["embedding_misses"] == 1
    assert controller.record["online_cost_model_after"]["confidence"] == 0.5
    assert controller.record["physical_admission_count"] == 1
    assert controller.record["transfer_execution"]["submitted_count"] == 1
    assert controller.record["metadata_future_wait_ms"] == pytest.approx(0.1)
    for field in (
        "physical_admitted_keys",
        "physical_evicted_keys",
        "state_trace",
        "policy_inputs",
        "consistency",
    ):
        assert field not in controller.record
    assert "submitted_ids" not in controller.record["transfer_execution"]
    assert "queued" not in controller.record["transfer_scheduler_after"]
    assert "post_control" not in controller.record
    assert "scored_state_summary" not in controller.record
    assert "dynamic_feature_id_ranges" not in controller.record


def test_full_inference_trace_is_explicitly_opt_in(monkeypatch):
    monkeypatch.setenv("HOTSTATE_TRACE_DETAIL", "full")

    class Features:
        @staticmethod
        def keys():
            return ["item_feat"]

        @staticmethod
        def values():
            return torch.tensor([1, 2])

        @staticmethod
        def lengths():
            return torch.tensor([2])

    class Batch:
        features = Features()
        max_num_candidates = 10

    class Cache:
        last_wait_timings = {}

    class Dense:
        async_kvcache = Cache()

    class Model:
        dense_module = Dense()

        @staticmethod
        def forward_with_kvcache(_batch, _uids, _history_lengths):
            return torch.tensor([1.0])

    class Controller:
        policy_config = PolicyConfig()
        configured_state_budget_bytes = 100
        trace_detail = "full"

        def __init__(self):
            self.record = None

        @staticmethod
        def before_batch(_batch, _uids, _history_lengths, _batch_idx):
            return {
                "physical_admitted_keys": ["emb:item:11"],
                "physical_evicted_keys": [],
                "state_trace": [{"logical_key": "per-key-state"}],
                "policy_inputs": [{"logical_key": "per-key-policy-input"}],
                "transfer_execution": {"submitted_ids": ["transfer-1"]},
            }

        @staticmethod
        def after_batch(_batch, _latency_ms):
            return {"post_directory_state": [{"logical_key": "post-key"}]}

        @staticmethod
        def physical_hbm_snapshot():
            return {
                "embedding_physical_hbm_bytes": 20,
                "kv_physical_hbm_bytes": 30,
                "managed_state_physical_hbm_bytes": 50,
            }

        def record_inference_trace(self, record):
            self.record = record

        @staticmethod
        def abort_batch(_error):
            raise AssertionError("inference request should not fail")

    controller = Controller()
    run_inference_request(
        Model(),
        Batch(),
        torch.tensor([7]),
        torch.tensor([128]),
        controller=controller,
    )

    assert controller.record["trace_detail"] == "full"
    assert controller.record["physical_admitted_keys"] == ["emb:item:11"]
    assert controller.record["policy_inputs"][0]["logical_key"] == "per-key-policy-input"
    assert controller.record["post_control"]["post_directory_state"][0]["logical_key"] == "post-key"


def test_untraced_warmup_does_not_create_or_append_trace(tmp_path, monkeypatch):
    trace_path = tmp_path / "warmup.jsonl"
    monkeypatch.setenv("HOTSTATE_TRACE_PATH", str(trace_path))

    class Features:
        @staticmethod
        def keys():
            return ["item_feat"]

    class Batch:
        features = Features()

    class Model:
        def __init__(self):
            self.forward_calls = 0

        def forward_with_kvcache(self, _batch, _uids, _history_lengths):
            self.forward_calls += 1
            return torch.tensor([1.0])

    class Controller:
        _trace_enabled = True

        def __init__(self):
            self.recorded = False
            self.trace_path = trace_path

        def before_batch(self, *_args):
            assert not self._trace_enabled
            return {}

        def after_batch(self, *_args):
            assert not self._trace_enabled
            return {}

        def record_inference_trace(self, _record):
            self.recorded = True

        @staticmethod
        def abort_batch(_error):
            raise AssertionError("warmup inference request should not fail")

    controller = Controller()
    model = Model()
    logits, record = run_inference_request(
        model,
        Batch(),
        torch.tensor([7]),
        torch.tensor([128]),
        controller=controller,
        trace_enabled=False,
    )

    assert logits.tolist() == [1.0]
    assert record is None
    assert model.forward_calls == 1
    assert not controller.recorded
    assert not trace_path.exists()


def test_scalar_sweep_metrics_use_action_counts_without_key_lists():
    metrics = _metrics(
        [
            {
                "end_to_end_latency_ms": 3.0,
                "forward_latency_ms": 2.0,
                "hotstate_control_ms": 0.8,
                "hotstate_post_control_ms": 0.2,
                "managed_state_physical_hbm_bytes": 50,
                "physical_admission_count": 4,
                "physical_eviction_count": 2,
                "request_metrics": {"embedding_requests": 10},
                "policy_inputs": [],
            }
        ]
    )

    assert metrics["physical_admissions"] == 4
    assert metrics["physical_evictions"] == 2
    assert metrics["requests"] == 1


def test_physical_evidence_accepts_scalar_action_counts():
    row = _row(0, 4.0, hotstate=True)
    row.pop("physical_admitted_keys")
    row.pop("physical_evicted_keys")
    row["physical_admission_count"] = 3
    row["physical_eviction_count"] = 2

    report = validate_hotstate_trace([row])

    assert report["physical_action_rows"] == 1
