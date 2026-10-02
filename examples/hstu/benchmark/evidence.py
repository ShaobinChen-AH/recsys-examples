"""Evidence accounting for HotState benchmark traces.

The prototype has several benchmark modes with deliberately different
semantics.  This module keeps their evidence claims explicit and validates the
strongest claims (physical HBM accounting and paired latency comparison) from
JSONL data rather than from policy estimates.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class EvidenceValidationError(ValueError):
    """Raised when a trace cannot support the claim it is labeled with."""


CLAIM_CATALOG = {
    "fixed_shared_hbm_envelope": "measured",
    "physical_embedding_and_kv_accounting": "measured",
    "per_batch_inference_latency": "measured",
    "paired_static_baseline_comparison": "requires_static_baseline",
    "online_cross_type_reallocation": "not_demonstrated",
    "unified_training_inference_arbitration": "not_demonstrated",
    "paper_scale_generalization": "not_demonstrated",
}


def load_jsonl(path: str | Path) -> List[dict]:
    records: List[dict] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceValidationError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise EvidenceValidationError(
                    f"JSONL record at {path}:{line_number} is not an object"
                )
            record["_line_number"] = line_number
            records.append(record)
    return records


def _latency_records(records: Iterable[dict]) -> List[dict]:
    return [
        record
        for record in records
        if record.get("record_type") != "evidence_summary"
        and record.get("latency_ms") is not None
    ]


def _number(record: dict, name: str) -> int:
    try:
        raw = record[name]
        if isinstance(raw, bool):
            raise TypeError("boolean is not a byte count")
        value = int(raw)
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError("fractional value")
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceValidationError(
            f"record line {record.get('_line_number', '?')} has no integer {name}"
        ) from exc
    if value < 0:
        raise EvidenceValidationError(
            f"record line {record.get('_line_number', '?')} has negative {name}"
        )
    return value


def _latency_value(record: dict) -> float:
    """Return a finite, non-negative latency value from one trace row."""
    try:
        value = float(record["latency_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceValidationError(
            f"record line {record.get('_line_number', '?')} has no numeric latency_ms"
        ) from exc
    if not math.isfinite(value) or value < 0:
        raise EvidenceValidationError(
            f"record line {record.get('_line_number', '?')} has invalid latency_ms={value}"
        )
    return value


def _budget_from_record(record: dict) -> Optional[int]:
    """Read the common state-budget field used by either benchmark."""
    for name in (
        "configured_state_budget_bytes",
        "state_budget_bytes",
        "inference_state_budget_bytes",
    ):
        if name in record and record[name] is not None:
            return _number(record, name)
    return None


def _index_by_sample(records: Iterable[dict]) -> Dict[Tuple[Any, ...], dict]:
    indexed: Dict[Tuple[Any, ...], dict] = {}
    for record in records:
        key = _sample_key(record)
        if key in indexed:
            raise EvidenceValidationError(
                f"duplicate paired sample key {key!r} in trace"
            )
        indexed[key] = record
    return indexed


def _sample_key(record: dict) -> Tuple[Any, ...]:
    fingerprint = record.get("batch_fingerprint")
    if fingerprint:
        # Keep the ordinal when a deterministic workload legitimately repeats
        # an identical request payload.  The ordinal is shared by paired
        # traces, while the fingerprint still guards against misordered data.
        if "batch_idx" in record:
            try:
                batch_idx = int(record["batch_idx"])
            except (TypeError, ValueError) as exc:
                raise EvidenceValidationError(
                    "paired sample batch_idx must be an integer"
                ) from exc
            return ("fingerprint", str(fingerprint), batch_idx)
        return ("fingerprint", str(fingerprint))
    required = ("batch_idx", "user_id", "seq_history_len")
    if all(name in record for name in required):
        try:
            coordinates = (
                int(record["batch_idx"]),
                int(record["user_id"]),
                int(record["seq_history_len"]),
            )
        except (TypeError, ValueError) as exc:
            raise EvidenceValidationError(
                "paired sample coordinates must be integers"
            ) from exc
        return ("coordinates", *coordinates)
    raise EvidenceValidationError(
        "paired comparison requires batch_fingerprint or "
        "(batch_idx, user_id, seq_history_len)"
    )


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def latency_stats(records: Iterable[dict]) -> dict:
    values = [_latency_value(record) for record in records]
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    return {
        "n": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values),
    }


def validate_hotstate_trace(records: Iterable[dict], *, require_physical: bool = True) -> dict:
    """Validate a HotState trace and return an evidence report.

    ``require_physical=False`` is reserved for the admission smoke path.  That
    path intentionally exercises the admission callback, but it is not allowed
    to inherit the fixed-envelope claims merely because the runtime happened to
    emit a few allocation counters.
    """
    rows = _latency_records(records)
    if not rows:
        raise EvidenceValidationError("trace has no latency records")

    budgets = {
        budget
        for row in rows
        for budget in [_budget_from_record(row)]
        if budget is not None
    }
    if len(budgets) != 1:
        raise EvidenceValidationError(
            "HotState trace must contain one state budget value"
        )
    budget = budgets.pop()
    if budget <= 0:
        raise EvidenceValidationError("configured state budget must be positive")

    variants = {str(row.get("benchmark_variant", "physical_envelope")) for row in rows}
    if len(variants) != 1:
        raise EvidenceValidationError(
            f"HotState trace mixes benchmark variants: {sorted(variants)}"
        )
    smoke_variant = variants == {"admission_smoke"}
    if not require_physical and (
        not smoke_variant
        or not all(bool(row.get("admission_strategy_enabled")) for row in rows)
    ):
        raise EvidenceValidationError(
            "require_physical=False is reserved for admission_smoke rows with "
            "admission_strategy_enabled=true"
        )
    physical_rows = 0
    action_rows = 0
    online_reallocation = False
    physical_reallocation_events = 0
    rebuild_count = 0
    remaining_values = []
    for row in rows:
        _latency_value(row)
        if row.get("latency_semantics") != "inference_only":
            raise EvidenceValidationError(
                "HotState evidence requires latency_semantics='inference_only'"
            )
        if row.get("implementation_scope") not in (None, "fixed_shared_hbm_inference"):
            raise EvidenceValidationError(
                "trace is not a fixed_shared_hbm_inference trace"
            )
        if (smoke_variant or row.get("admission_strategy_enabled")) and require_physical:
            raise EvidenceValidationError(
                "admission smoke traces cannot support the physical HotState claim"
            )
        if require_physical:
            measured = _number(row, "managed_state_physical_hbm_bytes")
            embedding = _number(row, "embedding_physical_hbm_bytes")
            kv = _number(row, "kv_physical_hbm_bytes")
            if measured != embedding + kv:
                raise EvidenceValidationError(
                    f"line {row.get('_line_number', '?')} has inconsistent physical byte sum"
                )
            if measured > budget:
                raise EvidenceValidationError(
                    f"line {row.get('_line_number', '?')} exceeds physical state budget"
                )
            physical_rows += 1
            remaining_values.append(budget - measured)
            if row.get("physical_admitted_keys") or row.get("physical_evicted_keys"):
                action_rows += 1
        if row.get("online_reallocation_observed"):
            events = row.get("physical_reallocation_events", ())
            if not isinstance(events, (list, tuple)):
                raise EvidenceValidationError(
                    "physical_reallocation_events must be a list when present"
                )
            if events:
                online_reallocation = True
                physical_reallocation_events += len(events)
        rebuild_count = max(rebuild_count, int(row.get("model_rebuild_count", 0)))

    if require_physical and physical_rows != len(rows):
        raise EvidenceValidationError("every latency row must contain physical HBM measurements")

    claims = dict(CLAIM_CATALOG)
    if not require_physical:
        claims.update(
            {
                "fixed_shared_hbm_envelope": "not_measured",
                "physical_embedding_and_kv_accounting": "not_measured",
                "per_batch_inference_latency": "diagnostic_only",
                "paired_static_baseline_comparison": "not_applicable",
            }
        )
    else:
        claims["paired_static_baseline_comparison"] = "requires_static_baseline"
    claims["online_cross_type_reallocation"] = (
        "measured" if online_reallocation and rebuild_count == 0 else "not_demonstrated"
    )
    return {
        "evidence_schema_version": 1,
        "implementation_scope": (
            "admission_smoke_diagnostic" if not require_physical
            else "fixed_shared_hbm_inference"
        ),
        "record_count": len(rows),
        "configured_state_budget_bytes": budget,
        "physical_rows_validated": physical_rows,
        "min_remaining_state_budget_bytes": (
            min(remaining_values) if remaining_values else None
        ),
        "physical_action_rows": action_rows,
        "online_reallocation_observed": online_reallocation,
        "physical_reallocation_events": physical_reallocation_events,
        "model_rebuild_count": rebuild_count,
        "latency_stats": latency_stats(rows),
        "claims": claims,
    }


def validate_static_baseline_trace(records: Iterable[dict]) -> dict:
    """Validate a physically measured static-split baseline trace."""
    rows = _latency_records(records)
    if not rows:
        raise EvidenceValidationError("static trace has no latency records")
    budgets = set()
    splits = set()
    for row in rows:
        _latency_value(row)
        if row.get("latency_semantics") != "inference_only":
            raise EvidenceValidationError(
                "static baseline rows require latency_semantics='inference_only'"
            )
        if row.get("implementation_scope") != "physical_static_shared_hbm_baseline":
            raise EvidenceValidationError(
                "static baseline is missing physical_static_shared_hbm_baseline scope"
            )
        budget = _budget_from_record(row)
        if budget is None or budget <= 0:
            raise EvidenceValidationError(
                "static baseline rows require a positive state budget"
            )
        budgets.add(budget)
        measured = _number(row, "managed_state_physical_hbm_bytes")
        embedding = _number(row, "embedding_physical_hbm_bytes")
        kv = _number(row, "kv_physical_hbm_bytes")
        if measured != embedding + kv:
            raise EvidenceValidationError(
                f"line {row.get('_line_number', '?')} has inconsistent static physical byte sum"
            )
        if measured > budget:
            raise EvidenceValidationError(
                f"line {row.get('_line_number', '?')} exceeds static state budget"
            )
        split = str(row.get("split", ""))
        if not split.startswith("static_"):
            raise EvidenceValidationError(
                "static baseline rows require a static_* split label"
            )
        splits.add(split)
    if len(budgets) != 1:
        raise EvidenceValidationError(
            "static baseline must contain one state budget value"
        )
    return {
        "record_count": len(rows),
        "configured_state_budget_bytes": budgets.pop(),
        "static_splits": sorted(splits),
        "physical_rows_validated": len(rows),
        "latency_stats": latency_stats(rows),
    }


def compare_static_baseline(
    hotstate_records: Iterable[dict],
    baseline_records: Iterable[dict],
    *,
    require_complete_pairing: bool = True,
) -> dict:
    """Compare HotState against each static split on paired samples."""
    hot_rows = _latency_records(hotstate_records)
    static_rows = _latency_records(baseline_records)
    if not hot_rows or not static_rows:
        raise EvidenceValidationError("both HotState and static traces need latency records")

    # A baseline is useful evidence only when it is a physically measured
    # allocation under the same envelope.  Aggregate-only or policy-estimate
    # traces are deliberately rejected here.
    hot_report = validate_hotstate_trace(hot_rows, require_physical=True)
    hot_budget = hot_report["configured_state_budget_bytes"]
    baseline_report = validate_static_baseline_trace(static_rows)
    baseline_budgets = {baseline_report["configured_state_budget_bytes"]}
    if baseline_budgets != {hot_budget}:
        raise EvidenceValidationError(
            "HotState and static baseline use different state budgets: "
            f"hotstate={hot_budget}, baseline={sorted(baseline_budgets)}"
        )

    hot_by_key = _index_by_sample(hot_rows)
    by_split: Dict[str, Dict[Tuple[Any, ...], dict]] = defaultdict(dict)
    for row in static_rows:
        split = str(row.get("split", ""))
        if not split.startswith("static_"):
            continue
        key = _sample_key(row)
        if key in by_split[split]:
            raise EvidenceValidationError(
                f"duplicate paired sample key {key!r} in {split}"
            )
        by_split[split][key] = row
    if not by_split:
        raise EvidenceValidationError("baseline trace contains no static_* latency rows")

    split_reports = []
    for split, rows in sorted(by_split.items()):
        common = sorted(set(hot_by_key) & set(rows), key=repr)
        if not common:
            continue
        if require_complete_pairing and len(common) != len(hot_by_key):
            raise EvidenceValidationError(
                f"{split} has incomplete pairing: common_samples={len(common)} "
                f"hotstate_samples={len(hot_by_key)}"
            )
        hot_values = [float(hot_by_key[key]["latency_ms"]) for key in common]
        static_values = [float(rows[key]["latency_ms"]) for key in common]
        hot_mean = sum(hot_values) / len(hot_values)
        static_mean = sum(static_values) / len(static_values)
        split_reports.append({
            "split": split,
            "common_samples": len(common),
            "pairing_coverage": len(common) / len(hot_by_key),
            "hotstate": latency_stats([hot_by_key[key] for key in common]),
            "static": latency_stats([rows[key] for key in common]),
            "paired_mean_delta_ms": hot_mean - static_mean,
            "hotstate_gain_vs_split_pct": (
                100.0 * (static_mean - hot_mean) / static_mean if static_mean > 0 else None
            ),
        })
    if not split_reports:
        raise EvidenceValidationError("HotState and static traces have no paired samples")

    best_static = min(split_reports, key=lambda item: item["static"]["mean_ms"])
    hot_mean = best_static["hotstate"]["mean_ms"]
    best_mean = best_static["static"]["mean_ms"]
    return {
        "comparison_schema_version": 1,
        "paired": True,
        "static_splits": split_reports,
        "best_static_split": best_static["split"],
        "hotstate_stats_on_common_samples": best_static["hotstate"],
        "best_static_stats_on_common_samples": best_static["static"],
        "gain_vs_best_static_pct": (
            100.0 * (best_mean - hot_mean) / best_mean if best_mean > 0 else None
        ),
        "claim": "paired_static_baseline_comparison",
        "claim_status": "measured",
    }


def evidence_summary(trace_report: dict, comparison: Optional[dict] = None) -> dict:
    if trace_report["implementation_scope"] == "admission_smoke_diagnostic":
        claim_boundary = (
            "admission callback smoke diagnostic only; physical HBM envelope and "
            "online reallocation are not measured"
        )
    else:
        claim_boundary = (
            "fixed physical shared-HBM inference prototype; no online model resize"
        )
    summary = {
        "record_type": "evidence_summary",
        "evidence_schema_version": 1,
        "implementation_scope": trace_report["implementation_scope"],
        "claim_boundary": claim_boundary,
        "trace_report": trace_report,
        "comparison": comparison,
    }
    if comparison is not None:
        trace_report = dict(trace_report)
        trace_report["claims"] = dict(trace_report["claims"])
        trace_report["claims"]["paired_static_baseline_comparison"] = "measured"
        summary["trace_report"] = trace_report
    return summary
