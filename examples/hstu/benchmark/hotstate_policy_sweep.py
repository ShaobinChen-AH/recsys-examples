"""Run paired inference sweeps and select policy settings on held-out requests.

The runner starts a fresh inference process for every policy/repetition so
online model state cannot leak between variants. The analyzer validates the
physical trace and compares identical requests using end-to-end latency.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))

from benchmark.evidence import (  # noqa: E402
    validate_hotstate_trace,
    validate_static_baseline_trace,
)
from modules.hotstate.policy import PolicyConfig  # noqa: E402


def _read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resolve_input_path(path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and not candidate.exists():
        candidate = HSTU_ROOT / candidate
    return candidate.resolve()


def _load_variants(path):
    payload = _read_json(path)
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("sweep manifest must contain a non-empty variants list")

    names = set()
    validated = []
    for raw in variants:
        name = str(raw.get("name", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"invalid variant name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate variant name: {name}")
        names.add(name)
        hotstate_enabled = bool(raw.get("hotstate_enabled", True))
        config = PolicyConfig(**raw.get("policy", {}))
        ratio = config.construction_embedding_ratio(
            hotstate_enabled=hotstate_enabled
        )
        if not hotstate_enabled and not str(raw.get("static_split", "")).startswith(
            "static_"
        ):
            raise ValueError(f"static variant {name} requires a static_split label")
        validated.append(
            {
                "name": name,
                "hotstate_enabled": hotstate_enabled,
                "static_split": raw.get("static_split"),
                "stage": raw.get("stage", "unspecified"),
                "reference": bool(raw.get("reference", False)),
                "parameters_under_test": raw.get("parameters_under_test", {}),
                "policy_config": config.snapshot(),
                "construction_embedding_hbm_ratio": ratio,
            }
        )
    return validated


def run_sweep(args):
    all_variants = _load_variants(_resolve_input_path(args.variants_json))
    if args.stage and args.variant_filter:
        raise ValueError("use either --stage or --variant-filter, not both")
    variants = all_variants
    if args.stage:
        available_stages = {variant["stage"] for variant in all_variants}
        if args.stage not in available_stages:
            raise ValueError(
                f"unknown sweep stage {args.stage!r}; choose from "
                f"{', '.join(sorted(available_stages))}"
            )
        variants = [
            variant for variant in variants
            if not variant["hotstate_enabled"]
            or variant["stage"] == args.stage
            or variant["reference"]
        ]
    if args.variant_filter:
        requested = {
            name.strip()
            for name in args.variant_filter.split(",")
            if name.strip()
        }
        known_names = {variant["name"] for variant in all_variants}
        missing = requested - known_names
        if missing:
            raise ValueError(f"unknown variant(s): {', '.join(sorted(missing))}")
        variants = [variant for variant in all_variants if variant["name"] in requested]
        if any(variant["hotstate_enabled"] for variant in variants):
            variants.extend(
                variant for variant in all_variants
                if not variant["hotstate_enabled"]
                and variant["name"] not in requested
            )
    if not variants:
        raise ValueError("no variants selected")
    if (
        args.repeats <= 0
        or args.max_batches < 0
        or args.warmup_batches < 0
        or args.max_bs <= 0
        or args.timeout_seconds < 0
    ):
        raise ValueError("repeats/max_bs must be positive and max_batches non-negative")
    if args.budget_gib <= 0:
        raise ValueError("budget_gib must be positive")

    out_dir = Path(args.out_dir).expanduser().resolve()
    config_dir = out_dir / "configs"
    trace_dir = out_dir / "traces"
    log_dir = out_dir / "logs"
    for directory in (config_dir, trace_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    gin_path = _resolve_input_path(args.gin_config_file)
    checkpoint_path = _resolve_input_path(args.checkpoint_dir)
    if not gin_path.is_file():
        raise FileNotFoundError(f"gin config not found: {gin_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    manifest = {
        "sweep_schema_version": 1,
        "budget_gib": args.budget_gib,
        "max_bs": args.max_bs,
        "max_batches": args.max_batches,
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
        "stage": args.stage or "all",
        "gin_config_file": str(gin_path),
        "checkpoint_dir": str(checkpoint_path),
        "variants": [],
    }
    policy_by_name = {}
    for variant in variants:
        parameters = variant["policy_config"]["parameters"]
        config_path = config_dir / f"{variant['name']}.json"
        _write_json(config_path, parameters)
        policy_by_name[variant["name"]] = config_path
        manifest["variants"].append({**variant, "runs": []})
    _write_json(out_dir / "sweep_manifest.json", manifest)

    selected_names = [variant["name"] for variant in variants]
    inference_script = HSTU_ROOT / "inference" / "inference_gr_ranking.py"
    examples_root = HSTU_ROOT.parent
    failures = []

    for repeat in range(args.repeats):
        run_order = list(selected_names)
        random.Random(args.seed + repeat).shuffle(run_order)
        for name in run_order:
            variant = next(item for item in variants if item["name"] == name)
            trace_path = trace_dir / f"{name}.repeat_{repeat}.jsonl"
            log_path = log_dir / f"{name}.repeat_{repeat}.log"
            trace_path.unlink(missing_ok=True)
            env = os.environ.copy()
            python_path = [str(examples_root), str(HSTU_ROOT)]
            if env.get("PYTHONPATH"):
                python_path.append(env["PYTHONPATH"])
            env.update(
                {
                    "PYTHONPATH": os.pathsep.join(python_path),
                    "HBM_BUDGET_GIB": str(args.budget_gib),
                    "HOTSTATE_ENABLED": "1" if variant["hotstate_enabled"] else "0",
                    "HOTSTATE_POLICY_CONFIG": str(policy_by_name[name]),
                    "HOTSTATE_EMBEDDING_HBM_RATIO": str(
                        variant["construction_embedding_hbm_ratio"]
                    ),
                    "HOTSTATE_TRACE_PATH": str(trace_path),
                    "HOTSTATE_TRACE_DETAIL": "scalar",
                    "HOTSTATE_SWEEP_SEED": str(args.seed),
                    "HOTSTATE_SWEEP_REPEAT": str(repeat),
                    "HOTSTATE_STATIC_SPLIT": (
                        variant.get("static_split") or "static_disabled"
                    ),
                }
            )
            command = [
                args.python,
                str(inference_script),
                "--gin_config_file",
                str(gin_path),
                "--checkpoint_dir",
                str(checkpoint_path),
                "--mode",
                "simulate",
                "--disable_auc",
                "--max_bs",
                str(args.max_bs),
                "--warmup_batches",
                str(args.warmup_batches),
            ]
            if args.max_batches:
                command.extend(["--max_batches", str(args.max_batches)])
            if args.disable_context:
                command.append("--disable_context")

            print(f"[{repeat + 1}/{args.repeats}] {name}: starting")
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(HSTU_ROOT),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.timeout_seconds or None,
                    )
                    return_code = completed.returncode
                except subprocess.TimeoutExpired:
                    return_code = 124
            run_record = {
                "repeat": repeat,
                "return_code": return_code,
                "trace_path": str(trace_path.relative_to(out_dir)),
                "log_path": str(log_path.relative_to(out_dir)),
            }
            next(item for item in manifest["variants"] if item["name"] == name)[
                "runs"
            ].append(run_record)
            if return_code:
                failures.append(f"{name} repeat {repeat}: exit {return_code}")
                print(f"[{repeat + 1}/{args.repeats}] {name}: failed (see {log_path})")
            else:
                print(f"[{repeat + 1}/{args.repeats}] {name}: complete")
            _write_json(out_dir / "sweep_manifest.json", manifest)

    _write_json(out_dir / "sweep_manifest.json", manifest)
    print(f"manifest={out_dir / 'sweep_manifest.json'}")
    return 1 if failures else 0


def _read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if (
                isinstance(record, dict)
                and record.get("end_to_end_latency_ms") is not None
            ):
                records.append(record)
    if not records:
        raise ValueError(f"trace contains no measured requests: {path}")
    return records


def _sample_key(record):
    fingerprint = record.get("batch_fingerprint")
    batch_idx = record.get("batch_idx")
    if not fingerprint or batch_idx is None:
        raise ValueError("policy sweep rows require batch_fingerprint and batch_idx")
    return str(fingerprint), int(batch_idx)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _stats(values):
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    return {
        "n": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
    }


def _metrics(records):
    def values(field):
        return [float(row[field]) for row in records if row.get(field) is not None]

    physical_values = values("managed_state_physical_hbm_bytes")
    result = {
        "requests": len(records),
        "end_to_end_latency": _stats(values("end_to_end_latency_ms")),
        "forward_latency": _stats(values("forward_latency_ms")),
        "control_latency": _stats(values("hotstate_control_ms")),
        "post_control_latency": _stats(values("hotstate_post_control_ms")),
        "physical_state_bytes": {
            "mean": (
                sum(physical_values) / len(physical_values)
                if physical_values else None
            ),
            "max": max(physical_values, default=None),
        },
        "physical_admissions": sum(
            len(row.get("physical_admitted_keys", [])) for row in records
        ),
        "physical_evictions": sum(
            len(row.get("physical_evicted_keys", [])) for row in records
        ),
    }
    request_fields = (
        "history_tokens",
        "history_length",
        "num_candidates",
        "embedding_requests",
        "unique_embedding_requests",
        "hbm_pressure",
    )
    result["request_workload"] = {
        field: _stats(
            request[field]
            for row in records
            for request in [row.get("request_metrics", {})]
            if isinstance(request, dict) and request.get(field) is not None
        )
        for field in request_fields
    }
    feature_ranges = {}
    for row in records:
        for feature_name, observed in row.get(
            "dynamic_feature_id_ranges", {}
        ).items():
            summary = feature_ranges.setdefault(
                feature_name,
                {
                    "table_name": observed.get("table_name"),
                    "vocab_size": observed.get("vocab_size"),
                    "count": 0,
                    "min_id": None,
                    "max_id": None,
                    "negative_id_count": 0,
                },
            )
            summary["count"] += int(observed.get("count", 0))
            summary["negative_id_count"] += int(
                observed.get("negative_id_count", 0)
            )
            min_id = observed.get("min_id")
            max_id = observed.get("max_id")
            if min_id is not None:
                summary["min_id"] = (
                    int(min_id)
                    if summary["min_id"] is None
                    else min(summary["min_id"], int(min_id))
                )
            if max_id is not None:
                summary["max_id"] = (
                    int(max_id)
                    if summary["max_id"] is None
                    else max(summary["max_id"], int(max_id))
                )
    result["dynamic_feature_id_ranges"] = feature_ranges
    if records:
        result["dynamic_embedding_key_spaces"] = records[0].get(
            "dynamic_embedding_key_spaces", []
        )
    cost_snapshots = [
        row.get("online_cost_model_after", {}) for row in records
        if isinstance(row.get("online_cost_model_after"), dict)
    ]
    if cost_snapshots:
        final = cost_snapshots[-1]
        prediction_errors = [
            float(snapshot["last_prediction_ms"])
            - float(snapshot["last_observation_ms"])
            for snapshot in cost_snapshots
            if snapshot.get("last_prediction_ms") is not None
            and snapshot.get("last_observation_ms") is not None
        ]
        result["online_cost_model"] = {
            "calibration_state": final.get("calibration_state"),
            "confidence": final.get("confidence"),
            "batch_samples": final.get("batch_samples"),
            "batch_error_ewma_ms": final.get("batch_error_ewma_ms"),
            "batch_coefficients": final.get("batch_coefficients", {}),
            "transfer_models": final.get("transfer_models", {}),
            "forward_latency_prediction_error": {
                "n": len(prediction_errors),
                "mae_ms": (
                    sum(abs(error) for error in prediction_errors)
                    / len(prediction_errors)
                    if prediction_errors else None
                ),
                "bias_ms": (
                    sum(prediction_errors) / len(prediction_errors)
                    if prediction_errors else None
                ),
                "rmse_ms": (
                    math.sqrt(
                        sum(error * error for error in prediction_errors)
                        / len(prediction_errors)
                    )
                    if prediction_errors else None
                ),
            },
        }
    profile_fields = ("extract", "record_keys", "run_epoch", "scheduler", "total")
    result["control_profile_ms"] = {
        field: _stats(
            row["control_profile_ms"][field]
            for row in records
            if field in row.get("control_profile_ms", {})
        )
        for field in profile_fields
    }
    observed = [
        row.get("observed_cost_inputs") for row in records
        if isinstance(row.get("observed_cost_inputs"), dict)
    ]
    if observed:
        result["observed_cost_totals"] = {
            field: sum(float(item.get(field, 0) or 0) for item in observed)
            for field in (
                "embedding_requests",
                "embedding_misses",
                "kv_miss_tokens",
                "concurrent_transfer_bytes",
                "synchronous_control_transfer_bytes",
                "synchronous_kv_released_bytes",
                "offload_pages",
            )
        }
    state_components = {}
    for record in records:
        for state in record.get("policy_inputs", []):
            if not isinstance(state, dict):
                continue
            state_type = str(state.get("state_type", "unknown"))
            summary = state_components.setdefault(
                state_type,
                {"records": 0, "decision_counts": {}, "components": {}},
            )
            summary["records"] += 1
            decision = str(state.get("decision", "unknown"))
            summary["decision_counts"][decision] = (
                summary["decision_counts"].get(decision, 0) + 1
            )
            for field in (
                "gross_benefit_ms",
                "movement_penalty_ms",
                "semantic_risk_ms",
                "opportunity_cost_ms",
                "net_benefit_ms",
                "value_density_ms_per_byte",
            ):
                value = state.get(field)
                if isinstance(value, (int, float)):
                    summary["components"].setdefault(field, []).append(float(value))
    for summary in state_components.values():
        summary["components"] = {
            field: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for field, values in summary["components"].items()
        }
    result["scored_state_summary"] = state_components
    return result


def analyze_sweep(args):
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _read_json(manifest_path)
    variants = manifest.get("variants", [])
    if not variants:
        raise ValueError("sweep manifest contains no variants")

    runs_by_variant = {}
    budget_values = set()
    expected_repeats = int(manifest.get("repeats", 0))
    for variant in variants:
        name = variant["name"]
        runs = {}
        for run in variant.get("runs", []):
            if run.get("return_code") != 0:
                raise ValueError(f"{name} repeat {run.get('repeat')} did not complete")
            repeat = int(run["repeat"])
            trace_path = manifest_path.parent / run["trace_path"]
            records = _read_jsonl(trace_path)
            if variant["hotstate_enabled"]:
                report = validate_hotstate_trace(records, require_physical=True)
                config_ids = {
                    row.get("policy", {}).get("config_id") for row in records
                    if isinstance(row.get("policy"), dict)
                }
                expected_id = variant["policy_config"]["config_id"]
                if config_ids != {expected_id}:
                    raise ValueError(
                        f"{name} trace policy identity does not match its manifest"
                    )
            else:
                report = validate_static_baseline_trace(records)
            budget_values.add(report["configured_state_budget_bytes"])
            indexed = {}
            for record in records:
                key = _sample_key(record)
                if key in indexed:
                    raise ValueError(
                        f"duplicate request {key!r} in {name} repeat {repeat}"
                    )
                indexed[key] = record
            runs[repeat] = {
                "records": records,
                "indexed": indexed,
                "validation": report,
            }
        if not runs:
            raise ValueError(f"variant {name} has no completed runs")
        if expected_repeats and set(runs) != set(range(expected_repeats)):
            raise ValueError(
                f"variant {name} has incomplete repeats: "
                f"expected={expected_repeats}, actual={sorted(runs)}"
            )
        runs_by_variant[name] = {"variant": variant, "runs": runs}

    if len(budget_values) != 1:
        raise ValueError(
            f"variants do not share one state budget: {sorted(budget_values)}"
        )

    repeat_sets = [set(entry["runs"]) for entry in runs_by_variant.values()]
    if any(repeats != repeat_sets[0] for repeats in repeat_sets[1:]):
        raise ValueError("variants have different completed repeat sets")
    repeats = sorted(repeat_sets[0])
    key_sets_by_repeat = {}
    for repeat in repeats:
        key_sets = [
            set(entry["runs"][repeat]["indexed"])
            for entry in runs_by_variant.values()
        ]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError(
                f"request pairing differs between variants in repeat {repeat}"
            )
        key_sets_by_repeat[repeat] = key_sets[0]

    static_names = [
        name for name, entry in runs_by_variant.items()
        if not entry["variant"]["hotstate_enabled"]
    ]
    hot_names = [
        name for name, entry in runs_by_variant.items()
        if entry["variant"]["hotstate_enabled"]
    ]
    if not static_names or not hot_names:
        raise ValueError(
            "sweep must include at least one static baseline and one HotState variant"
        )

    discovery_keys = set()
    holdout_keys = set()
    for repeat in repeats:
        rows = next(iter(runs_by_variant.values()))["runs"][repeat]["records"]
        batch_indices = sorted({int(row["batch_idx"]) for row in rows})
        split_at = max(
            1,
            min(
                len(batch_indices) - 1,
                int(len(batch_indices) * args.discovery_fraction),
            ),
        )
        discovery_indices = set(batch_indices[:split_at])
        holdout_indices = set(batch_indices[split_at:])
        if not holdout_indices:
            raise ValueError(
                "each repeat needs at least two requests for discovery and holdout"
            )
        discovery_keys.update(
            (repeat, key)
            for key in key_sets_by_repeat[repeat]
            if key[1] in discovery_indices
        )
        holdout_keys.update(
            (repeat, key)
            for key in key_sets_by_repeat[repeat]
            if key[1] in holdout_indices
        )

    def rows_for(name, selected):
        rows = []
        for repeat, key in selected:
            rows.append(runs_by_variant[name]["runs"][repeat]["indexed"][key])
        return rows

    discovery_mean = {
        name: _stats(
            [row["end_to_end_latency_ms"] for row in rows_for(name, discovery_keys)]
        )["mean_ms"]
        for name in runs_by_variant
    }
    best_static = min(static_names, key=lambda name: discovery_mean[name])
    best_hotstate = min(hot_names, key=lambda name: discovery_mean[name])
    static_holdout = rows_for(best_static, holdout_keys)
    hotstate_holdout = rows_for(best_hotstate, holdout_keys)
    paired_deltas = [
        float(
            runs_by_variant[best_hotstate]["runs"][repeat]["indexed"][key][
                "end_to_end_latency_ms"
            ]
        )
        - float(
            runs_by_variant[best_static]["runs"][repeat]["indexed"][key][
                "end_to_end_latency_ms"
            ]
        )
        for repeat, key in sorted(holdout_keys)
    ]
    static_holdout_mean = _stats(
        [row["end_to_end_latency_ms"] for row in static_holdout]
    )["mean_ms"]
    hotstate_holdout_mean = _stats(
        [row["end_to_end_latency_ms"] for row in hotstate_holdout]
    )["mean_ms"]

    variant_reports = {}
    for name, entry in runs_by_variant.items():
        all_rows = [
            row
            for repeat in repeats
            for row in entry["runs"][repeat]["records"]
        ]
        variant_reports[name] = {
            "hotstate_enabled": entry["variant"]["hotstate_enabled"],
            "stage": entry["variant"].get("stage"),
            "parameters_under_test": entry["variant"].get("parameters_under_test", {}),
            "policy_config_id": entry["variant"]["policy_config"]["config_id"],
            "construction_embedding_hbm_ratio": entry["variant"][
                "construction_embedding_hbm_ratio"
            ],
            "discovery_end_to_end_latency": _stats(
                [row["end_to_end_latency_ms"] for row in rows_for(name, discovery_keys)]
            ),
            "holdout_end_to_end_latency": _stats(
                [row["end_to_end_latency_ms"] for row in rows_for(name, holdout_keys)]
            ),
            "all_request_metrics": _metrics(all_rows),
            "physical_validation": [
                entry["runs"][repeat]["validation"] for repeat in repeats
            ],
        }

    report = {
        "analysis_schema_version": 1,
        "sweep_stage": manifest.get("stage", "unknown"),
        "selection_metric": "end_to_end_latency_ms",
        "selection_protocol": {
            "discovery_fraction": args.discovery_fraction,
            "ranking": "mean end-to-end latency on discovery requests",
            "evaluation": "paired end-to-end latency on held-out requests",
            "request_pairing": "batch_fingerprint plus batch_idx within repeat",
        },
        "configured_state_budget_bytes": next(iter(budget_values)),
        "best_static_on_discovery": best_static,
        "best_hotstate_on_discovery": best_hotstate,
        "heldout_comparison": {
            "hotstate_variant": best_hotstate,
            "static_variant": best_static,
            "paired_requests": len(paired_deltas),
            "hotstate_mean_ms": hotstate_holdout_mean,
            "static_mean_ms": static_holdout_mean,
            "paired_mean_delta_ms": sum(paired_deltas) / len(paired_deltas),
            "gain_vs_best_static_pct": (
                100.0
                * (static_holdout_mean - hotstate_holdout_mean)
                / static_holdout_mean
                if static_holdout_mean > 0 else None
            ),
        },
        "variants": variant_reports,
    }
    output_path = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else manifest_path.parent / "policy_sweep_analysis.json"
    )
    _write_json(output_path, report)
    print(f"best_static_on_discovery={best_static}")
    print(f"best_hotstate_on_discovery={best_hotstate}")
    print(
        "heldout_gain_vs_best_static_pct="
        f"{report['heldout_comparison']['gain_vs_best_static_pct']}"
    )
    print(f"analysis={output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run controlled policy configurations")
    run_parser.add_argument("--gin-config-file", required=True)
    run_parser.add_argument("--checkpoint-dir", required=True)
    run_parser.add_argument("--variants-json", required=True)
    run_parser.add_argument("--out-dir", default="./logs/hotstate_policy_sweep")
    run_parser.add_argument("--budget-gib", type=float, required=True)
    run_parser.add_argument("--max-bs", type=int, default=1)
    run_parser.add_argument("--max-batches", type=int, default=0)
    run_parser.add_argument("--warmup-batches", type=int, default=64)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=2026)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--timeout-seconds", type=int, default=0)
    run_parser.add_argument("--disable-context", action="store_true")
    run_parser.add_argument("--variant-filter", default="")
    run_parser.add_argument("--stage", default="")
    run_parser.set_defaults(function=run_sweep)

    analyze_parser = commands.add_parser(
        "analyze", help="validate and compare completed sweep traces"
    )
    analyze_parser.add_argument("--manifest", required=True)
    analyze_parser.add_argument("--out-json", default="")
    analyze_parser.add_argument("--discovery-fraction", type=float, default=0.5)
    analyze_parser.set_defaults(function=analyze_sweep)

    args = parser.parse_args()
    if args.command == "analyze" and not 0.0 < args.discovery_fraction < 1.0:
        parser.error("--discovery-fraction must be in (0, 1)")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
