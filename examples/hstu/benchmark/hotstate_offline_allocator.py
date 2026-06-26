#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass
from collections import defaultdict


BYTES_PER_GIB = 1024 ** 3
BYTES_PER_MIB = 1024 ** 2


@dataclass(frozen=True)
class TraceObject:
    key: str
    state_type: str
    footprint_bytes: int
    placement: tuple
    reuse_imminence: float
    stall_sensitivity_ms: float
    movement_cost_ms: float
    score: float
    benefit_density: float
    semantic_risk: float

    @property
    def is_kv(self):
        return self.key.startswith("kv:uid:")

    @property
    def is_embedding(self):
        return self.state_type.startswith("EMBEDDING") or self.key.startswith("emb:")

    @property
    def is_hbm_resident(self):
        return "HBM" in self.placement


def as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def as_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def load_records(path):
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_no"] = line_no
            records.append(record)
    if not records:
        raise SystemExit(f"empty trace: {path}")
    return records


def snapshot_field(name):
    if name in ("pre", "state_trace"):
        return "state_trace"
    if name in ("post", "post_state_trace"):
        return "post_state_trace"
    raise SystemExit("--snapshot must be pre/state_trace or post/post_state_trace")


def normalize_snapshot(record, field):
    if field not in record:
        raise SystemExit(
            f"record line {record.get('_line_no')} missing required field {field}"
        )
    raw = record[field]
    if not isinstance(raw, list):
        raise SystemExit(
            f"record line {record.get('_line_no')} field {field} is not a list"
        )

    objects = []
    for h in raw:
        key = h.get("logical_key")
        state_type = h.get("state_type")
        footprint = as_int(h.get("footprint_bytes"), -1)
        if not key or not state_type or footprint < 0:
            raise SystemExit(
                f"bad state handle at line {record.get('_line_no')}: {h}"
            )

        placement = h.get("placement", [])
        if isinstance(placement, str):
            placement = (placement,)
        else:
            placement = tuple(str(x) for x in placement)

        objects.append(
            TraceObject(
                key=str(key),
                state_type=str(state_type),
                footprint_bytes=footprint,
                placement=placement,
                reuse_imminence=as_float(h.get("reuse_imminence")),
                stall_sensitivity_ms=as_float(h.get("stall_sensitivity_ms")),
                movement_cost_ms=as_float(h.get("movement_cost_ms")),
                score=as_float(h.get("score")),
                benefit_density=as_float(h.get("benefit_density")),
                semantic_risk=as_float(h.get("semantic_risk")),
            )
        )
    return objects


def parse_kv_uid(key):
    try:
        return int(key.split(":")[-1])
    except Exception:
        return None


def object_accessed(obj, record):
    if obj.is_kv:
        return parse_kv_uid(obj.key) == as_int(record.get("user_id"), -999999)

    # Current trace has coarse embedding state, not row-level identities.
    # Do not invent fake rows: treat hot-row aggregate as accessed by every batch.
    if obj.state_type == "EMBEDDING_HOT_ROWS":
        return True

    return False


def future_access_count(obj, start_idx, records, horizon):
    end = min(len(records), start_idx + horizon)
    return sum(1 for j in range(start_idx, end) if object_accessed(obj, records[j]))


def future_stall_benefit(obj, start_idx, records, snapshots_by_key, horizon):
    end = min(len(records), start_idx + horizon)
    total = 0.0
    for j in range(start_idx, end):
        if not object_accessed(obj, records[j]):
            continue
        future_obj = snapshots_by_key[j].get(obj.key)
        total += max(0.0, future_obj.stall_sensitivity_ms if future_obj else obj.stall_sensitivity_ms)
    return total

def net_benefit_ms(obj):
    gross = obj.reuse_imminence * obj.stall_sensitivity_ms
    movement = 0.0 if obj.is_hbm_resident else obj.movement_cost_ms
    return gross - movement


def value_density_ms_per_byte(obj):
    return net_benefit_ms(obj) / max(1, obj.footprint_bytes)

def current_value_utility(obj):
    movement_cost = 0.0 if obj.is_hbm_resident else obj.movement_cost_ms
    return max(0.0, obj.reuse_imminence * obj.stall_sensitivity_ms - movement_cost)


def select_under_budget(objects, utilities, budget_bytes, exact_limit):
    candidates = [
        obj for obj in objects
        if obj.footprint_bytes > 0
        and obj.footprint_bytes <= budget_bytes
        and utilities.get(obj.key, 0.0) > 0.0
    ]

    if len(candidates) <= exact_limit:
        best_value = -1.0
        best_bytes = 0
        best = []
        n = len(candidates)
        for mask in range(1 << n):
            total_value = 0.0
            total_bytes = 0
            chosen = []
            for bit, obj in enumerate(candidates):
                if not (mask & (1 << bit)):
                    continue
                total_bytes += obj.footprint_bytes
                if total_bytes > budget_bytes:
                    break
                total_value += utilities[obj.key]
                chosen.append(obj)
            else:
                if (
                    total_value > best_value
                    or (math.isclose(total_value, best_value) and total_bytes < best_bytes)
                ):
                    best_value = total_value
                    best_bytes = total_bytes
                    best = chosen
        return best

    selected = []
    used = 0
    for obj in sorted(
        candidates,
        key=lambda o: (
            utilities[o.key] / max(1, o.footprint_bytes),
            utilities[o.key],
            -o.footprint_bytes,
        ),
        reverse=True,
    ):
        if used + obj.footprint_bytes <= budget_bytes:
            selected.append(obj)
            used += obj.footprint_bytes
    return selected

def select_ranked_under_budget(objects, rank_values, budget_bytes):
    selected = []
    used = 0

    for obj in sorted(
        [o for o in objects if o.footprint_bytes > 0 and o.footprint_bytes <= budget_bytes],
        key=lambda o: (rank_values.get(o.key, float("-inf")), -o.footprint_bytes),
        reverse=True,
    ):
        if used + obj.footprint_bytes <= budget_bytes:
            selected.append(obj)
            used += obj.footprint_bytes

    return selected


def select_policy(policy, objects, utilities, budget_bytes, static_kv_ratio, exact_limit):
    if policy == "runtime":
        return [obj for obj in objects if obj.is_hbm_resident]

    if policy == "static_split":
        kv_budget = int(budget_bytes * static_kv_ratio)
        emb_budget = budget_bytes - kv_budget
        kv_objects = [obj for obj in objects if obj.is_kv]
        emb_objects = [obj for obj in objects if obj.is_embedding]
        return (
            select_under_budget(kv_objects, utilities, kv_budget, exact_limit)
            + select_under_budget(emb_objects, utilities, emb_budget, exact_limit)
        )

    return select_under_budget(objects, utilities, budget_bytes, exact_limit)


def pct(values, ratio):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * ratio))]


def mib(value):
    return value / BYTES_PER_MIB


def summarize(decisions):
    grouped = defaultdict(list)
    for row in decisions:
        grouped[(row["budget_gib"], row["policy"])].append(row)

    summaries = []
    for (budget_gib, policy), rows in sorted(grouped.items()):
        total_eval = sum(r["eval_benefit_ms"] for r in rows)
        total_possible = sum(r["possible_benefit_ms"] for r in rows)
        summaries.append({
            "budget_gib": budget_gib,
            "policy": policy,
            "batches": len(rows),
            "mean_selected_mib": sum(r["selected_bytes"] for r in rows) / len(rows) / BYTES_PER_MIB,
            "mean_kv_mib": sum(r["selected_kv_bytes"] for r in rows) / len(rows) / BYTES_PER_MIB,
            "mean_embedding_mib": sum(r["selected_embedding_bytes"] for r in rows) / len(rows) / BYTES_PER_MIB,
            "total_eval_benefit_ms": total_eval,
            "mean_eval_benefit_ms": total_eval / len(rows),
            "mean_possible_benefit_ms": total_possible / len(rows),
            "mean_benefit_coverage": (total_eval / total_possible) if total_possible > 0 else 0.0,
            "total_churn_mib": sum(r["churn_in_bytes"] + r["churn_out_bytes"] for r in rows) / BYTES_PER_MIB,
            "infeasible_batches": sum(1 for r in rows if r["selected_bytes"] > r["budget_bytes"]),
            "p95_selected_mib": pct([mib(r["selected_bytes"]) for r in rows], 0.95),
            "feasible_batches": sum(1 for r in rows if r["is_feasible"]),
            "infeasible_batches": sum(1 for r in rows if not r["is_feasible"]),
            "mean_budget_violation_mib": sum(r["budget_violation_bytes"] for r in rows) / len(rows) / BYTES_PER_MIB,
            "max_budget_violation_mib": max(r["budget_violation_bytes"] for r in rows) / BYTES_PER_MIB,
        })
    return summaries


def parse_budget_sweep(args):
    if args.budget_sweep_gib:
        return [float(x.strip()) for x in args.budget_sweep_gib.split(",") if x.strip()]
    return [args.managed_hbm_gib]


def main():
    parser = argparse.ArgumentParser(
        description="Offline trace-driven HotState allocator simulator"
    )
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--snapshot", default="post", choices=["pre", "post", "state_trace", "post_state_trace"])
    parser.add_argument("--managed-hbm-gib", type=float, default=1.0)
    parser.add_argument("--budget-sweep-gib", default=None)
    parser.add_argument("--lookahead-batches", type=int, default=32)
    parser.add_argument("--static-kv-ratio", type=float, default=0.5)
    parser.add_argument("--exact-limit", type=int, default=16)
    parser.add_argument(
        "--policies",
        default="runtime,static_split,popularity,value,net_benefit,density,score,score_rank,oracle",
        help="comma-separated: runtime,static_split,popularity,value,oracle",
    )
    parser.add_argument("--out-summary-json", default=None)
    parser.add_argument("--out-summary-csv", default=None)
    parser.add_argument("--out-decisions-jsonl", default=None)
    args = parser.parse_args()

    records = load_records(args.trace_jsonl)
    field = snapshot_field(args.snapshot)
    snapshots = [normalize_snapshot(r, field) for r in records]
    snapshots_by_key = [{obj.key: obj for obj in objs} for objs in snapshots]

    budgets_gib = parse_budget_sweep(args)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    valid = {"runtime", "static_split", "popularity", "value","net_benefit", "density","score", "score_rank", "oracle"}
    unknown = [p for p in policies if p not in valid]
    if unknown:
        raise SystemExit(f"unknown policies: {unknown}")

    previous_selected = {}
    decisions = []

    for i, (record, objects) in enumerate(zip(records, snapshots)):
        oracle_eval = {
            obj.key: future_stall_benefit(
                obj, i, records, snapshots_by_key, args.lookahead_batches
            )
            for obj in objects
        }
        possible_benefit = sum(max(0.0, v) for v in oracle_eval.values())

        utilities_by_policy = {
            "runtime": oracle_eval,
            "static_split": {obj.key: current_value_utility(obj) for obj in objects},
            "value": {obj.key: current_value_utility(obj) for obj in objects},
            "net_benefit": {obj.key: max(0.0, net_benefit_ms(obj)) for obj in objects},
            "density": {obj.key: max(0.0, value_density_ms_per_byte(obj)) for obj in objects},
            "score": {obj.key: obj.score for obj in objects},
            "score_rank": {obj.key: obj.score for obj in objects},
            "popularity": {
                obj.key: float(future_access_count(obj, i, records, args.lookahead_batches))
                for obj in objects
            },
            "oracle": {
                obj.key: max(
                    0.0,
                    oracle_eval[obj.key] - (0.0 if obj.is_hbm_resident else obj.movement_cost_ms),
                )
                for obj in objects
            },
        }

        object_by_key = {obj.key: obj for obj in objects}

        for budget_gib in budgets_gib:
            budget_bytes = int(budget_gib * BYTES_PER_GIB)
            for policy in policies:
                if policy in ("score_rank", "density"):
                    selected = select_ranked_under_budget(
                        objects,
                        utilities_by_policy[policy],
                        budget_bytes,
                    )
                else:
                    selected = select_policy(
                        policy,
                        objects,
                        utilities_by_policy[policy],
                        budget_bytes,
                        args.static_kv_ratio,
                        args.exact_limit,
                    )
                selected_by_key = {obj.key: obj for obj in selected}
                prev = previous_selected.get((budget_gib, policy), {})

                entered = set(selected_by_key) - set(prev)
                left = set(prev) - set(selected_by_key)

                churn_in = sum(selected_by_key[k].footprint_bytes for k in entered)
                churn_out = sum(prev[k] for k in left)

                selected_bytes = sum(obj.footprint_bytes for obj in selected)
                selected_kv_bytes = sum(obj.footprint_bytes for obj in selected if obj.is_kv)
                selected_emb_bytes = sum(obj.footprint_bytes for obj in selected if obj.is_embedding)
                eval_benefit = sum(oracle_eval.get(obj.key, 0.0) for obj in selected)
                budget_violation_bytes = max(0, selected_bytes - budget_bytes)
                is_feasible = budget_violation_bytes == 0

                decisions.append({
                    "batch_idx": record.get("batch_idx", i),
                    "trace_line": record.get("_line_no"),
                    "budget_gib": budget_gib,
                    "budget_bytes": budget_bytes,
                    "policy": policy,
                    "latency_ms": record.get("latency_ms"),
                    "seq_history_len": record.get("seq_history_len"),
                    "user_id": record.get("user_id"),
                    "selected_count": len(selected),
                    "selected_bytes": selected_bytes,
                    "selected_kv_bytes": selected_kv_bytes,
                    "selected_embedding_bytes": selected_emb_bytes,
                    "eval_benefit_ms": eval_benefit,
                    "possible_benefit_ms": possible_benefit,
                    "benefit_coverage": (eval_benefit / possible_benefit) if possible_benefit > 0 else 0.0,
                    "churn_in_bytes": churn_in,
                    "churn_out_bytes": churn_out,
                    "selected_keys": sorted(selected_by_key),
                    "is_feasible": is_feasible,
                    "budget_violation_bytes": budget_violation_bytes,
                })

                previous_selected[(budget_gib, policy)] = {
                    k: object_by_key[k].footprint_bytes for k in selected_by_key
                }

    summaries = summarize(decisions)

    oracle_by_budget = {
        s["budget_gib"]: s["total_eval_benefit_ms"]
        for s in summaries
        if s["policy"] == "oracle"
    }
    for s in summaries:
        oracle_total = oracle_by_budget.get(s["budget_gib"], 0.0)

        if s["policy"] == "oracle":
            s["gap_vs_oracle"] = 0.0
        elif s["infeasible_batches"] > 0:
            s["gap_vs_oracle"] = None
        elif oracle_total > 0:
            s["gap_vs_oracle"] = 1.0 - s["total_eval_benefit_ms"] / oracle_total
        else:
            s["gap_vs_oracle"] = None

    payload = {
        "trace_jsonl": args.trace_jsonl,
        "snapshot": field,
        "lookahead_batches": args.lookahead_batches,
        "static_kv_ratio": args.static_kv_ratio,
        "summaries": summaries,
    }

    if args.out_summary_json:
        with open(args.out_summary_json, "w") as f:
            json.dump(payload, f, indent=2)

    if args.out_summary_csv:
        with open(args.out_summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    if args.out_decisions_jsonl:
        with open(args.out_decisions_jsonl, "w") as f:
            for row in decisions:
                f.write(json.dumps(row) + "\n")

    print("budget_gib policy          mean_eval_ms coverage gap_vs_oracle mean_sel_mib kv_mib emb_mib churn_mib infeasible")
    for s in summaries:
        gap = "n/a" if s["gap_vs_oracle"] is None else f"{s['gap_vs_oracle']:.3f}"
        print(
            f"{s['budget_gib']:9.3f} "
            f"{s['policy']:14s} "
            f"{s['mean_eval_benefit_ms']:12.4f} "
            f"{s['mean_benefit_coverage']:8.3f} "
            f"{gap:>13s} "
            f"{s['mean_selected_mib']:12.2f} "
            f"{s['mean_kv_mib']:7.2f} "
            f"{s['mean_embedding_mib']:7.2f} "
            f"{s['total_churn_mib']:9.2f} "
            f"{s['infeasible_batches']:10d}"
        )


if __name__ == "__main__":
    main()
