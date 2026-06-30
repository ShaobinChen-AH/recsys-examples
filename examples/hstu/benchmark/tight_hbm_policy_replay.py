#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import defaultdict


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "latency_ms" in row and "split" in row:
                records.append(row)
    return records


def parse_split(split_name):
    if not split_name.startswith("static_"):
        raise ValueError(f"unsupported split name: {split_name}")
    _, lhs, rhs = split_name.split("_")
    return int(lhs), int(rhs)


def split_to_ratio(split_name):
    lhs, rhs = parse_split(split_name)
    return f"{lhs}:{rhs}"


def kv_ratio(split_name):
    return parse_split(split_name)[1]


def pct(sorted_vals, ratio):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * ratio))
    return sorted_vals[idx]


def compute_stats(latencies):
    vals = sorted(float(x) for x in latencies)
    n = len(vals)
    if n == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    return {
        "mean": sum(vals) / n,
        "p50": pct(vals, 0.50),
        "p95": pct(vals, 0.95),
        "p99": pct(vals, 0.99),
        "max": vals[-1],
        "n": n,
    }


def better_stats(lhs, rhs):
    if rhs is None:
        return True
    lhs_key = (lhs["mean"], lhs["p95"], lhs["max"])
    rhs_key = (rhs["mean"], rhs["p95"], rhs["max"])
    return lhs_key < rhs_key


def sample_key(row):
    return (
        int(row["batch_idx"]),
        int(row["user_id"]),
        int(row["seq_history_len"]),
    )


def build_sample_maps(records):
    lat_by_sample = defaultdict(dict)
    meta_by_sample = {}
    split_names = set()

    for row in records:
        key = sample_key(row)
        split = row["split"]
        split_names.add(split)
        lat_by_sample[key][split] = float(row["latency_ms"])
        meta_by_sample[key] = {
            "batch_idx": int(row["batch_idx"]),
            "user_id": int(row["user_id"]),
            "seq_history_len": int(row["seq_history_len"]),
        }

    splits = sorted(split_names, key=lambda s: (kv_ratio(s), parse_split(s)[0]))
    return lat_by_sample, meta_by_sample, splits


def stratified_partition(meta_by_sample):
    keys_by_hist = defaultdict(list)
    for key, meta in meta_by_sample.items():
        keys_by_hist[meta["seq_history_len"]].append(key)

    discovery = []
    evaluation = []
    counts = {}

    for hist in sorted(keys_by_hist):
        keys = sorted(keys_by_hist[hist], key=lambda k: (k[0], k[1]))
        disc = keys[::2]
        eva = keys[1::2]
        discovery.extend(disc)
        evaluation.extend(eva)
        counts[hist] = {
            "discovery": len(disc),
            "evaluation": len(eva),
        }

    return discovery, evaluation, counts


def build_hist_map(keys, lat_by_sample, meta_by_sample):
    hist_map = defaultdict(lambda: defaultdict(list))
    count_by_hist = defaultdict(int)

    for key in keys:
        hist = meta_by_sample[key]["seq_history_len"]
        count_by_hist[hist] += 1
        for split, lat in lat_by_sample[key].items():
            hist_map[hist][split].append(lat)

    return hist_map, count_by_hist


def choose_split(policy, hist_len):
    chosen = policy["default"]
    for threshold, split in policy["rules"]:
        if hist_len >= threshold:
            chosen = split
        else:
            break
    return chosen


def format_policy(policy):
    parts = [f"default={split_to_ratio(policy['default'])}"]
    for threshold, split in policy["rules"]:
        parts.append(f"hist>={threshold}->{split_to_ratio(split)}")
    return "; ".join(parts)


def parse_policy_string(policy_str):
    default = None
    rules = []
    for raw in policy_str.split(";"):
        token = raw.strip()
        if not token:
            continue
        if token.startswith("default="):
            ratio = token.split("=", 1)[1].strip()
            lhs, rhs = ratio.split(":")
            default = f"static_{int(lhs)}_{int(rhs)}"
        elif token.startswith("hist>="):
            left, right = token.split("->", 1)
            threshold = int(left.split(">=", 1)[1].strip())
            lhs, rhs = right.strip().split(":")
            rules.append((threshold, f"static_{int(lhs)}_{int(rhs)}"))
        else:
            raise ValueError(f"bad policy token: {token}")

    if default is None:
        raise ValueError("policy must contain default=...")

    rules.sort(key=lambda x: x[0])
    return {"default": default, "rules": rules}


def eval_policy_hist(policy, hist_map, count_by_hist):
    latencies = []
    missing = 0

    for hist in sorted(hist_map):
        split = choose_split(policy, hist)
        chosen = hist_map[hist].get(split)
        if not chosen:
            missing += count_by_hist[hist]
            continue
        latencies.extend(chosen)

    return compute_stats(latencies), missing


def eval_policy_samples(policy, keys, lat_by_sample, meta_by_sample):
    latencies = []
    rows = []
    missing = 0

    for key in keys:
        meta = meta_by_sample[key]
        chosen_split = choose_split(policy, meta["seq_history_len"])
        chosen_lat = lat_by_sample[key].get(chosen_split)
        if chosen_lat is None:
            missing += 1
            continue

        oracle_split, oracle_lat = min(lat_by_sample[key].items(), key=lambda kv: kv[1])
        latencies.append(chosen_lat)
        rows.append({
            "batch_idx": meta["batch_idx"],
            "user_id": meta["user_id"],
            "seq_history_len": meta["seq_history_len"],
            "chosen_split": chosen_split,
            "chosen_latency_ms": chosen_lat,
            "oracle_split": oracle_split,
            "oracle_latency_ms": oracle_lat,
            "excess_latency_ms": chosen_lat - oracle_lat,
        })

    return compute_stats(latencies), rows, missing


def eval_oracle(keys, lat_by_sample):
    vals = []
    for key in keys:
        vals.append(min(lat_by_sample[key].values()))
    return compute_stats(vals)


def select_best_static(keys, lat_by_sample, splits):
    best_split = None
    best_stats = None
    all_stats = {}

    for split in splits:
        vals = [lat_by_sample[key][split] for key in keys if split in lat_by_sample[key]]
        stats = compute_stats(vals)
        all_stats[split] = stats
        if better_stats(stats, best_stats):
            best_split = split
            best_stats = stats

    return best_split, best_stats, all_stats


def search_one_threshold(discovery_hist_map, discovery_count_by_hist, splits, thresholds):
    best = None
    for low in splits:
        for high in splits:
            if kv_ratio(high) < kv_ratio(low):
                continue
            for threshold in thresholds:
                policy = {"default": low, "rules": [(threshold, high)]}
                stats, missing = eval_policy_hist(policy, discovery_hist_map, discovery_count_by_hist)
                if missing != 0:
                    continue
                candidate = {
                    "kind": "1-threshold",
                    "policy": policy,
                    "discovery_stats": stats,
                }
                if best is None or better_stats(candidate["discovery_stats"], best["discovery_stats"]):
                    best = candidate
    return best


def search_two_threshold(discovery_hist_map, discovery_count_by_hist, splits, thresholds):
    best = None
    for split0 in splits:
        for split1 in splits:
            if kv_ratio(split1) < kv_ratio(split0):
                continue
            for split2 in splits:
                if kv_ratio(split2) < kv_ratio(split1):
                    continue
                for i, threshold1 in enumerate(thresholds):
                    for threshold2 in thresholds[i + 1:]:
                        policy = {
                            "default": split0,
                            "rules": [(threshold1, split1), (threshold2, split2)],
                        }
                        stats, missing = eval_policy_hist(policy, discovery_hist_map, discovery_count_by_hist)
                        if missing != 0:
                            continue
                        candidate = {
                            "kind": "2-threshold",
                            "policy": policy,
                            "discovery_stats": stats,
                        }
                        if best is None or better_stats(candidate["discovery_stats"], best["discovery_stats"]):
                            best = candidate
    return best


def bucket_winners(records, bucket_size):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in records:
        bucket = (int(row["seq_history_len"]) // bucket_size) * bucket_size
        buckets[bucket][row["split"]].append(float(row["latency_ms"]))

    out = []
    for bucket in sorted(buckets):
        means = {split: sum(vals) / len(vals) for split, vals in buckets[bucket].items()}
        ranked = sorted(means.items(), key=lambda kv: kv[1])
        best_split, best_lat = ranked[0]
        second_split, second_lat = ranked[1]
        out.append({
            "bucket_start": bucket,
            "best_split": best_split,
            "best_ratio": split_to_ratio(best_split),
            "best_mean_ms": best_lat,
            "second_split": second_split,
            "second_ratio": split_to_ratio(second_split),
            "second_mean_ms": second_lat,
            "margin_ms": second_lat - best_lat,
            "ranked": [
                {"split": split, "ratio": split_to_ratio(split), "mean_ms": lat}
                for split, lat in ranked
            ],
        })
    return out


def summarize_excess(rows):
    rows = sorted(rows, key=lambda r: r["excess_latency_ms"], reverse=True)
    total_excess = sum(max(0.0, r["excess_latency_ms"]) for r in rows)
    shares = {}
    for topk in (1, 5, 10):
        top_sum = sum(max(0.0, r["excess_latency_ms"]) for r in rows[:topk])
        shares[f"top_{topk}_share_pct"] = (100.0 * top_sum / total_excess) if total_excess > 0 else 0.0
    return rows[:20], shares


def main():
    parser = argparse.ArgumentParser(description="Tight-HBM stratified replay for simple threshold policies")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--budget-gib", type=float, required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-diagnosis-json", default=None)
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--reference-policy", type=str, default=None)
    args = parser.parse_args()

    records = load_records(args.jsonl)
    lat_by_sample, meta_by_sample, splits = build_sample_maps(records)
    discovery_keys, eval_keys, partition_counts = stratified_partition(meta_by_sample)

    discovery_hist_map, discovery_count_by_hist = build_hist_map(discovery_keys, lat_by_sample, meta_by_sample)

    thresholds = sorted(partition_counts.keys())[1:]

    best_static_eval_split, best_static_eval_stats, _ = select_best_static(eval_keys, lat_by_sample, splits)
    best_static_disc_split, _, _ = select_best_static(discovery_keys, lat_by_sample, splits)
    best_static_disc_eval_stats = compute_stats(
        [lat_by_sample[key][best_static_disc_split] for key in eval_keys]
    )

    oracle_eval_stats = eval_oracle(eval_keys, lat_by_sample)

    best1 = search_one_threshold(discovery_hist_map, discovery_count_by_hist, splits, thresholds)
    best2 = search_two_threshold(discovery_hist_map, discovery_count_by_hist, splits, thresholds)

    selected = best1
    if best2 is not None and better_stats(best2["discovery_stats"], best1["discovery_stats"]):
        selected = best2

    selected_eval_stats, selected_eval_rows, missing_eval_batches = eval_policy_samples(
        selected["policy"], eval_keys, lat_by_sample, meta_by_sample
    )

    best1_eval_stats, _, _ = eval_policy_samples(best1["policy"], eval_keys, lat_by_sample, meta_by_sample)
    best2_eval_stats, _, _ = eval_policy_samples(best2["policy"], eval_keys, lat_by_sample, meta_by_sample)

    reference = None
    if args.reference_policy:
        ref_policy = parse_policy_string(args.reference_policy)
        ref_disc_stats, ref_disc_missing = eval_policy_hist(ref_policy, discovery_hist_map, discovery_count_by_hist)
        ref_eval_stats, ref_eval_rows, ref_eval_missing = eval_policy_samples(
            ref_policy, eval_keys, lat_by_sample, meta_by_sample
        )
        reference = {
            "policy": format_policy(ref_policy),
            "discovery_stats": ref_disc_stats,
            "discovery_missing": ref_disc_missing,
            "eval_stats": ref_eval_stats,
            "eval_missing": ref_eval_missing,
        }

    gain_vs_best_static_pct = (
        100.0 * (best_static_eval_stats["mean"] - selected_eval_stats["mean"]) / best_static_eval_stats["mean"]
    )
    gap_to_oracle_pct = (
        100.0 * (selected_eval_stats["mean"] - oracle_eval_stats["mean"]) / oracle_eval_stats["mean"]
    )

    summary = {
        "budget_gib": args.budget_gib,
        "num_total_samples": len(meta_by_sample),
        "num_discovery_samples": len(discovery_keys),
        "num_eval_samples": len(eval_keys),
        "splits": splits,
        "best_static": best_static_eval_split,
        "best_static_stats": best_static_eval_stats,
        "best_static_discovery_selected": best_static_disc_split,
        "best_static_discovery_selected_eval_stats": best_static_disc_eval_stats,
        "oracle_stats": oracle_eval_stats,
        "best_1_threshold_policy": format_policy(best1["policy"]),
        "best_1_threshold_discovery_stats": best1["discovery_stats"],
        "best_1_threshold_eval_stats": best1_eval_stats,
        "best_2_threshold_policy": format_policy(best2["policy"]),
        "best_2_threshold_discovery_stats": best2["discovery_stats"],
        "best_2_threshold_eval_stats": best2_eval_stats,
        "selected_simple_policy_kind": selected["kind"],
        "selected_simple_policy": format_policy(selected["policy"]),
        "selected_simple_policy_discovery_stats": selected["discovery_stats"],
        "selected_simple_policy_eval_stats": selected_eval_stats,
        "gain_vs_best_static_pct": gain_vs_best_static_pct,
        "gap_to_oracle_pct": gap_to_oracle_pct,
        "missing_eval_batches": missing_eval_batches,
        "reference_policy_result": reference,
        "partition_counts_by_hist": partition_counts,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if args.out_diagnosis_json:
        top_rows, excess_shares = summarize_excess(selected_eval_rows)
        diagnosis = {
            "budget_gib": args.budget_gib,
            "selected_simple_policy_kind": selected["kind"],
            "selected_simple_policy": format_policy(selected["policy"]),
            "bucket_winners": bucket_winners(records, args.bucket_size),
            "top_excess_eval_rows": top_rows,
            "excess_share_summary": excess_shares,
        }
        with open(args.out_diagnosis_json, "w") as f:
            json.dump(diagnosis, f, indent=2)
        print(f"Wrote diagnosis: {args.out_diagnosis_json}")

    print(f"Wrote summary: {args.out_json}")


if __name__ == "__main__":
    main()
