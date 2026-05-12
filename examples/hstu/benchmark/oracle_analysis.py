"""Compute oracle latency from static sweep JSONL traces."""
import json
import argparse
from collections import defaultdict

def load_traces(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "latency_ms" in r:
                records.append(r)
    return records

def compute_oracle(records):
    # Group by (user_id, seq_history_len) — each batch is uniquely identified
    batch_key = lambda r: (r["user_id"], r["seq_history_len"])
    grouped = defaultdict(list)
    for r in records:
        grouped[batch_key(r)].append(r)

    oracle_records = []
    for key, recs in grouped.items():
        best = min(recs, key=lambda r: r["latency_ms"])
        oracle_records.append(best)

    oracle_mean = sum(r["latency_ms"] for r in oracle_records) / len(oracle_records)
    return oracle_mean, oracle_records

def best_static(records):
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r["latency_ms"])
    results = {}
    for split, lats in by_split.items():
        results[split] = sum(lats) / len(lats)
    return results

def per_bucket_analysis(records, show_margins=False):
    """Show which split wins for each history bucket, with optional margins."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in records:
        bucket = (r["seq_history_len"] // 256) * 256
        buckets[bucket][r["split"]].append(r["latency_ms"])

    for bucket in sorted(buckets):
        # Compute mean per split in this bucket
        means = {}
        for split, lats in buckets[bucket].items():
            means[split] = sum(lats) / len(lats)
        ranked = sorted(means.items(), key=lambda x: x[1])
        best_split, best_lat = ranked[0]

        if show_margins:
            entries = []
            for split, lat in ranked[:3]:
                margin = lat - best_lat
                entries.append(f"{split}={lat:.2f}ms [+{margin:.2f}]")
            print(f"  hist={bucket:5d}: {'  '.join(entries)}")
        else:
            print(f"  hist={bucket:5d}: best={best_split:>12s} ({best_lat:.2f}ms)")

def margin_summary(records):
    """Compute: for each split and history bucket, how often it's within noise of best."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in records:
        bucket = (r["seq_history_len"] // 256) * 256
        buckets[bucket][r["split"]].append(r["latency_ms"])

    threshold = 0.05  # ms — within noise
    contestable = 0
    clear = 0
    for bucket, splits in buckets.items():
        means = {s: sum(l)/len(l) for s, l in splits.items()}
        ranked = sorted(means.items(), key=lambda x: x[1])
        if len(ranked) >= 2 and (ranked[1][1] - ranked[0][1]) <= threshold:
            contestable += 1
        else:
            clear += 1

    print(f"\n=== Margin Summary ===")
    print(f"  Buckets with clear winner (margin > {threshold}ms):  {clear}")
    print(f"  Buckets contestable (margin ≤ {threshold}ms):        {contestable}")

def per_split_tail(records):
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r["latency_ms"])
    print(f"\n=== P99 and Max Per Split ===")
    header = f"{'Split':>16s} | {'P50':>7s} | {'P95':>7s} | {'P99':>7s} | {'Max':>8s} | {'Mean':>7s}"
    print(header)
    print("-" * len(header))
    for split in sorted(by_split.keys()):
        lats = sorted(by_split[split])
        n = len(lats)
        print(f"  {split:>12s}:  {lats[n//2]:6.2f}ms  {lats[int(n*0.95)]:6.2f}ms  "
              f"{lats[int(n*0.99)]:6.2f}ms  {max(lats):7.2f}ms  {sum(lats)/n:6.2f}ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to row2_trace.jsonl")
    parser.add_argument("--show-margins", action="store_true",
                    help="Show top-3 splits per bucket with margins")
    args = parser.parse_args()

    records = load_traces(args.jsonl)
    print(f"Loaded {len(records)} records")

    # Best static analysis
    static = best_static(records)
    print("\n=== Best Static ===")
    best_split = min(static, key=static.get)
    for split, mean in sorted(static.items()):
        marker = " <-- BEST" if split == best_split else ""
        print(f"  {split}: mean={mean:.2f}ms{marker}")

    # Oracle analysis
    oracle_mean, oracle_recs = compute_oracle(records)
    print(f"\n=== Oracle ===")
    print(f"  Mean latency: {oracle_mean:.2f}ms")
    print(f"  Improvement:  {(static[best_split] - oracle_mean):.2f}ms over best static")
    print(f"  Gain:         {(static[best_split] - oracle_mean)/static[best_split]*100:.1f}%")

    # Per-bucket breakdown
    print(f"\n=== Per-History-Bucket Winners ===")
    per_bucket_analysis(records, show_margins=args.show_margins)
    if args.show_margins:
        margin_summary(records)
    per_split_tail(records)

if __name__ == "__main__":
    main()
