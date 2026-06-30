
import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

_HSTU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HSTU_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "commons"))
sys.path.insert(0, _HSTU_DIR)
sys.path.insert(0, os.path.join(_HSTU_DIR, "model"))

from commons.datasets import get_data_loader
from commons.datasets.hstu_batch import FeatureConfig
from commons.datasets.random_inference_dataset import RandomInferenceDataset
from configs import InferenceEmbeddingConfig, RankingConfig, get_inference_hstu_config, get_kvcache_config
from inference_ranking_gr import get_inference_ranking_gr

DEFAULT_KV_PAGE_SIZE = 32
DEFAULT_OFFLOAD_CHUNKSIZE = 8192
DEFAULT_SPLITS = [(20,80),(30,70),(40,60),(50,50),(60,40),(70,30),(80,20)]


def pct(values, ratio):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * ratio))]


def stats(records):
    lats = [r["latency_ms"] for r in records if "latency_ms" in r]
    return {
        "mean": sum(lats) / len(lats) if lats else float("nan"),
        "p50": pct(lats, 0.50),
        "p95": pct(lats, 0.95),
        "p99": pct(lats, 0.99),
        "max": max(lats) if lats else float("nan"),
        "n": len(lats),
    }


def parse_splits(text):
    result = []
    for part in text.split(","):
        lhs, rhs = part.strip().split(":")
        result.append((int(lhs), int(rhs)))
    return result


def count_dataset_batches(max_history_length, max_incremental_seqlen, num_users):
    return len(range(max_incremental_seqlen, max_history_length, max_incremental_seqlen)) * num_users


def build_dataset(max_history_length, max_num_candidates, max_incremental_seqlen, num_users):
    max_seqlen = max_history_length * 2 + max_num_candidates
    feature_configs = [
        FeatureConfig(
            feature_names=["item_feat", "act_feat"],
            max_item_ids=[10_000_000 - 1, 128 - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        )
    ]
    total_batches = count_dataset_batches(max_history_length, max_incremental_seqlen, num_users)
    dataset = RandomInferenceDataset(
        feature_configs=feature_configs,
        item_feature_name="item_feat",
        contextual_feature_names=[],
        action_feature_name="act_feat",
        max_num_users=num_users,
        max_batch_size=1,
        max_history_length=max_history_length,
        max_num_candidates=max_num_candidates,
        max_incremental_seqlen=max_incremental_seqlen,
        max_num_cached_batches=total_batches,
        full_mode=True,
    )
    return dataset, total_batches


def compute_split_params(total_hbm_bytes, emb_pct, kv_pct, num_layers, num_heads, head_dim, dtype):
    bytes_per_elem = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    kv_page_bytes = num_layers * 2 * DEFAULT_KV_PAGE_SIZE * num_heads * head_dim * bytes_per_elem
    emb_budget = math.floor(emb_pct / (emb_pct + kv_pct) * total_hbm_bytes)
    kv_budget = total_hbm_bytes - emb_budget
    blocks = max(1, kv_budget // kv_page_bytes)
    return emb_budget, kv_budget, blocks


def build_model(hidden_dim, num_layers, num_heads, head_dim, dtype, max_seqlen, blocks):
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=1,
        max_seq_len=max_seqlen * 2,
        dtype=dtype,
    )
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks,
        page_size=DEFAULT_KV_PAGE_SIZE,
        offload_chunksize=DEFAULT_OFFLOAD_CHUNKSIZE,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=["item_feat"],
            table_name="item",
            vocab_size=10_000_000,
            dim=hidden_dim,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=128,
            dim=hidden_dim,
            use_dynamicemb=False,
        ),
    ]
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[512, 8],
        num_tasks=8,
    )
    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=False,
    )
    if dtype == torch.bfloat16:
        model.bfloat16()
    model.eval()
    return model


def teardown_model(model):
    try:
        kvcache = model.dense_module.async_kvcache
        kvcache.executor.shutdown(wait=True, cancel_futures=True)
        kvcache.onload_worker.shutdown(wait=True, cancel_futures=True)
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_forward(model, batch, uids, thl, measure):
    if measure:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    with torch.inference_mode():
        model.forward_with_kvcache(batch, uids, thl)
    if measure:
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0
    return None


def run_static_split(args, budget_gib, split, out_path):
    dtype = torch.bfloat16 if args.dtype in ("bfloat16", "float16") else torch.float32
    total_hbm_bytes = int(budget_gib * 1024**3)
    max_seqlen = args.max_history_seqlen * 2 + args.max_num_candidates
    emb_budget, kv_budget, blocks = compute_split_params(
        total_hbm_bytes, split[0], split[1],
        args.num_layers, args.num_heads, args.head_dim, dtype,
    )

    dataset, total_available = build_dataset(
        args.max_history_seqlen,
        args.max_num_candidates,
        args.max_incremental_seqlen,
        args.num_users,
    )
    warmup_batches = max(1, int(total_available * args.warmup_ratio))
    measure_batches = total_available - warmup_batches

    print(f"budget={budget_gib:.2f} split={split[0]}:{split[1]} blocks={blocks}")
    model = build_model(args.hidden_dim, args.num_layers, args.num_heads, args.head_dim, dtype, max_seqlen, blocks)
    it = iter(get_data_loader(dataset))

    for _ in range(warmup_batches):
        batch, uids, thl = next(it)
        run_forward(model, batch, uids, thl, False)

    records = []
    for batch_idx in range(measure_batches):
        batch, uids, thl = next(it)
        latency_ms = run_forward(model, batch, uids, thl, True)
        hist_len = int(thl[0].item()) // 2
        records.append({
            "record_type": "static",
            "status": "ok",
            "budget_gib": budget_gib,
            "split": f"static_{split[0]}_{split[1]}",
            "split_lhs": split[0],
            "split_rhs": split[1],
            "emb_budget_bytes": emb_budget,
            "kv_budget_bytes": kv_budget,
            "blocks_in_primary_pool": blocks,
            "batch_idx": batch_idx,
            "latency_ms": latency_ms,
            "seq_history_len": hist_len,
            "user_id": int(uids[0].item()),
        })
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx+1}/{measure_batches}] lat={latency_ms:.2f}ms hist={hist_len}")

    with out_path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    teardown_model(model)
    return records


def split_key(record):
    return (record["split_lhs"], record["split_rhs"])


class ThresholdPolicy:
    def __init__(self, default, thresholds):
        self.default = default
        self.thresholds = sorted(thresholds, key=lambda x: x[0])

    def get_split(self, hist_len):
        selected = self.default
        for boundary, split in self.thresholds:
            if hist_len >= boundary:
                selected = split
        return selected

    def describe(self):
        parts = [f"default={self.default[0]}:{self.default[1]}"]
        for boundary, split in self.thresholds:
            parts.append(f"hist>={boundary}->{split[0]}:{split[1]}")
        return "; ".join(parts)


def derive_policy(records, bucket_size, margin_ms, discovery_ratio):
    max_batch = max(r["batch_idx"] for r in records)
    discovery_cutoff = int((max_batch + 1) * discovery_ratio)
    discovery = [r for r in records if r["batch_idx"] < discovery_cutoff]

    by_bucket = defaultdict(lambda: defaultdict(list))
    for r in discovery:
        bucket = (r["seq_history_len"] // bucket_size) * bucket_size
        by_bucket[bucket][split_key(r)].append(r["latency_ms"])

    bucket_winners = []
    previous_split = None
    previous_mean = None
    for bucket in sorted(by_bucket):
        means = {
            split: sum(values) / len(values)
            for split, values in by_bucket[bucket].items()
        }
        ranked = sorted(means.items(), key=lambda x: x[1])
        best_split, best_mean = ranked[0]
        if previous_split is not None and previous_split in means:
            if means[previous_split] - best_mean <= margin_ms:
                best_split = previous_split
                best_mean = means[previous_split]
        bucket_winners.append((bucket, best_split, best_mean))
        previous_split = best_split
        previous_mean = best_mean

    if not bucket_winners:
        raise RuntimeError("No bucket winners derived")

    default = bucket_winners[0][1]
    thresholds = []
    last_split = default
    for bucket, winner, _ in bucket_winners[1:]:
        if winner != last_split:
            thresholds.append((bucket, winner))
            last_split = winner

    return ThresholdPolicy(default, thresholds), discovery_cutoff, bucket_winners


def eval_static_and_oracle(records, discovery_cutoff, protocol):
    eval_records = records if protocol == "same" else [r for r in records if r["batch_idx"] >= discovery_cutoff]

    by_split = defaultdict(list)
    by_batch = defaultdict(list)
    for r in eval_records:
        by_split[r["split"]].append(r)
        by_batch[r["batch_idx"]].append(r)

    best_name, best_records = min(by_split.items(), key=lambda item: stats(item[1])["mean"])
    oracle_records = [min(v, key=lambda r: r["latency_ms"]) for v in by_batch.values()]
    return best_name, stats(best_records), stats(oracle_records)


def run_adaptive(args, budget_gib, policy, discovery_cutoff, protocol, out_path):
    dtype = torch.bfloat16 if args.dtype in ("bfloat16", "float16") else torch.float32
    total_hbm_bytes = int(budget_gib * 1024**3)
    max_seqlen = args.max_history_seqlen * 2 + args.max_num_candidates

    dataset, total_available = build_dataset(
        args.max_history_seqlen,
        args.max_num_candidates,
        args.max_incremental_seqlen,
        args.num_users,
    )
    warmup_batches = max(1, int(total_available * args.warmup_ratio))
    measure_batches = total_available - warmup_batches

    current_split = policy.default
    emb_budget, kv_budget, blocks = compute_split_params(
        total_hbm_bytes, current_split[0], current_split[1],
        args.num_layers, args.num_heads, args.head_dim, dtype,
    )
    model = build_model(args.hidden_dim, args.num_layers, args.num_heads, args.head_dim, dtype, max_seqlen, blocks)
    it = iter(get_data_loader(dataset))

    for _ in range(warmup_batches):
        batch, uids, thl = next(it)
        run_forward(model, batch, uids, thl, False)

    records = []
    switch_count = 0
    measured_idx = 0
    while measured_idx < measure_batches:
        batch, uids, thl = next(it)
        hist_len = int(thl[0].item()) // 2
        target_split = policy.get_split(hist_len)

        if target_split != current_split and (measure_batches - measured_idx) > args.num_users:
            teardown_model(model)
            emb_budget, kv_budget, blocks = compute_split_params(
                total_hbm_bytes, target_split[0], target_split[1],
                args.num_layers, args.num_heads, args.head_dim, dtype,
            )
            model = build_model(args.hidden_dim, args.num_layers, args.num_heads, args.head_dim, dtype, max_seqlen, blocks)
            current_split = target_split
            switch_count += 1

            for w in range(args.num_users):
                run_forward(model, batch, uids, thl, False)
                if w < args.num_users - 1:
                    batch, uids, thl = next(it)
            measure_batches -= args.num_users
            continue

        should_measure = protocol == "same" or measured_idx >= discovery_cutoff
        latency_ms = run_forward(model, batch, uids, thl, should_measure)
        if should_measure:
            records.append({
                "record_type": "adaptive",
                "status": "ok",
                "budget_gib": budget_gib,
                "split": f"adaptive_{current_split[0]}_{current_split[1]}",
                "split_lhs": current_split[0],
                "split_rhs": current_split[1],
                "emb_budget_bytes": emb_budget,
                "kv_budget_bytes": kv_budget,
                "blocks_in_primary_pool": blocks,
                "batch_idx": measured_idx,
                "latency_ms": latency_ms,
                "seq_history_len": hist_len,
                "user_id": int(uids[0].item()),
                "switch_count": switch_count,
            })
        measured_idx += 1

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    teardown_model(model)
    return records, switch_count


def run_budget(args, budget_gib, splits, out_dir):
    tag = f"{budget_gib:.2f}".replace(".", "p")
    static_path = out_dir / f"static_budget_{tag}.jsonl"
    adaptive_path = out_dir / f"adaptive_budget_{tag}_{args.eval_protocol}.jsonl"
    policy_path = out_dir / f"policy_budget_{tag}.json"
    static_path.unlink(missing_ok=True)

    all_static = []
    for split in splits:
        all_static.extend(run_static_split(args, budget_gib, split, static_path))

    policy, discovery_cutoff, bucket_winners = derive_policy(
        all_static,
        args.bucket_size,
        args.margin_ms,
        args.discovery_ratio,
    )

    with policy_path.open("w") as f:
        json.dump({
            "budget_gib": budget_gib,
            "eval_protocol": args.eval_protocol,
            "discovery_ratio": args.discovery_ratio,
            "discovery_cutoff": discovery_cutoff,
            "policy": policy.describe(),
            "bucket_winners": [
                {"bucket": b, "split": f"{s[0]}:{s[1]}", "mean_ms": m}
                for b, s, m in bucket_winners
            ],
        }, f, indent=2)

    best_name, best_static_stats, oracle_stats = eval_static_and_oracle(
        all_static,
        discovery_cutoff,
        args.eval_protocol,
    )
    adaptive_records, switch_count = run_adaptive(
        args,
        budget_gib,
        policy,
        discovery_cutoff,
        args.eval_protocol,
        adaptive_path,
    )
    adaptive_stats = stats(adaptive_records)

    gain = (
        (best_static_stats["mean"] - adaptive_stats["mean"]) / best_static_stats["mean"] * 100.0
        if best_static_stats["mean"] > 0 else float("nan")
    )
    oracle_gap = (
        (adaptive_stats["mean"] - oracle_stats["mean"]) / oracle_stats["mean"] * 100.0
        if oracle_stats["mean"] > 0 else float("nan")
    )

    return {
        "budget_gib": budget_gib,
        "eval_protocol": args.eval_protocol,
        "best_static": best_name,
        "best_static_stats": best_static_stats,
        "adaptive_stats": adaptive_stats,
        "oracle_stats": oracle_stats,
        "gain_vs_best_static_pct": gain,
        "gap_to_oracle_pct": oracle_gap,
        "switch_count": switch_count,
        "policy": policy.describe(),
        "static_jsonl": str(static_path),
        "adaptive_jsonl": str(adaptive_path),
        "policy_json": str(policy_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-history-seqlen", type=int, default=4096)
    parser.add_argument("--max-num-candidates", type=int, default=100)
    parser.add_argument("--max-incremental-seqlen", type=int, default=64)
    parser.add_argument("--num-users", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--budgets-gib", type=str, default="0.05,0.10,0.20,0.40,1.00")
    parser.add_argument("--splits", type=str, default="20:80,30:70,40:60,50:50,60:40,70:30,80:20")
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--margin-ms", type=float, default=0.05)
    parser.add_argument("--eval-protocol", choices=["heldout", "same"], default="heldout")
    parser.add_argument("--discovery-ratio", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="./logs/tight_hbm")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    summary_path.unlink(missing_ok=True)

    budgets = [float(x) for x in args.budgets_gib.split(",")]
    splits = parse_splits(args.splits)

    summaries = []
    for budget_gib in budgets:
        print("=" * 80)
        print(f"Budget {budget_gib:.2f} GiB")
        summary = run_budget(args, budget_gib, splits, out_dir)
        summaries.append(summary)
        with summary_path.open("a") as f:
            f.write(json.dumps(summary) + "\n")

    print("=" * 80)
    print("SUMMARY")
    print("budget best_static best_mean adaptive_mean oracle_mean gain_pct oracle_gap_pct p99_adapt max_adapt policy")
    for s in summaries:
        print(
            f"{s['budget_gib']:.2f} "
            f"{s['best_static']} "
            f"{s['best_static_stats']['mean']:.3f} "
            f"{s['adaptive_stats']['mean']:.3f} "
            f"{s['oracle_stats']['mean']:.3f} "
            f"{s['gain_vs_best_static_pct']:.2f} "
            f"{s['gap_to_oracle_pct']:.2f} "
            f"{s['adaptive_stats']['p99']:.3f} "
            f"{s['adaptive_stats']['max']:.3f} "
            f"{s['policy']}"
        )
    print(f"summary_jsonl={summary_path}")


if __name__ == "__main__":
    main()
