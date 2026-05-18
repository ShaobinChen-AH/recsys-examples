# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 3: Online ε-Greedy Self-Discovering HBM Arbiter.

Discovers the optimal embedding:KV split threshold through online exploration,
then exploits it — all in a single integrated loop with no prior knowledge.

    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6064 \
      --module benchmark.unified_arbiter \
      --mode discover \
      --max-history-seqlen 4096 --max-num-candidates 100 --max-incremental-seqlen 64 \
      --num-users 8 --total-hbm-budget-gib 1.0 --warmup-ratio 0.1 \
      --out-jsonl ./logs/arbiter_phase3_trace.jsonl
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
_HSTU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HSTU_DIR, ".."))

sys.path.insert(0, os.path.join(_REPO_ROOT, "commons"))
sys.path.insert(0, _HSTU_DIR)
sys.path.insert(0, os.path.join(_HSTU_DIR, "model"))

from commons.datasets import get_data_loader
from commons.datasets.hstu_batch import FeatureConfig
from commons.datasets.random_inference_dataset import RandomInferenceDataset
from configs import (
    InferenceEmbeddingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from inference_ranking_gr import get_inference_ranking_gr

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_KV_PAGE_SIZE = 32
DEFAULT_OFFLOAD_CHUNKSIZE = 8192

# All 7 candidate splits, ordered by emb:kv
ALL_CANDIDATE_SPLITS = [
    (70, 30), (60, 40), (50, 50), (40, 60),
    (30, 70), (80, 20), (20, 80),
]
DEFAULT_INITIAL_SPLIT = (70, 30)


# ── Utility functions ────────────────────────────────────────────────────────

def count_dataset_batches(max_history_length, max_incremental_seqlen, num_users):
    num_steps = len(range(max_incremental_seqlen, max_history_length,
                          max_incremental_seqlen))
    return num_steps * num_users


def build_inference_model(hidden_dim, num_layers, num_heads, head_dim, dtype,
                          max_seqlen, blocks_in_primary_pool, max_batch_size=1):
    """Build an HSTU inference model for a given KV page budget."""
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim, num_layers=num_layers,
        num_attention_heads=num_heads, head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen * 2,       # ×2 for interleaved item+action
        dtype=dtype,
    )
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=DEFAULT_KV_PAGE_SIZE,
        offload_chunksize=DEFAULT_OFFLOAD_CHUNKSIZE,
    )
    emb_configs = [
        InferenceEmbeddingConfig(feature_names=["item_feat"], table_name="item",
            vocab_size=10_000_000, dim=hidden_dim, use_dynamicemb=True),
        InferenceEmbeddingConfig(feature_names=["act_feat"], table_name="act",
            vocab_size=128, dim=hidden_dim, use_dynamicemb=False),
    ]
    task_config = RankingConfig(
        embedding_configs=emb_configs, prediction_head_arch=[512, 8], num_tasks=8,
    )
    model = get_inference_ranking_gr(
        hstu_config=hstu_config, kvcache_config=kv_cache_config,
        task_config=task_config, use_cudagraph=False,
    )
    if dtype == torch.bfloat16:
        model.bfloat16()
    model.eval()
    return model


def build_dataset(max_history_length, max_num_candidates, max_incremental_seqlen,
                  num_users):
    max_seqlen = max_history_length * 2 + max_num_candidates
    feature_configs = [
        FeatureConfig(feature_names=["item_feat", "act_feat"],
            max_item_ids=[10_000_000 - 1, 128 - 1],
            max_sequence_length=max_seqlen, is_jagged=False),
    ]
    total_batches = count_dataset_batches(
        max_history_length, max_incremental_seqlen, num_users)
    dataset = RandomInferenceDataset(
        feature_configs=feature_configs,
        item_feature_name="item_feat", contextual_feature_names=[],
        action_feature_name="act_feat", max_num_users=num_users,
        max_batch_size=1, max_history_length=max_history_length,
        max_num_candidates=max_num_candidates,
        max_incremental_seqlen=max_incremental_seqlen,
        max_num_cached_batches=total_batches, full_mode=True,
    )
    return dataset, total_batches


def compute_split_params(total_hbm_bytes, emb_pct, kv_pct, num_layers,
                         num_heads, head_dim, page_size, dtype):
    bytes_per_elem = 2  # bf16
    kv_page_bytes = (num_layers * 2 * page_size * num_heads * head_dim
                     * bytes_per_elem)
    emb_budget = math.floor(emb_pct / (emb_pct + kv_pct) * total_hbm_bytes)
    kv_budget = total_hbm_bytes - emb_budget
    blocks = max(1, kv_budget // kv_page_bytes)
    return emb_budget, kv_budget, blocks


def build_model_for_split(split, hidden_dim, num_layers, num_heads, head_dim,
                          dtype, max_seqlen, total_hbm_bytes):
    """Build an inference model with the given (emb_pct, kv_pct) split."""
    _, _, blocks = compute_split_params(
        total_hbm_bytes, split[0], split[1],
        num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)
    return build_inference_model(hidden_dim, num_layers, num_heads, head_dim,
                                 dtype, max_seqlen, blocks)


def teardown_model(model):
    """Safely destroy model and release all GPU memory."""
    try:
        kvcache = model.dense_module.async_kvcache
        kvcache.executor.shutdown(wait=True)
        kvcache.onload_worker.shutdown(wait=True)
    except Exception:
        pass
    torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Threshold Policy (shared between Phase 2 and Phase 3) ───────────────────

class ThresholdPolicy:
    """Maps per-feature history length to (emb_pct, kv_pct) split."""

    def __init__(self, thresholds: List[Tuple[int, int, int]],
                 default: Tuple[int, int]):
        self.thresholds = sorted(thresholds, key=lambda t: t[0])
        self.default = default
        for hist, emb, kv in self.thresholds:
            if emb + kv != 100:
                raise ValueError(f"Threshold hist={hist} {emb}:{kv} != 100%")

    def get_split(self, hist_len: int) -> Tuple[int, int]:
        for boundary, emb_pct, kv_pct in self.thresholds:
            if hist_len >= boundary:
                return (emb_pct, kv_pct)
        return self.default

    def describe(self) -> str:
        if not self.thresholds:
            return f"always {self.default[0]}:{self.default[1]}"
        parts = [f"hist < {self.thresholds[0][0]} → "
                 f"{self.default[0]}:{self.default[1]}"]
        for b, e, k in self.thresholds:
            parts.append(f"hist ≥ {b} → {e}:{k}")
        return "; ".join(parts)


# ── Phase 2: exploit mode (threshold known a-priori) ────────────────────────

def run_exploit_arbiter(dataset, total_available, warmup_batches, measure_batches,
                        num_users, policy, out_jsonl,
                        hidden_dim, num_layers, num_heads, head_dim, dtype,
                        max_seqlen, total_hbm_bytes):
    """Run Phase 2: split-switching with pre-computed threshold."""
    current_split = policy.default
    emb_budget, kv_budget, blocks = compute_split_params(
        total_hbm_bytes, current_split[0], current_split[1],
        num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)
    model = build_inference_model(hidden_dim, num_layers, num_heads, head_dim,
                                  dtype, max_seqlen, blocks)

    print(f"=== Phase 2: Split-Switching Controller ===")
    print(f"Policy: {policy.describe()}")
    print(f"Initial: {current_split[0]}:{current_split[1]} ({blocks} KV pages)\n")

    dataloader = get_data_loader(dataset)
    dataloader_iter = iter(dataloader)

    print(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        batch, uids, thl = next(dataloader_iter)
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)

    trace_records = []
    switch_count = 0
    i = 0

    print(f"\nMeasuring (up to {measure_batches} batches)...")
    while i < measure_batches:
        batch, uids, thl = next(dataloader_iter)
        hist_len = thl[0].item() // 2

        target_split = policy.get_split(hist_len)

        if (target_split != current_split
                and (measure_batches - i) > num_users):
            teardown_model(model)
            torch.cuda.synchronize()

            emb_budget, kv_budget, blocks = compute_split_params(
                total_hbm_bytes, target_split[0], target_split[1],
                num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)
            model = build_inference_model(hidden_dim, num_layers, num_heads,
                                          head_dim, dtype, max_seqlen, blocks)
            current_split = target_split
            switch_count += 1

            print(f"\n  [Switch #{switch_count}] hist={hist_len}: "
                  f"→ {target_split[0]}:{target_split[1]} ({blocks} KV pages)")
            print(f"  Mini-warmup ({num_users} batches)...")

            for w in range(num_users):
                with torch.inference_mode():
                    model.forward_with_kvcache(batch, uids, thl)
                if w < num_users - 1:
                    batch, uids, thl = next(dataloader_iter)
            measure_batches -= num_users
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        user_id = int(uids[0].item())
        record = {
            "split": f"exploit_{current_split[0]}_{current_split[1]}",
            "split_lhs": current_split[0], "split_rhs": current_split[1],
            "emb_budget_bytes": emb_budget, "kv_budget_bytes": kv_budget,
            "blocks_in_primary_pool": blocks,
            "batch_idx": len(trace_records),
            "latency_ms": latency_ms,
            "seq_history_len": hist_len,
            "user_id": user_id,
        }
        trace_records.append(record)
        i += 1

        if i % 20 == 0:
            print(f"  [{i}/{measure_batches}] latency={latency_ms:.2f}ms "
                  f"hist={hist_len}")

    teardown_model(model)

    lats = sorted(r["latency_ms"] for r in trace_records)
    n = len(lats)

    def pct(sorted_vals, ratio):
        if not sorted_vals:
            return float("nan")
        idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * ratio)))
        return sorted_vals[idx]

    results = {
        "mean": sum(lats) / n if n else float("nan"),
        "p50": pct(lats, 0.50), "p95": pct(lats, 0.95),
        "p99": pct(lats, 0.99), "p99_9": pct(lats, 0.999),
        "max": max(lats) if n else float("nan"),
        "switch_count": switch_count, "num_records": n,
    }

    with open(out_jsonl, "w") as f:
        for r in trace_records:
            f.write(json.dumps(r) + "\n")

    return results


# ── Phase 3: Q-table and ε-greedy learning ──────────────────────────────────

class QTable:
    """Tracks running-mean latency per (history_bucket, split)."""

    def __init__(self, bucket_size=256):
        self.bucket_size = bucket_size
        self._data = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    def bucket(self, hist_len: int) -> int:
        return (hist_len // self.bucket_size) * self.bucket_size

    def update(self, bucket: int, split: Tuple[int, int], latency: float):
        s, c = self._data[bucket][split]
        self._data[bucket][split] = [s + latency, c + 1]

    def mean(self, bucket: int, split: Tuple[int, int]) -> float:
        s, c = self._data[bucket].get(split, [0.0, 0])
        return s / c if c > 0 else float("inf")

    def count(self, bucket: int, split: Tuple[int, int]) -> int:
        _, c = self._data[bucket].get(split, [0.0, 0])
        return c

    def best_split(self, bucket: int) -> Optional[Tuple[int, int]]:
        entries = self._data.get(bucket, {})
        if not entries:
            return None
        return min(entries, key=lambda k: entries[k][0] / max(1, entries[k][1]))

    def explored_count(self, bucket: int) -> int:
        return len(self._data.get(bucket, {}))

    def discovered_policy(self, margin_threshold=0.05):
        """Extract a simplified ThresholdPolicy from accumulated data."""
        from collections import Counter

        per_bucket = {}
        for bucket in sorted(self._data):
            entries = self._data[bucket]
            ranked = sorted(entries.items(),
                            key=lambda x: x[1][0] / max(1, x[1][1]))
            best_s, (best_sum, best_cnt) = ranked[0]
            best_mean = best_sum / best_cnt
            per_bucket[bucket] = (best_s, best_mean, ranked)

        buckets = sorted(per_bucket)
        if not buckets:
            return ThresholdPolicy([], DEFAULT_INITIAL_SPLIT)

        thresholds = []
        prev_split = per_bucket[buckets[0]][0]

        for bucket in buckets[1:]:
            cur_split, cur_mean, cur_ranked = per_bucket[bucket]
            if cur_split == prev_split:
                continue

            # Check if previous split is contestable at this bucket
            prev_mean = None
            for s, (sm, cnt) in cur_ranked:
                if s == prev_split:
                    prev_mean = sm / cnt
                    break
            if (prev_mean is not None
                    and (prev_mean - cur_mean) <= margin_threshold
                    and cur_mean >= 0):
                per_bucket[bucket] = (prev_split, prev_mean, cur_ranked)
                continue

            thresholds.append((bucket, cur_split[0], cur_split[1]))
            prev_split = cur_split

        split_counts = Counter(v[0] for v in per_bucket.values())
        default = split_counts.most_common(1)[0][0]
        return ThresholdPolicy(thresholds, default)

    def summary(self):
        """Print a compact Q-table summary."""
        print(f"\n=== Q-Table Summary ===")
        header = (f"{'Bucket':>6s} | {'Samples':>7s} | {'Best':>8s} "
                  f"| {'Mean':>6s} | {'Runner-up':>8s} | {'2nd':>6s}")
        print(header)
        print("-" * len(header))
        for bucket in sorted(self._data):
            entries = self._data[bucket]
            total = sum(c for _, c in entries.values())
            ranked = sorted(entries.items(),
                            key=lambda x: x[1][0] / max(1, x[1][1]))
            best_s, (bst_sum, bst_cnt) = ranked[0]
            best_m = bst_sum / bst_cnt
            ru_s = f"{ranked[1][0][0]}:{ranked[1][0][1]}" if len(ranked) > 1 else "-"
            ru_m = ranked[1][1][0]/max(1,ranked[1][1][1]) if len(ranked)>1 else 0
            print(f"  {bucket:4d}: {total:6d}  "
                  f"{best_s[0]:>3d}:{best_s[1]:<3d} {best_m:5.2f}ms  "
                  f"{ru_s:>8s} {ru_m:5.2f}ms")


def epsilon_schedule(measured_idx: int, max_explore: int = 300) -> float:
    """Linear decay from 0.5 to 0.0 over max_explore measured batches."""
    if measured_idx >= max_explore:
        return 0.0
    return 0.5 * (1.0 - measured_idx / max_explore)

def run_hotstate_arbiter(dataset, total_available, warmup_batches,
                         measure_batches, num_users, out_jsonl,
                         hidden_dim, num_layers, num_heads, head_dim,
                         dtype, max_seqlen, total_hbm_bytes):
    """Run with full HotState controller — zero rebuilds, self-discovering."""

    # Build ONE model with max KV pages
    max_pages = compute_split_params(total_hbm_bytes, 20, 80,
        num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)[2]

    model = build_inference_model(hidden_dim, num_layers, num_heads,
                                  head_dim, dtype, max_seqlen, max_pages)

    # Get embedding module reference (from the inference ranking model)
    # This is set up by get_inference_ranking_gr
    emb_module = model.sparse_module

    # Enable HotState
    model.dense_module.enable_hotstate(
        total_hbm_bytes=total_hbm_bytes,
        embedding_module=emb_module,
        kv_module=model.dense_module.async_kvcache,
    )
    controller = model.dense_module.hotstate

    print(f"=== HotState: Unified HBM Control Plane ===")
    print(f"Total HBM: {total_hbm_bytes / 1024**3:.2f} GiB")
    print(f"KV pages: {max_pages} (max), dynamically managed")
    print(f"Embedding: hot/cold row-group granularity")
    print(f"")

    dataset._iloc = 0
    dataloader = get_data_loader(dataset)
    it = iter(dataloader)

    # Warmup
    print(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        batch, uids, thl = next(it)
        controller.before_batch(batch, uids, thl)
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)
        controller.after_batch(batch, 0.0)

    # Measure
    trace_records = []
    kv_budget_history = []
    print(f"\nMeasuring ({measure_batches} batches)...")
    for i in range(measure_batches):
        batch, uids, thl = next(it)
        control = controller.before_batch(batch, uids, thl)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        controller.after_batch(batch, latency_ms)

        hist_len = thl[0].item() // 2
        user_id = int(uids[0].item())
        kv_budget_history.append(control["kv_page_budget"])

        trace_records.append({
            "split": f"hotstate",
            "split_lhs": 0, "split_rhs": 0,
            "kv_page_budget": control["kv_page_budget"],
            "hbm_bytes_used": control["hbm_bytes_used"],
            "evicted": control["evicted"],
            "admitted": control["admitted"],
            "batch_idx": i,
            "latency_ms": latency_ms,
            "seq_history_len": hist_len,
            "user_id": user_id,
            "epoch": control["epoch"],
        })
        if i % 20 == 0:
            print(f"  [{i+1}/{measure_batches}] latency={latency_ms:.2f}ms "
                  f"hist={hist_len} kv_pages={control['kv_page_budget']}")

    teardown_model(model)

    lats = sorted(r["latency_ms"] for r in trace_records)
    n = len(lats)
    def pct(v, r):
        return v[min(len(v)-1, int(len(v)*r))] if v else float("nan")

    results = {
        "mean": sum(lats)/n, "p50": pct(lats,0.5), "p95": pct(lats,0.95),
        "p99": pct(lats,0.99), "p99_9": pct(lats,0.999),
        "max": max(lats), "num_records": n,
        "kv_budget_range": f"{min(kv_budget_history)}-{max(kv_budget_history)}",
    }
    with open(out_jsonl, "w") as f:
        for r in trace_records:
            f.write(json.dumps(r) + "\n")
    return results


def run_discover_arbiter(dataset, total_available, warmup_batches,
                         initial_measure_batches, num_users, out_jsonl,
                         hidden_dim, num_layers, num_heads, head_dim, dtype,
                         max_seqlen, total_hbm_bytes):
    """Run Phase 3: ε-greedy online discovery + exploitation."""

    q_table = QTable(bucket_size=256)
    current_split = DEFAULT_INITIAL_SPLIT

    _, _, blocks = compute_split_params(
        total_hbm_bytes, current_split[0], current_split[1],
        num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)
    model = build_inference_model(hidden_dim, num_layers, num_heads, head_dim,
                                  dtype, max_seqlen, blocks)

    print(f"=== Phase 3: Online ε-Greedy Self-Discovering Arbiter ===")
    print(f"Candidates: 7 splits {[f'{e}:{k}' for e,k in ALL_CANDIDATE_SPLITS]}")
    print(f"Initial:    {current_split[0]}:{current_split[1]} ({blocks} KV pages)")
    print(f"ε schedule: 0.50 → 0.00 over 300 measured batches")
    print(f"")

    dataloader = get_data_loader(dataset)
    dataloader_iter = iter(dataloader)

    # ── Warmup ──────────────────────────────────────────────────────────────
    print(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        batch, uids, thl = next(dataloader_iter)
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)

    # ── Online measurement + learning ──────────────────────────────────────
    trace_records = []
    switch_count = 0
    i = 0
    measure_batches = initial_measure_batches

    print(f"\nOnline learning (up to {measure_batches} measured batches)...")
    while i < measure_batches:
        batch, uids, thl = next(dataloader_iter)
        hist_len = thl[0].item() // 2
        bucket = q_table.bucket(hist_len)
        eps = epsilon_schedule(i)

        # ── Action selection ───────────────────────────────────────────────
        best = q_table.best_split(bucket)
        if best is None:
            action = random.choice(ALL_CANDIDATE_SPLITS)
        elif q_table.explored_count(bucket) < 2 and random.random() < 0.3:
            action = random.choice(ALL_CANDIDATE_SPLITS)
        elif random.random() < eps:
            action = random.choice(ALL_CANDIDATE_SPLITS)
        else:
            action = best

        # ── Model rebuild ─────────────────────────────────────────────────
        if action != current_split and (measure_batches - i) > num_users:
            teardown_model(model)
            torch.cuda.synchronize()

            _, _, blocks = compute_split_params(
                total_hbm_bytes, action[0], action[1],
                num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)
            model = build_inference_model(hidden_dim, num_layers, num_heads,
                                          head_dim, dtype, max_seqlen, blocks)
            current_split = action
            switch_count += 1

            tag = "explore" if eps > 0 or best is None else "exploit"
            print(f"\n  [Switch #{switch_count}] {tag}: "
                  f"→ {action[0]}:{action[1]} ({blocks} KV pages) "
                  f"ε={eps:.2f}, hist={hist_len}")
            print(f"  Mini-warmup ({num_users} batches)...")

            for w in range(num_users):
                with torch.inference_mode():
                    model.forward_with_kvcache(batch, uids, thl)
                if w < num_users - 1:
                    batch, uids, thl = next(dataloader_iter)
            measure_batches -= num_users
            continue

        # ── Measure ──────────────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        q_table.update(bucket, action, latency_ms)

        user_id = int(uids[0].item())
        record = {
            "split": f"discover_{action[0]}_{action[1]}",
            "split_lhs": action[0], "split_rhs": action[1],
            "emb_budget_bytes": int(total_hbm_bytes * action[0] / (action[0]+action[1])),
            "kv_budget_bytes": 0,  # computed below
            "blocks_in_primary_pool": 0,
            "batch_idx": len(trace_records),
            "latency_ms": latency_ms,
            "seq_history_len": hist_len,
            "user_id": user_id,
            "epsilon": eps,
        }
        # Fill actual budget numbers
        emb_b, kv_b, blk = compute_split_params(
            total_hbm_bytes, action[0], action[1],
            num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)
        record["emb_budget_bytes"] = emb_b
        record["kv_budget_bytes"] = kv_b
        record["blocks_in_primary_pool"] = blk
        trace_records.append(record)
        i += 1

        if i % 20 == 0:
            tag = "e" if eps > 0 and random.random() < eps else "x"
            print(f"  [{i}/{measure_batches}] ε={eps:.2f} "
                  f"split={action[0]}:{action[1]} "
                  f"latency={latency_ms:.2f}ms hist={hist_len}")

    # ── Cleanup ───────────────────────────────────────────────────────────
    teardown_model(model)

    # ── Discovered policy ─────────────────────────────────────────────────
    discovered = q_table.discovered_policy()
    q_table.summary()
    print(f"\n=== Discovered Policy ===")
    print(f"  {discovered.describe()}")

    # ── Statistics ────────────────────────────────────────────────────────
    lats = sorted(r["latency_ms"] for r in trace_records)
    n = len(lats)

    def pct(sorted_vals, ratio):
        if not sorted_vals:
            return float("nan")
        idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * ratio)))
        return sorted_vals[idx]

    results = {
        "mean": sum(lats) / n if n else float("nan"),
        "p50": pct(lats, 0.50), "p95": pct(lats, 0.95),
        "p99": pct(lats, 0.99), "p99_9": pct(lats, 0.999),
        "max": max(lats) if n else float("nan"),
        "switch_count": switch_count, "num_records": n,
        "discovered_policy": discovered.describe(),
    }

    with open(out_jsonl, "w") as f:
        for r in trace_records:
            f.write(json.dumps(r) + "\n")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_thresholds(s):
    if not s:
        return []
    result = []
    for part in s.split(","):
        hist, emb, kv = part.strip().split(":")
        result.append((int(hist), int(emb), int(kv)))
    return result


def parse_split(s):
    parts = s.split(":")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Unified HBM Arbiter — Phase 2 (exploit) or Phase 3 (discover)")
    # Model config
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # Dataset config
    parser.add_argument("--max-history-seqlen", type=int, default=4096)
    parser.add_argument("--max-num-candidates", type=int, default=100)
    parser.add_argument("--max-incremental-seqlen", type=int, default=64)
    parser.add_argument("--num-users", type=int, default=8)
    # Arbiter config
    parser.add_argument("--mode", type=str, default="discover",
                        choices=["discover", "exploit", "hotstate"],
                        help="discover = Phase 3 (ε-greedy), exploit = Phase 2")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="For exploit mode: e.g. '3072:50:50'")
    parser.add_argument("--default-split", type=str, default="70:30",
                        help="For exploit mode: default emb:kv split")
    parser.add_argument("--total-hbm-budget-gib", type=float, default=1.0)
    # Run config
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--out-jsonl", type=str, required=True,
                        help="Output JSONL trace file")
    args = parser.parse_args()

    dtype = (torch.bfloat16
             if args.dtype in ("bfloat16", "float16")
             else torch.float32)
    max_seqlen = args.max_history_seqlen * 2 + args.max_num_candidates
    total_hbm_bytes = int(args.total_hbm_budget_gib * 1024**3)

    dataset, total_available = build_dataset(
        args.max_history_seqlen, args.max_num_candidates,
        args.max_incremental_seqlen, args.num_users,
    )
    warmup_batches = max(1, int(total_available * args.warmup_ratio))
    measure_batches = total_available - warmup_batches

    common = dict(
        dataset=dataset, total_available=total_available,
        warmup_batches=warmup_batches, num_users=args.num_users,
        out_jsonl=args.out_jsonl,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, head_dim=args.head_dim,
        dtype=dtype, max_seqlen=max_seqlen, total_hbm_bytes=total_hbm_bytes,
    )

    if args.mode == "exploit":
        if args.thresholds is None:
            parser.error("--thresholds required for exploit mode")
        policy = ThresholdPolicy(
            thresholds=parse_thresholds(args.thresholds),
            default=parse_split(args.default_split),
        )
        results = run_exploit_arbiter(
            measure_batches=measure_batches, policy=policy, **common)
    elif args.mode == "hotstate":
        results = run_hotstate_arbiter(
            dataset=dataset, total_available=total_available,
            warmup_batches=warmup_batches, measure_batches=measure_batches,
            num_users=args.num_users, out_jsonl=args.out_jsonl,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            num_heads=args.num_heads, head_dim=args.head_dim,
            dtype=dtype, max_seqlen=max_seqlen,
            total_hbm_bytes=total_hbm_bytes)

    else:  # discover
        results = run_discover_arbiter(
            initial_measure_batches=measure_batches, **common)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS ({args.mode} mode)")
    print(f"{'=' * 60}")
    print(f"  Mean:       {results['mean']:6.2f}ms")
    print(f"  P50:        {results['p50']:6.2f}ms")
    print(f"  P95:        {results['p95']:6.2f}ms")
    print(f"  P99:        {results['p99']:6.2f}ms")
    print(f"  P99.9:      {results['p99_9']:6.2f}ms")
    print(f"  Max:        {results['max']:6.2f}ms")
    print(f"  Switches:   {results['switch_count']}")
    print(f"  Records:    {results['num_records']}")
    if "discovered_policy" in results:
        print(f"  Discovered: {results['discovered_policy']}")


if __name__ == "__main__":
    main()
