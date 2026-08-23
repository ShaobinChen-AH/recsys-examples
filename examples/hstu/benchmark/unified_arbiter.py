# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2: Split-Switching Online Controller.

Implements a threshold-based adaptive arbiter that dynamically switches
between embedding:KV budget splits during inference, rebuilding the model
when a pre-computed threshold is crossed.

Example:
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6064 \
      --module benchmark.unified_arbiter \
      --max-history-seqlen 4096 --max-num-candidates 100 --max-incremental-seqlen 64 \
      --num-users 8 --total-hbm-budget-gib 1.0 --warmup-ratio 0.1 \
      --thresholds "3072:50:50" --default-split "70:30" \
      --out-jsonl ./logs/arbiter_trace.jsonl
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

# Path setup (same as unified_plane_profiling.py)
_HSTU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HSTU_DIR, ".."))

sys.path.insert(0, os.path.join(_REPO_ROOT, "commons"))   # examples/commons/
sys.path.insert(0, _HSTU_DIR)                              # examples/hstu/
sys.path.insert(0, os.path.join(_HSTU_DIR, "model"))       # examples/hstu/model/

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
import math
from modules.hotstate.admission_adapter import HotStateAdmissionStrategy

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_KV_PAGE_SIZE = 32
DEFAULT_OFFLOAD_CHUNKSIZE = 8192

# ── Utility functions (replicated from unified_plane_profiling.py) ────────────

def count_dataset_batches(max_history_length, max_incremental_seqlen, num_users):
    num_steps = len(range(max_incremental_seqlen, max_history_length,
                          max_incremental_seqlen))
    return num_steps * num_users


def build_inference_model(hidden_dim, num_layers, num_heads, head_dim, dtype,
                          max_seqlen, blocks_in_primary_pool, max_batch_size=1, hotstate_admit_strategy=None, hotstate_admission_counters=None):
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen * 2,       # ×2 for interleaved item+action features
        dtype=dtype,
    )
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=DEFAULT_KV_PAGE_SIZE,
        offload_chunksize=DEFAULT_OFFLOAD_CHUNKSIZE,
    )
    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=["item_feat"], table_name="item",
            vocab_size=10_000_000, dim=hidden_dim, use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=["act_feat"], table_name="act",
            vocab_size=128, dim=hidden_dim, use_dynamicemb=False,
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
        hotstate_admit_strategy=hotstate_admit_strategy,
        hotstate_admission_counters=hotstate_admission_counters,
    )
    if dtype == torch.bfloat16:
        model.bfloat16()
    model.eval()
    return model


def build_dataset(max_history_length, max_num_candidates, max_incremental_seqlen,
                  num_users):
    max_seqlen = max_history_length * 2 + max_num_candidates
    feature_configs = [
        FeatureConfig(
            feature_names=["item_feat", "act_feat"],
            max_item_ids=[10_000_000 - 1, 128 - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        ),
    ]
    total_batches = count_dataset_batches(
        max_history_length, max_incremental_seqlen, num_users)
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


def compute_split_params(total_hbm_bytes, emb_pct, kv_pct, num_layers,
                         num_heads, head_dim, page_size, dtype):
    """Return (emb_budget_bytes, kv_budget_bytes, blocks_in_primary_pool)."""
    bytes_per_elem = 2  # bfloat16
    kv_page_bytes = (num_layers * 2 * page_size * num_heads * head_dim
                     * bytes_per_elem)
    emb_budget = math.floor(emb_pct / (emb_pct + kv_pct) * total_hbm_bytes)
    kv_budget = total_hbm_bytes - emb_budget
    blocks = max(1, kv_budget // kv_page_bytes)
    return emb_budget, kv_budget, blocks


def teardown_model(model, wait_for_workers=True):
    try:
        kvcache = model.dense_module.async_kvcache
        if wait_for_workers:
            kvcache.executor.shutdown(wait=True)
            kvcache.onload_worker.shutdown(wait=True)
        else:
            kvcache.executor.shutdown(wait=False, cancel_futures=True)
            kvcache.onload_worker.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    if wait_for_workers:
        torch.cuda.synchronize()
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ── Threshold Policy ─────────────────────────────────────────────────────────

class ThresholdPolicy:
    """
    Maps per-feature history length to (emb_pct, kv_pct) split.

    Parameters
    ----------
    thresholds : list of (history_boundary, emb_pct, kv_pct)
        Sorted by boundary ascending.  First matching boundary wins.
    default : (emb_pct, kv_pct)
        Split used when no threshold boundary is crossed.

    Example
    -------
    >>> p = ThresholdPolicy([(3072, 50, 50)], (70, 30))
    >>> p.get_split(2048)   # → (70, 30)
    >>> p.get_split(3328)   # → (50, 50)
    """

    def __init__(self, thresholds: List[Tuple[int, int, int]],
                 default: Tuple[int, int]):
        self.thresholds = sorted(thresholds, key=lambda t: t[0])
        self.default = default
        for hist, emb, kv in self.thresholds:
            if emb + kv != 100:
                raise ValueError(
                    f"Threshold hist={hist} split {emb}:{kv} != 100%")

    def get_split(self, hist_len: int) -> Tuple[int, int]:
        for boundary, emb_pct, kv_pct in self.thresholds:
            if hist_len >= boundary:
                return (emb_pct, kv_pct)
        return self.default

    def describe(self) -> str:
        if not self.thresholds:
            return f"default {self.default[0]}:{self.default[1]}"
        parts = [
            f"hist < {self.thresholds[0][0]} → "
            f"{self.default[0]}:{self.default[1]}"]
        for b, e, k in self.thresholds:
            parts.append(f"hist ≥ {b} → {e}:{k}")
        return "; ".join(parts)


# ── Arbiter core ─────────────────────────────────────────────────────────────

def run_arbiter(dataset, total_available, warmup_batches, measure_batches,
                num_users, policy, model_config, out_jsonl,
                hidden_dim, num_layers, num_heads, head_dim, dtype, max_seqlen,
                total_hbm_bytes):
    """Run the adaptive controller over the full dataset.

    Returns a dict with keys: mean, p50, p95, p99, p99_9, max, switch_count,
    num_records.
    """

    # ── Build initial model with default split ────────────────────────────
    current_split = policy.default
    emb_budget, kv_budget, blocks = compute_split_params(
        total_hbm_bytes, current_split[0], current_split[1],
        num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype,
    )
    model = build_inference_model(
        hidden_dim, num_layers, num_heads, head_dim, dtype,
        max_seqlen, blocks,
    )

    print(f"=== Phase 2: Split-Switching Controller ===")
    print(f"Policy: {policy.describe()}")
    print(f"Model: {num_layers}L, {num_heads}H, hdim={head_dim}, "
          f"{'bf16' if dtype == torch.bfloat16 else dtype}")
    print(f"Initial split: {current_split[0]}:{current_split[1]} "
          f"({blocks} KV pages)")
    print(f"")

    dataloader = get_data_loader(dataset)
    dataloader_iter = iter(dataloader)

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"Warming up ({warmup_batches} batches)...")
    for _ in range(warmup_batches):
        batch, uids, thl = next(dataloader_iter)
        with torch.inference_mode():
            model.forward_with_kvcache(batch, uids, thl)

    # ── Measurement ───────────────────────────────────────────────────────
    trace_records = []
    switch_count = 0
    i = 0  # measured-batch counter

    print(f"\nMeasuring (up to {measure_batches} batches)...")
    while i < measure_batches:
        batch, uids, thl = next(dataloader_iter)
        hist_len = thl[0].item() // 2            # per-feature history tokens

        target_split = policy.get_split(hist_len)

        # ── Model rebuild on threshold crossing ──────────────────────────
        if (target_split != current_split
                and (measure_batches - i) >= num_users):
            teardown_model(model)
            torch.cuda.synchronize()

            emb_budget, kv_budget, blocks = compute_split_params(
                total_hbm_bytes, target_split[0], target_split[1],
                num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype,
            )
            model = build_inference_model(
                hidden_dim, num_layers, num_heads, head_dim, dtype,
                max_seqlen, blocks,
            )
            current_split = target_split
            switch_count += 1

            print(f"\n  [Switch #{switch_count}] hist={hist_len}: "
                  f"→ {target_split[0]}:{target_split[1]} ({blocks} KV pages)")
            print(f"  Mini-warmup ({num_users} batches)...")

            # Drain num_users batches through new model (NOT measured)
            for w in range(num_users):
                with torch.inference_mode():
                    model.forward_with_kvcache(batch, uids, thl)
                if w < num_users - 1:
                    batch, uids, thl = next(dataloader_iter)
            measure_batches -= num_users
            continue       # back to while loop — these batches consumed

        # Normal measurement
        origin_cached_length = None
        max_origin_cached_length = None
        new_tokens = None
        offload_pages = None

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward_with_kvcache(batch, uids, thl)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            async_kvcache = model.dense_module.async_kvcache
            origin_cached_lengths = async_kvcache.last_origin_cached_lengths
            origin_cached_length = (
                int(origin_cached_lengths[0])
                if origin_cached_lengths is not None and len(origin_cached_lengths) > 0
                else None
            )
            max_origin_cached_length = (
                max(int(x) for x in origin_cached_lengths)
                if origin_cached_lengths is not None
                else None
            )
            new_tokens = async_kvcache.last_new_tokens
            offload_pages = async_kvcache.last_num_offload_pages
        except Exception as e:
            if i == 0:
                print(f"[HotState metric debug] failed: {repr(e)}")

        user_id = int(uids[0].item())

        record = {
            "split": f"adaptive_{current_split[0]}_{current_split[1]}",
            "split_lhs": current_split[0],
            "split_rhs": current_split[1],
            "emb_budget_bytes": emb_budget,
            "kv_budget_bytes": kv_budget,
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

    # ── Cleanup ───────────────────────────────────────────────────────────
    teardown_model(model)

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
        "p50": pct(lats, 0.50),
        "p95": pct(lats, 0.95),
        "p99": pct(lats, 0.99),
        "p99_9": pct(lats, 0.999),
        "max": max(lats) if n else float("nan"),
        "switch_count": switch_count,
        "num_records": n,
    }

    # ── Write JSONL ───────────────────────────────────────────────────────
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

def run_hotstate_arbiter(dataset, total_available, warmup_batches,
                         measure_batches, num_users, out_jsonl,
                         hidden_dim, num_layers, num_heads, head_dim,
                         dtype, max_seqlen, total_hbm_bytes, hotstate_trace_detail="scalar",
                         skip_hotstate_kv_handles_for_admission_smoke=False,
                         hotstate_admission_smoke_max_keys=None,
                         hotstate_admission_admit_all_control=False):
    """Run with full HotState controller — zero rebuilds, self-discovering."""

    # Build ONE model with max KV pages
    max_pages = compute_split_params(total_hbm_bytes, 20, 80,
        num_layers, num_heads, head_dim, DEFAULT_KV_PAGE_SIZE, dtype)[2]
    
    hotstate_admit_strategy = HotStateAdmissionStrategy(admit_all_when_empty=True)

    model = build_inference_model(hidden_dim, num_layers, num_heads,
                                  head_dim, dtype, max_seqlen, max_pages, hotstate_admit_strategy=hotstate_admit_strategy,)

    emb_module = model.sparse_module

    model.dense_module.enable_hotstate(
        total_hbm_bytes=total_hbm_bytes,
        skip_kv_handles_for_admission_smoke=skip_hotstate_kv_handles_for_admission_smoke,
        admission_smoke_max_admitted_keys=hotstate_admission_smoke_max_keys,
    )
    model.dense_module.set_hotstate_embedding_module(emb_module)
    controller = model.dense_module.hotstate

    controller.admission_admit_all_control = hotstate_admission_admit_all_control

    controller.set_trace_detail(hotstate_trace_detail)

    print(f"Trace detail: {hotstate_trace_detail}")

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

    # Measure
    trace_records = []
    kv_budget_history = []
    print(f"\nMeasuring ({measure_batches} batches)...")
    for i in range(measure_batches):
        batch, uids, thl = next(it)
        
        control = controller.before_batch(batch, uids, thl)

        origin_cached_length = None
        max_origin_cached_length = None
        new_tokens = None
        offload_pages = None
        max_seqlen = None

        accepts_before = int(hotstate_admit_strategy.num_accepted)
        rejects_before = int(hotstate_admit_strategy.num_rejected)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            logits = model.forward_with_kvcache(batch, uids, thl)

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        accepts_after = int(hotstate_admit_strategy.num_accepted)
        rejects_after = int(hotstate_admit_strategy.num_rejected)
        accept_delta = accepts_after - accepts_before
        reject_delta = rejects_after - rejects_before

        try:
            async_kvcache = model.dense_module.async_kvcache
            origin_cached_lengths = async_kvcache.last_origin_cached_lengths
            origin_cached_length = (
                int(origin_cached_lengths[0])
                if origin_cached_lengths is not None and len(origin_cached_lengths) > 0
                else None
            )
            max_origin_cached_length = (
                max(int(x) for x in origin_cached_lengths)
                if origin_cached_lengths is not None
                else None
            )
            new_tokens = async_kvcache.last_new_tokens
            offload_pages = async_kvcache.last_num_offload_pages
            max_seqlen = async_kvcache.last_max_seqlen
        except Exception as e:
            if i == 0:
                print(f"[HotState metric debug] failed: {repr(e)}")

        post_control = controller.after_batch(batch, latency_ms)

        hist_len = thl[0].item() // 2
        user_id = int(uids[0].item())
        kv_budget_history.append(control["kv_page_budget"])

        if skip_hotstate_kv_handles_for_admission_smoke:
            active_kv_page_limit = control["kv_page_budget"]
            resident_kv_pages = None
            empty_kv_pages = None
            withheld_kv_pages = None
            logical_kv_budget_bytes = None
            physical_kv_cache_bytes = None
            actual_resident_kv_bytes = None
        else:
            active_kv_page_limit = controller.kv_adapter.get_current_page_limit()
            resident_kv_pages = controller.kv_adapter.get_resident_page_count()
            empty_kv_pages = controller.kv_adapter.get_empty_page_count()
            withheld_kv_pages = controller.kv_adapter.get_withheld_page_count()
            logical_kv_budget_bytes = controller.kv_adapter.logical_kv_budget_bytes()
            physical_kv_cache_bytes = controller.kv_adapter.physical_kv_cache_bytes()
            actual_resident_kv_bytes = controller.kv_adapter.actual_resident_kv_bytes()

        trace_records.append({
            "split": "hotstate",
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
            "active_kv_page_limit": active_kv_page_limit,
            "resident_kv_pages": resident_kv_pages,
            "empty_kv_pages": empty_kv_pages,
            "withheld_kv_pages": withheld_kv_pages,
            "logical_kv_budget_bytes": logical_kv_budget_bytes,
            "physical_kv_cache_bytes": physical_kv_cache_bytes,
            "actual_resident_kv_bytes": actual_resident_kv_bytes,
            "origin_cached_length": origin_cached_length,
            "max_origin_cached_length": max_origin_cached_length,
            "new_tokens": new_tokens,
            "offload_pages": offload_pages,
            "max_seqlen": max_seqlen,
            "completed_transfers": control.get("completed_transfers", 0),
            "hotstate_profile_ms": control.get("profile_ms", {}),
            "num_state_handles": len(control.get("state_trace", [])),
            "state_trace": control.get("state_trace", []),
            "post_num_state_handles": post_control.get("post_num_state_handles", 0),
            "post_state_trace": post_control.get("post_state_trace", []),
            "selected_hbm_bytes": control.get("selected_hbm_bytes", 0),
            "selected_kv_bytes": control.get("selected_kv_bytes", 0),
            "selected_embedding_bytes": control.get("selected_embedding_bytes", 0),
            "embedding_admission_policy_size": hotstate_admit_strategy.last_policy_size,
            "embedding_admission_accepts": hotstate_admit_strategy.num_accepted,
            "embedding_admission_rejects": hotstate_admit_strategy.num_rejected,
            "embedding_admission_calls": hotstate_admit_strategy.num_admit_calls,
            "hotstate_admission_smoke_max_keys": hotstate_admission_smoke_max_keys,
            "embedding_admission_budget_bytes": control.get("embedding_admission_budget_bytes", 0),
            "embedding_admission_budget_keys": control.get("embedding_admission_budget_keys", 0),
            "embedding_admission_max_keys": control.get("embedding_admission_max_keys", None),
            "embedding_selected_budget_bytes": control.get("embedding_selected_budget_bytes", 0),
            "embedding_kv_reserved_bytes": control.get("embedding_kv_reserved_bytes", 0),
            "embedding_residual_budget_bytes": control.get("embedding_residual_budget_bytes", 0),
            "embedding_requested_unique_keys": control.get("embedding_requested_unique_keys", 0),
            "embedding_admission_cap_source": control.get("embedding_admission_cap_source", ""),
            "embedding_admission_accept_delta": accept_delta,
            "embedding_admission_reject_delta": reject_delta,
        })
        if i < 5 or latency_ms > 10:
            print(
                f"[KVDBG {i+1}] latency={latency_ms:.2f}ms hist={hist_len} "
                f"origin={origin_cached_length} max_origin={max_origin_cached_length} "
                f"new_tokens={new_tokens} offload_pages={offload_pages} "
                f"max_seqlen={max_seqlen}"
        )
        if i % 20 == 0:
            print(f"  [{i+1}/{measure_batches}] latency={latency_ms:.2f}ms "
                  f"hist={hist_len} kv_pages={control['kv_page_budget']}")

    teardown_model(model, wait_for_workers=False)

    lats = sorted(r["latency_ms"] for r in trace_records)
    n = len(lats)
    def pct(v, r):
        return v[min(len(v)-1, int(len(v)*r))] if v else float("nan")

    results = {
        "mean": sum(lats)/n, "p50": pct(lats,0.5), "p95": pct(lats,0.95),
        "p99": pct(lats,0.99), "p99_9": pct(lats,0.999),
        "max": max(lats), "num_records": n,
        "switch_count": 0,
        "kv_budget_range": f"{min(kv_budget_history)}-{max(kv_budget_history)}",
    }
    with open(out_jsonl, "w") as f:
        for r in trace_records:
            f.write(json.dumps(r) + "\n")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Unified HBM Arbiter — exploit / calibrate / hotstate")
    parser.add_argument("--mode", type=str, default="exploit",
                        choices=["exploit", "calibrate", "hotstate"],
                        help="exploit=Phase 2, calibrate=Phase 3, hotstate=full control plane")
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
    parser.add_argument("--thresholds", type=str, default=None,
                        help="For exploit mode: e.g. '3072:50:50'")
    parser.add_argument("--default-split", type=str, default=None,
                        help="For exploit mode: default emb:kv split, e.g. '70:30'")
    parser.add_argument("--total-hbm-budget-gib", type=float, default=1.0)
    # Run config
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--out-jsonl", type=str, required=True,
                        help="Output JSONL trace file")

    parser.add_argument("--hotstate-trace-detail", type=str, default="scalar",
                    choices=["scalar", "full"],
                    help="scalar=performance-safe, full=offline state trace collection")
    parser.add_argument(
        "--skip-hotstate-kv-handles-for-admission-smoke",
        action="store_true",
        help="Skip KV HotState handle export/page-budget mutation; only for DynamicEmb admission smoke tests.",
    )
    parser.add_argument(
        "--hotstate-admission-smoke-max-keys",
        type=int,
        default=None,
        help="Smoke-only cap on admitted DynamicEmb keys per batch.",
    )
    parser.add_argument("--hotstate-admission-admit-all-control", action="store_true")
    args = parser.parse_args()

    # ── Build config dicts ──────────────────────────────────────────────
    dtype = (torch.bfloat16
             if args.dtype in ("bfloat16", "float16")
             else torch.float32)
    max_seqlen = args.max_history_seqlen * 2 + args.max_num_candidates
    total_hbm_bytes = int(args.total_hbm_budget_gib * 1024**3)

    # ── Create dataset ───────────────────────────────────────────────────
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
        dtype=dtype, max_seqlen=max_seqlen, total_hbm_bytes=total_hbm_bytes)

    # ── Dispatch by mode ────────────────────────────────────────────────
    if args.mode == "exploit":
        if args.thresholds is None:
            parser.error("--thresholds required for exploit mode")
        if args.default_split is None:
            parser.error("--default-split required for exploit mode")
        policy = ThresholdPolicy(
            thresholds=parse_thresholds(args.thresholds),
            default=parse_split(args.default_split))
        results = run_exploit_arbiter(
            measure_batches=measure_batches, policy=policy, **common)
    elif args.mode == "calibrate":
        results = run_calibrate_arbiter(
            measure_batches=measure_batches, **common)
    elif args.mode == "hotstate":
        results = run_hotstate_arbiter(
            measure_batches=measure_batches,
            hotstate_trace_detail=args.hotstate_trace_detail,
            skip_hotstate_kv_handles_for_admission_smoke=args.skip_hotstate_kv_handles_for_admission_smoke,
            hotstate_admission_smoke_max_keys=args.hotstate_admission_smoke_max_keys,
            hotstate_admission_admit_all_control=args.hotstate_admission_admit_all_control,
            **common)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS ({args.mode} mode)")
    print(f"{'=' * 60}")
    print(f"  Mean:          {results['mean']:6.2f}ms")
    print(f"  P50:           {results['p50']:6.2f}ms")
    print(f"  P95:           {results['p95']:6.2f}ms")
    print(f"  P99:           {results['p99']:6.2f}ms")
    print(f"  P99.9:         {results['p99_9']:6.2f}ms")
    print(f"  Max:           {results['max']:6.2f}ms")
    print(f"  Switches:      {results['switch_count']}")
    print(f"  Records:       {results['num_records']}")
    if "discovered_policy" in results:
        print(f"  Discovered:    {results['discovered_policy']}")
    if "calibration_cost_batches" in results:
        print(f"  Calib cost:    {results['calibration_cost_batches']} batches")

if __name__ == "__main__":
    main()
