# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified Plane Profiling — Trace Collection (Row 2)

Runs multiple static embedding/KV budget splits and records per-batch
latency into a JSONL trace file. Offline analysis will compute:
  - Oracle adaptive: per-batch pick best split → total latency
  - Best static: single best split for all batches → total latency
  - If oracle < best static → Row 2 proven
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# 定位到 examples/hstu/ 目录
_HSTU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_HSTU_DIR, ".."))

sys.path.insert(0, os.path.join(_REPO_ROOT, "commons"))  # commons/
sys.path.insert(0, _HSTU_DIR)                             # examples/hstu/ → 找到 modules/
sys.path.insert(0, os.path.join(_HSTU_DIR, "model"))      # examples/hstu/model/ → 找到 inference_ranking_gr

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


DEFAULT_KV_PAGE_SIZE = 32
DEFAULT_OFFLOAD_CHUNKSIZE = 8192


def build_inference_model(
    hidden_dim, num_layers, num_heads, head_dim, dtype,
    max_seqlen, blocks_in_primary_pool, max_batch_size=1,
):
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen*2,
        dtype=dtype,
    )
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
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


def count_dataset_batches(max_history_length, max_incremental_seqlen, num_users):
    num_seqlen_steps = len(range(max_incremental_seqlen, max_history_length, max_incremental_seqlen))
    return num_seqlen_steps * num_users


def build_dataset(
    max_history_length, max_num_candidates, max_incremental_seqlen,
    num_users,
):
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
        max_history_length, max_incremental_seqlen, num_users
    )
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


def run_static_sweep(
    hidden_dim, num_layers, num_heads, head_dim, dtype_str,
    max_history_length, max_num_candidates, max_incremental_seqlen,
    num_users,
    splits, total_hbm_budget_bytes,
    warmup_ratio,
    out_jsonl,
):
    dtype = torch.bfloat16 if dtype_str in ("bfloat16", "float16") else torch.float32
    max_seqlen = max_history_length * 2 + max_num_candidates

    dataset, total_available = build_dataset(
        max_history_length, max_num_candidates, max_incremental_seqlen, num_users,
    )
    warmup_batches = max(1, int(total_available * warmup_ratio))
    measure_batches = total_available - warmup_batches

    # KV cost per page (all layers)
    bytes_per_elem = 2 if dtype_str in ("bfloat16", "float16") else 4
    kv_page_bytes = num_layers * 2 * DEFAULT_KV_PAGE_SIZE * num_heads * head_dim * bytes_per_elem
    kv_page_mib = kv_page_bytes / (1024 * 1024)

    print(f"Dataset: {total_available} batches (warmup={warmup_batches}, measure={measure_batches})")
    print(f"  seqlen: {max_incremental_seqlen} → {max_history_length - max_incremental_seqlen}")
    print(f"  users={num_users}, candidates={max_num_candidates}")
    print(f"  KV page cost: {kv_page_mib:.3f} MiB/page ({kv_page_bytes} bytes)")
    print(f"  Total HBM budget: {total_hbm_budget_bytes / 1024**3:.2f} GiB")

    for lhs, rhs in splits:
        kv_budget = total_hbm_budget_bytes - math.floor(lhs / (lhs + rhs) * total_hbm_budget_bytes)
        blocks = max(1, kv_budget // kv_page_bytes)
        max_kv_tokens = blocks * DEFAULT_KV_PAGE_SIZE
        print(f"  {lhs}:{rhs} → {blocks} pages → {max_kv_tokens} KV tokens max")

    results = []

    for lhs, rhs in splits:
        split_name = f"static_{lhs}_{rhs}"
        emb_budget = math.floor(lhs / (lhs + rhs) * total_hbm_budget_bytes)
        kv_budget = total_hbm_budget_bytes - emb_budget
        blocks = max(1, kv_budget // kv_page_bytes)

        print(f"{'='*60}")
        print(f"Running: {split_name} (emb={emb_budget/1024**3:.2f}GiB, kv={kv_budget/1024**3:.2f}GiB, blocks={blocks})")
        print(f"{'='*60}")

        model = build_inference_model(
            hidden_dim, num_layers, num_heads, head_dim, dtype,
            max_seqlen, blocks,
        )

        dataloader = get_data_loader(dataset)
        dataloader_iter = iter(dataloader)
        trace_records = []

        try:
            dataloader = get_data_loader(dataset)
            dataloader_iter = iter(dataloader)
            for i in range(warmup_batches):
                batch, user_ids, total_history_lengths = next(dataloader_iter)
                with torch.inference_mode():
                    model.forward_with_kvcache(batch, user_ids, total_history_lengths)

            for i in range(measure_batches):
                batch, user_ids, total_history_lengths = next(dataloader_iter)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model.forward_with_kvcache(batch, user_ids, total_history_lengths)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0

                thl = total_history_lengths.tolist() if torch.is_tensor(total_history_lengths) else list(total_history_lengths)
                hist_len = thl[0] // 2

                record = {
                    "split": split_name,
                    "split_lhs": lhs,
                    "split_rhs": rhs,
                    "emb_budget_bytes": emb_budget,
                    "kv_budget_bytes": kv_budget,
                    "blocks_in_primary_pool": blocks,
                    "batch_idx": i,
                    "latency_ms": latency_ms,
                    "seq_history_len": hist_len,
                    "user_id": int(user_ids[0].item()) if torch.is_tensor(user_ids) else int(user_ids[0]),
                }
                trace_records.append(record)

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{measure_batches}] latency={latency_ms:.2f}ms hist={hist_len}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            trace_records.append({"split": split_name, "status": "error", "error_message": str(exc)})

        # Cleanup
        try:
            kvcache = model.dense_module.async_kvcache
            kvcache.executor.shutdown(wait=True, cancel_futures=True)
            kvcache.onload_worker.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        del kvcache
        del model
        gc.collect()
        torch.cuda.empty_cache()

        for r in trace_records:
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(r) + "")

        valid = [r for r in trace_records if "latency_ms" in r]
        mean_lat = sum(r["latency_ms"] for r in valid) / max(1, len(valid)) if valid else float("nan")
        results.append({"split": split_name, "num_records": len(valid), "mean_latency_ms": mean_lat})

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Plane Profiling — Row 2 Trace Collection")
    # Model config (from gin: kuairand_1k_inference_ranking_1024.gin)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    # Dataset config (from gin + aligned with official RandomInferenceDataset)
    parser.add_argument("--max-history-seqlen", type=int, default=4096)
    parser.add_argument("--max-num-candidates", type=int, default=100)
    parser.add_argument("--max-incremental-seqlen", type=int, default=64)
    parser.add_argument("--num-users", type=int, default=8)
    # Sweep config
    parser.add_argument("--splits", type=str, default="20:80,30:70,40:60,50:50,60:40,70:30,80:20")
    parser.add_argument("--total-hbm-budget-gib", type=float, default=1.0)
    # Run config
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--out-jsonl", type=str, required=True)
    args = parser.parse_args()

    splits = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in args.splits.split(",")]
    total_hbm_bytes = int(args.total_hbm_budget_gib * 1024**3)

    total_batches = count_dataset_batches(args.max_history_seqlen, args.max_incremental_seqlen, args.num_users)
    print(f"Expected batches: {total_batches}")

    results = run_static_sweep(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        num_heads=args.num_heads, head_dim=args.head_dim,
        dtype_str=args.dtype,
        max_history_length=args.max_history_seqlen,
        max_num_candidates=args.max_num_candidates,
        max_incremental_seqlen=args.max_incremental_seqlen,
        num_users=args.num_users,
        splits=splits, total_hbm_budget_bytes=total_hbm_bytes,
        warmup_ratio=args.warmup_ratio, out_jsonl=args.out_jsonl,
    )

    print("" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['split']}: mean={r['mean_latency_ms']:.2f}ms, n={r['num_records']}")


if __name__ == "__main__":
    main()

