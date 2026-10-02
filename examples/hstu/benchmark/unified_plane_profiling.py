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
import hashlib
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
from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter
from modules.hotstate.physical_budget import plan_physical_budget
from benchmark.evidence import validate_static_baseline_trace


DEFAULT_KV_PAGE_SIZE = 32
DEFAULT_OFFLOAD_CHUNKSIZE = 8192


def build_inference_model(
    hidden_dim, num_layers, num_heads, head_dim, dtype,
    max_seqlen, blocks_in_primary_pool, max_batch_size=1,
    dynamic_embedding_hbm_bytes=0,
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
        dynamic_embedding_hbm_bytes=dynamic_embedding_hbm_bytes,
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

def make_batch_fingerprint(batch, user_ids, total_history_lengths):
    digest = hashlib.sha256()
    fields = [
        ("item_feat", batch.features["item_feat"].values()),
        ("act_feat", batch.features["act_feat"].values()),
        ("user_ids", user_ids),
        ("history_lengths", total_history_lengths),
    ]

    for name, value in fields:
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()


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
    
    torch.manual_seed(42)
    dataset, total_available = build_dataset(
        max_history_length, max_num_candidates, max_incremental_seqlen, num_users,
    )
    output_path = Path(out_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A trace is one complete sweep.  Truncating at the start prevents stale
    # rows from a prior run from being silently included in a paired comparison.
    output_path.write_text("", encoding="utf-8")
    warmup_batches = max(1, int(total_available * warmup_ratio))
    measure_batches = total_available - warmup_batches

    print(f"Dataset: {total_available} batches (warmup={warmup_batches}, measure={measure_batches})")
    print(f"  seqlen: {max_incremental_seqlen} → {max_history_length - max_incremental_seqlen}")
    print(f"  users={num_users}, candidates={max_num_candidates}")
    print(f"  Total HBM budget: {total_hbm_budget_bytes / 1024**3:.2f} GiB")

    max_required_kv_tokens = 2 * (
        max_history_length - max_incremental_seqlen
    )
    for lhs, rhs in splits:
        plan = plan_physical_budget(
            state_budget_bytes=total_hbm_budget_bytes,
            embedding_hbm_ratio=lhs / (lhs + rhs),
            num_layers=num_layers,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            num_tokens_per_page=DEFAULT_KV_PAGE_SIZE,
            max_batch_size=1,
            max_sequence_length=max_seqlen * 2,
            num_tokens_per_chunk=DEFAULT_OFFLOAD_CHUNKSIZE,
            element_bytes=2 if dtype_str in ("bfloat16", "float16") else 4,
        )
        kv_budget = plan.kv_primary_page_bytes
        blocks = plan.blocks_in_primary_pool
        max_kv_tokens = blocks * DEFAULT_KV_PAGE_SIZE
        if (
            max_kv_tokens < max_required_kv_tokens
            and max_kv_tokens < DEFAULT_OFFLOAD_CHUNKSIZE
        ):
            raise ValueError(
                f"Static split {lhs}:{rhs} is infeasible: "
                f"KV capacity={max_kv_tokens} tokens, "
                f"max workload={max_required_kv_tokens} tokens, "
                f"offload chunk={DEFAULT_OFFLOAD_CHUNKSIZE} tokens. "
                "KV capacity is exhausted before offloading can begin."
            )
        print(f"  {lhs}:{rhs} → {blocks} pages → {max_kv_tokens} KV tokens max")

    results = []
    all_trace_records = []

    for lhs, rhs in splits:
        dataset._iloc = 0
        split_name = f"static_{lhs}_{rhs}"
        plan = plan_physical_budget(
            state_budget_bytes=total_hbm_budget_bytes,
            embedding_hbm_ratio=lhs / (lhs + rhs),
            num_layers=num_layers,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            num_tokens_per_page=DEFAULT_KV_PAGE_SIZE,
            max_batch_size=1,
            max_sequence_length=max_seqlen * 2,
            num_tokens_per_chunk=DEFAULT_OFFLOAD_CHUNKSIZE,
            element_bytes=2 if dtype_str in ("bfloat16", "float16") else 4,
        )
        emb_budget = plan.embedding_hbm_bytes
        kv_budget = plan.kv_primary_page_bytes
        blocks = plan.blocks_in_primary_pool

        print(f"{'='*60}")
        print(f"Running: {split_name} (emb={emb_budget/1024**3:.2f}GiB, kv={kv_budget/1024**3:.2f}GiB, blocks={blocks})")
        print(f"{'='*60}")

        model = build_inference_model(
            hidden_dim, num_layers, num_heads, head_dim, dtype,
            max_seqlen, blocks, dynamic_embedding_hbm_bytes=emb_budget,
        )

        embedding_adapter = EmbeddingAdapter(model.sparse_module)
        embedding_adapter.calibrate()
        kv_adapter = KVAdapter(model.dense_module.async_kvcache)
        physical_embedding_bytes = embedding_adapter.physical_hbm_bytes()
        physical_kv_bytes = kv_adapter.physical_hbm_bytes()
        physical_state_bytes = physical_embedding_bytes + physical_kv_bytes
        if physical_state_bytes > total_hbm_budget_bytes:
            raise RuntimeError(
                f"{split_name} physical state allocation exceeds budget: "
                f"embedding={physical_embedding_bytes} kv={physical_kv_bytes} "
                f"budget={total_hbm_budget_bytes}"
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
                torch.cuda.synchronize()

            for i in range(measure_batches):
                batch, user_ids, total_history_lengths = next(dataloader_iter)

                batch_fingerprint = make_batch_fingerprint(
                    batch, user_ids, total_history_lengths
                )

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    model.forward_with_kvcache(batch, user_ids, total_history_lengths)
                torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0

                # Re-measure after every request.  DynamicEmb and the KV
                # manager are expected to keep fixed capacities, but the trace
                # must prove that assumption rather than copy the construction
                # snapshot into every row.
                measured_embedding_bytes = embedding_adapter.physical_hbm_bytes()
                measured_kv_bytes = kv_adapter.physical_hbm_bytes()
                measured_state_bytes = (
                    measured_embedding_bytes + measured_kv_bytes
                )
                if measured_state_bytes > total_hbm_budget_bytes:
                    raise RuntimeError(
                        f"{split_name} physical state allocation exceeded budget "
                        f"during batch {i}: state={measured_state_bytes} "
                        f"budget={total_hbm_budget_bytes}"
                    )

                origin_cached_length = None
                max_origin_cached_length = None
                new_tokens = None
                offload_pages = None
                observed_max_seqlen = None

                try:
                    async_kvcache = model.dense_module.async_kvcache
                    origin_cached_lengths = getattr(
                        async_kvcache,
                        "last_origin_cached_lengths",
                        None,
                    )

                    if (
                        origin_cached_lengths is not None
                        and len(origin_cached_lengths) > 0
                    ):
                        origin_cached_length = int(origin_cached_lengths[0])
                        max_origin_cached_length = max(
                            int(value) for value in origin_cached_lengths
                        )

                    new_tokens = getattr(
                        async_kvcache, "last_new_tokens", None
                    )
                    offload_pages = getattr(
                        async_kvcache, "last_num_offload_pages", None
                    )
                    observed_max_seqlen = getattr(
                        async_kvcache, "last_max_seqlen", None
                    )
                except Exception as exc:
                    if i == 0:
                        print(
                            f"[Static metric debug] failed: {repr(exc)}"
                        )

                thl = total_history_lengths.tolist() if torch.is_tensor(total_history_lengths) else list(total_history_lengths)
                hist_len = thl[0] // 2

                record = {
                    "record_type": "static",
                    "split": split_name,
                    "split_lhs": lhs,
                    "split_rhs": rhs,
                    "emb_budget_bytes": emb_budget,
                    "kv_budget_bytes": kv_budget,
                    "state_budget_bytes": total_hbm_budget_bytes,
                    "configured_state_budget_bytes": total_hbm_budget_bytes,
                    "embedding_hbm_plan_bytes": plan.embedding_hbm_bytes,
                    "kv_primary_pool_plan_bytes": plan.kv_primary_page_bytes,
                    "kv_onload_page_plan_bytes": plan.kv_onload_page_bytes,
                    "kv_copy_buffer_plan_bytes": plan.kv_copy_buffer_bytes,
                    "safety_margin_plan_bytes": plan.safety_margin_bytes,
                    "embedding_physical_hbm_bytes": measured_embedding_bytes,
                    "kv_physical_hbm_bytes": measured_kv_bytes,
                    "managed_state_physical_hbm_bytes": measured_state_bytes,
                    "remaining_state_budget_bytes": total_hbm_budget_bytes - measured_state_bytes,
                    "implementation_scope": "physical_static_shared_hbm_baseline",
                    "online_reallocation_observed": False,
                    "model_rebuild_count": 0,
                    "blocks_in_primary_pool": blocks,
                    "batch_idx": i,
                    "latency_ms": latency_ms,
                    "latency_semantics": "inference_only",
                    "batch_fingerprint": batch_fingerprint,
                    "seq_history_len": hist_len,
                    "user_id": int(user_ids[0].item()) if torch.is_tensor(user_ids) else int(user_ids[0]),
                    "origin_cached_length": origin_cached_length,
                    "max_origin_cached_length": max_origin_cached_length,
                    "new_tokens": new_tokens,
                    "offload_pages": offload_pages,
                    "max_seqlen": observed_max_seqlen,
                }
                trace_records.append(record)

                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{measure_batches}] latency={latency_ms:.2f}ms hist={hist_len}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            trace_records.append({"split": split_name, "status": "error", "error_message": str(exc)})

        # Cleanup
        kvcache = None
        try:
            kvcache = model.dense_module.async_kvcache
            kvcache.executor.shutdown(wait=True, cancel_futures=True)
            kvcache.onload_worker.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        if kvcache is not None:
            del kvcache
        del model
        gc.collect()
        torch.cuda.empty_cache()

        for r in trace_records:
            with open(out_jsonl, "a") as f:
                f.write(json.dumps(r) + "\n")
        all_trace_records.extend(trace_records)

        valid = [r for r in trace_records if "latency_ms" in r]
        mean_lat = sum(r["latency_ms"] for r in valid) / max(1, len(valid)) if valid else float("nan")
        results.append({"split": split_name, "num_records": len(valid), "mean_latency_ms": mean_lat})

    baseline_report = validate_static_baseline_trace(all_trace_records)
    valid_record_count = int(baseline_report["record_count"])
    complete_sweep = len(results) == len(splits) and all(
        int(result["num_records"]) > 0 for result in results
    )
    with output_path.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {
                    "record_type": "evidence_summary",
                    "evidence_schema_version": 1,
                    "implementation_scope": "physical_static_shared_hbm_baseline",
                    "claim_boundary": (
                        "fixed static embedding/KV splits with measured physical state; "
                        "no online reallocation or unified controller"
                    ),
                    "record_count": valid_record_count,
                    "configured_state_budget_bytes": baseline_report[
                        "configured_state_budget_bytes"
                    ],
                    "physical_rows_validated": baseline_report[
                        "physical_rows_validated"
                    ],
                    "latency_stats": baseline_report["latency_stats"],
                    "static_splits": baseline_report["static_splits"],
                    "claims": {
                        "fixed_shared_hbm_envelope": (
                            "measured" if complete_sweep else "not_measured"
                        ),
                        "physical_embedding_and_kv_accounting": (
                            "measured" if complete_sweep else "not_measured"
                        ),
                        "static_latency_measurement": (
                            "measured" if complete_sweep else "not_measured"
                        ),
                        "paired_hotstate_comparison": "requires_hotstate_trace",
                        "online_cross_type_reallocation": "not_demonstrated",
                        "paper_scale_generalization": "not_demonstrated",
                    },
                }
            )
            + "\n"
        )
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
