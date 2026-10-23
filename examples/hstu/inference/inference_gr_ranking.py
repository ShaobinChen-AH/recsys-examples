# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import enum
import math
import os
import sys
import time

import gin
import numpy as np
import torch
from commons.datasets import get_data_loader
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.datasets.inference_dataset import InferenceDataset
from commons.hstu_data_preprocessor import get_common_preprocessors
from commons.utils.stringify import stringify_dict
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RankingConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from modules.metrics import get_multi_event_metric_module
from modules.hotstate.physical_budget import (
    get_embedding_hbm_ratio,
    get_hbm_budget_bytes,
    get_hotstate_enabled,
    plan_physical_budget,
)
from modules.hotstate.policy import load_policy_config
from modules.hotstate.inference_runtime import run_inference_request
from modules.hotstate.sweep_rng import seed_sweep_rngs
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr

VIDEO_ID_TABLE_CAPACITY = 1_000_000
VIDEO_ID_INPUT_KEY_SPACE = 10_000_000


def _sweep_seed() -> tuple[int, int]:
    """Read the deterministic seed contract supplied by the sweep runner."""
    return (
        int(os.getenv("HOTSTATE_SWEEP_SEED", "0")),
        int(os.getenv("HOTSTATE_SWEEP_REPEAT", "0")),
    )


class RunningMode(enum.Enum):
    EVAL = "eval"
    SIMULATE = "simulate"

    def __str__(self):
        return self.value


def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False,
):
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    if dataset_args.dataset_name == "kuairand-1k":
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=1000,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["user_active_degree"],
                table_name="user_active_degree",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["follow_user_num_range"],
                table_name="follow_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["fans_user_num_range"],
                table_name="fans_user_num_range",
                vocab_size=9,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["friend_user_num_range"],
                table_name="friend_user_num_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["register_days_range"],
                table_name="register_days_range",
                vocab_size=8,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
            InferenceEmbeddingConfig(
                feature_names=["video_id"],
                table_name="video_id",
                vocab_size=VIDEO_ID_TABLE_CAPACITY,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["action_weights"],
                table_name="action_weights",
                vocab_size=233,
                dim=embedding_dim,
                use_dynamicemb=False,
            ),
        ]
        return (
            dataset_args,
            embedding_configs
            if not disable_contextual_features
            else embedding_configs[-2:],
        )

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")




def get_inference_hstu_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
    use_kvcache,
    return_controller=False,
    max_num_candidates=100,
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    # elif network_args.dtype_str == "float16":
    #     inference_dtype = torch.float16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        max_batch_size=max_batch_size,
        max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
        scaling_seqlen=total_max_seqlen,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
    )

    # Compute HBM budget allocation
    total_hbm_bytes = get_hbm_budget_bytes()
    hotstate_enabled = get_hotstate_enabled()
    policy_config = load_policy_config()
    embedding_ratio = None

    if total_hbm_bytes > 0:
        ratio_override = get_embedding_hbm_ratio()
        embedding_ratio = policy_config.construction_embedding_ratio(
            hotstate_enabled=hotstate_enabled,
            override=ratio_override,
        )
        budget_plan = plan_physical_budget(
            state_budget_bytes=total_hbm_bytes,
            embedding_hbm_ratio=embedding_ratio,
            num_layers=network_args.num_layers,
            num_kv_heads=network_args.num_attention_heads,
            head_dim=network_args.kv_channels,
            num_tokens_per_page=32,
            max_batch_size=max_batch_size,
            max_sequence_length=math.ceil(total_max_seqlen / 32) * 32,
            num_tokens_per_chunk=1024,
        )
        embedding_hbm_bytes = budget_plan.embedding_hbm_bytes
        kv_hbm_bytes = budget_plan.kv_budget_bytes
        blocks_in_primary_pool = budget_plan.blocks_in_primary_pool
        kv_page_bytes = budget_plan.kv_page_bytes
    else:
        # No budget constraint
        embedding_hbm_bytes = 0
        kv_hbm_bytes = 0

    if total_hbm_bytes <= 0:
        blocks_in_primary_pool = 10240
        kv_page_bytes = 0

    # Log the construction-time pool split and fixed capacities.
    import sys
    print(f"[HOTSTATE_CONSTRUCTION] state_budget_bytes={total_hbm_bytes}", file=sys.stderr)
    print(f"[HOTSTATE_CONSTRUCTION] controller_enabled={hotstate_enabled}", file=sys.stderr)
    print(f"[HOTSTATE_CONSTRUCTION] embedding_pool_bytes={embedding_hbm_bytes}", file=sys.stderr)
    print(
        "[HOTSTATE_CONSTRUCTION] "
        f"embedding_pool_ratio={embedding_ratio if embedding_ratio is not None else 'unbounded'}",
        file=sys.stderr,
    )
    print(f"[HOTSTATE_CONSTRUCTION] kv_pool_bytes={kv_hbm_bytes}", file=sys.stderr)
    print(f"[HOTSTATE_CONSTRUCTION] kv_page_bytes={kv_page_bytes if kv_hbm_bytes > 0 else 0}", file=sys.stderr)
    print(f"[HOTSTATE_CONSTRUCTION] kv_primary_pages={blocks_in_primary_pool}", file=sys.stderr)
    print("[HOTSTATE_CONSTRUCTION] pool_split_fixed_after_model_build=true", file=sys.stderr)

    kvcache_args = {
        "blocks_in_primary_pool": blocks_in_primary_pool,
        "page_size": 32,
        "offload_chunksize": 1024,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    ranking_args = RankingArgs()
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )

    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }

    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config if use_kvcache else None,
        task_config=task_config,
        use_cudagraph=False,
        cudagraph_configs=hstu_cudagraph_configs,
        dynamic_embedding_hbm_bytes=embedding_hbm_bytes,
        hotstate_state_budget_bytes=0,
        hotstate_policy_config=policy_config,
        hotstate_trace_path=os.getenv("HOTSTATE_TRACE_PATH", "") or None,
    )

    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()

    model.load_checkpoint(checkpoint_dir)

    model.eval()

    if return_controller and hotstate_enabled and total_hbm_bytes > 0 and use_kvcache:
        model.dense_module.enable_hotstate(
            total_hbm_bytes=total_hbm_bytes,
            configured_state_budget_bytes=total_hbm_bytes,
            policy_config=policy_config,
            trace_path=os.getenv("HOTSTATE_TRACE_PATH") or None,
        )
        model.dense_module.set_hotstate_embedding_module(model.sparse_module)
        model.dense_module.hotstate.validate_physical_budget()
    hotstate_controller = model.dense_module.hotstate

    if return_controller:
        return model, hotstate_controller
    return model


class EmbeddingAccessTracer:
    """Instrument embedding accesses for offline value-model diagnostics.

    Tracks per-table statistics for each dynamicemb table separately:
    - Per-table unique items accessed vs vocab size
    - Per-table working set footprint
    - Per-table access frequency distribution

    The report aggregates these measurements and prints a candidate formula;
    it does not modify the runtime allocation heuristic.
    """
    def __init__(self):
        # Per-table tracking: feature_name -> stats
        self.table_stats = {}  # feature -> {unique_per_batch: [], cumulative: set(), counts: {}}

    def record_batch_access(self, features, emb_configs):
        """Extract and record per-table embedding IDs from KeyedJaggedTensor.

        Args:
            features: KeyedJaggedTensor from batch.features
            emb_configs: List of InferenceEmbeddingConfig with feature_names
        """
        # Build map of feature_name -> config for dynamicemb tables
        # Note: each config can have multiple feature_names
        dynamicemb_map = {}
        for cfg in emb_configs:
            if cfg.use_dynamicemb:
                for fname in cfg.feature_names:
                    dynamicemb_map[fname] = cfg

        # Extract IDs per feature
        if hasattr(features, 'keys') and callable(features.keys):
            feature_keys = features.keys()

            for feature_name in feature_keys:
                # Only track dynamicemb features
                if feature_name not in dynamicemb_map:
                    continue

                config = dynamicemb_map[feature_name]

                # Initialize tracking for this feature if first time
                if feature_name not in self.table_stats:
                    self.table_stats[feature_name] = {
                        'unique_per_batch': [],
                        'cumulative': set(),
                        'counts': {},  # item_id -> access count
                        'config': config,  # Store config reference
                    }

                stats = self.table_stats[feature_name]

                try:
                    # Get values for this specific feature
                    feature_tensor = features[feature_name]
                    if hasattr(feature_tensor, 'values'):
                        values_tensor = feature_tensor.values()
                    else:
                        values_tensor = feature_tensor

                    if torch.is_tensor(values_tensor):
                        # Convert to CPU and extract unique IDs for this batch
                        item_ids = values_tensor.cpu().numpy().tolist()
                        unique_this_batch = set(item_ids)

                        # Record per-batch unique count
                        stats['unique_per_batch'].append(len(unique_this_batch))

                        # Update cumulative unique set
                        stats['cumulative'].update(unique_this_batch)

                        # Update global frequency counts
                        for item_id in unique_this_batch:
                            stats['counts'][item_id] = stats['counts'].get(item_id, 0) + 1

                except Exception as e:
                    # Record zero for this batch if extraction fails
                    stats['unique_per_batch'].append(0)

    def analyze_and_report(self):
        """Report per-table traces and print an offline formula candidate."""
        if len(self.table_stats) == 0:
            print("[EmbeddingAccessTracer] No data collected")
            return None

        print("\n" + "="*80)
        print("EMBEDDING ACCESS PATTERN ANALYSIS (Per-Table, Data-Driven)")
        print("="*80)

        # Aggregate stats across all features
        total_capacity_bytes = 0
        table_analyses = []

        for feature_name, stats in self.table_stats.items():
            # Get config from stored reference
            config = stats.get('config')
            if config is None:
                continue

            vocab_size = config.vocab_size
            dim = config.dim

            # Per-batch statistics
            unique_per_batch = stats['unique_per_batch']
            if len(unique_per_batch) == 0 or sum(unique_per_batch) == 0:
                continue

            mean_unique = np.mean(unique_per_batch)
            p95_unique = np.percentile(unique_per_batch, 95)

            # Cumulative working set
            cumulative_unique = len(stats['cumulative'])
            coverage_ratio = cumulative_unique / vocab_size if vocab_size > 0 else 0

            # Working set footprint in bytes
            bytes_per_item = dim * 4  # fp32
            mean_ws_bytes = mean_unique * bytes_per_item
            p95_ws_bytes = p95_unique * bytes_per_item

            # Access frequency distribution
            sorted_counts = sorted(stats['counts'].values(), reverse=True)
            total_accesses = sum(sorted_counts)
            top_10_pct = int(0.10 * len(sorted_counts))
            top_10_pct_accesses = sum(sorted_counts[:top_10_pct]) if top_10_pct > 0 else 0
            access_concentration = top_10_pct_accesses / total_accesses if total_accesses > 0 else 0

            # Full capacity for this feature's vocab
            full_capacity_bytes = vocab_size * bytes_per_item
            total_capacity_bytes += full_capacity_bytes

            table_analyses.append({
                'feature_name': feature_name,
                'vocab_size': vocab_size,
                'dim': dim,
                'mean_unique_per_batch': mean_unique,
                'p95_unique_per_batch': p95_unique,
                'cumulative_unique': cumulative_unique,
                'coverage_ratio': coverage_ratio,
                'mean_ws_bytes': mean_ws_bytes,
                'p95_ws_bytes': p95_ws_bytes,
                'access_concentration': access_concentration,
                'full_capacity_bytes': full_capacity_bytes,
                'total_accesses': total_accesses,
            })

        if len(table_analyses) == 0:
            print("[EmbeddingAccessTracer] No valid table data collected")
            return None

        # Sort by capacity (largest tables first)
        table_analyses.sort(key=lambda x: x['full_capacity_bytes'], reverse=True)

        # Print per-table analysis
        print(f"\nPER-TABLE ANALYSIS ({len(table_analyses)} dynamicemb features):")
        print()

        for i, analysis in enumerate(table_analyses, 1):
            capacity_mb = analysis['full_capacity_bytes'] / (1024 * 1024)
            mean_ws_mb = analysis['mean_ws_bytes'] / (1024 * 1024)
            p95_ws_mb = analysis['p95_ws_bytes'] / (1024 * 1024)

            print(f"{i}. FEATURE: {analysis['feature_name']}")
            print(f"   Vocab size:              {analysis['vocab_size']:,}")
            print(f"   Full capacity:           {capacity_mb:.1f} MB")
            print(f"   Per-batch unique (mean): {analysis['mean_unique_per_batch']:.1f} items")
            print(f"   Per-batch unique (P95):  {analysis['p95_unique_per_batch']:.1f} items")
            print(f"   Working set (mean):      {mean_ws_mb:.2f} MB")
            print(f"   Working set (P95):       {p95_ws_mb:.2f} MB")
            print(f"   Cumulative unique:       {analysis['cumulative_unique']:,}")
            print(f"   Coverage ratio:          {analysis['coverage_ratio']:.2%}")
            print(f"   Access concentration:    {analysis['access_concentration']:.1%} (top 10%)")
            print()

        # Compute capacity-weighted metrics
        weighted_mean_ws_bytes = sum(a['mean_ws_bytes'] * a['full_capacity_bytes'] for a in table_analyses) / total_capacity_bytes
        weighted_p95_ws_bytes = sum(a['p95_ws_bytes'] * a['full_capacity_bytes'] for a in table_analyses) / total_capacity_bytes
        weighted_coverage = sum(a['coverage_ratio'] * a['full_capacity_bytes'] for a in table_analyses) / total_capacity_bytes
        weighted_concentration = sum(a['access_concentration'] * a['full_capacity_bytes'] for a in table_analyses) / total_capacity_bytes

        print("="*80)
        print("CAPACITY-WEIGHTED AGGREGATE METRICS:")
        print("="*80)
        print(f"Total embedding capacity:    {total_capacity_bytes / (1024**2):.1f} MB")
        print(f"Weighted mean working set:   {weighted_mean_ws_bytes / (1024**2):.2f} MB")
        print(f"Weighted P95 working set:    {weighted_p95_ws_bytes / (1024**2):.2f} MB")
        print(f"Weighted coverage ratio:     {weighted_coverage:.2%}")
        print(f"Weighted concentration:      {weighted_concentration:.1%}")
        print()

        # Diagnostic-only candidate formula generated from the collected trace.
        print("="*80)
        print("OFFLINE VALUE DENSITY FORMULA CANDIDATE (not executed):")
        print("="*80)

        typical_cache_mb = weighted_p95_ws_bytes / (1024 * 1024)

        print(f"""
def _estimate_embedding_value_density(emb_configs, max_batch_size, max_num_candidates):
    # Measured from trace: P95 working set across all dynamicemb tables
    MEASURED_TYPICAL_CACHE_MB = {typical_cache_mb:.2f}  # MB

    # Measured access concentration (top 10% items)
    MEASURED_CONCENTRATION = {weighted_concentration:.3f}

    # Measured coverage ratio (accessed / vocab)
    MEASURED_COVERAGE = {weighted_coverage:.4f}

    # Calculate actual working set footprint from configs
    total_working_set_bytes = 0
    for config in emb_configs:
        if config.use_dynamicemb:
            # Use measured coverage ratio per table
            accessed_items = int(config.vocab_size * MEASURED_COVERAGE)
            total_working_set_bytes += accessed_items * config.dim * 4

    working_set_mb = total_working_set_bytes / (1024 * 1024)

    # Footprint pressure: how much cache needed vs typical
    footprint_pressure = min(1.0, working_set_mb / MEASURED_TYPICAL_CACHE_MB)

    # Embedding value: lower than KV since working set is small
    # and regeneration is cheaper than KV recomputation
    # Range: [0.05, 0.20] vs KV's [0.4, 1.0]
    base_value = 0.05
    pressure_factor = 0.10 * footprint_pressure
    concentration_factor = 0.05 * MEASURED_CONCENTRATION

    return base_value + pressure_factor + concentration_factor
        """)

        print("="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        print(f"1. Actual working set ({weighted_mean_ws_bytes/(1024**2):.1f} MB mean, {weighted_p95_ws_bytes/(1024**2):.1f} MB P95)")
        print(f"   is MUCH smaller than full capacity ({total_capacity_bytes/(1024**2):.1f} MB)")
        print(f"2. Coverage: {weighted_coverage:.2%} - only a fraction of vocab is accessed")
        print(f"3. Access is concentrated: top 10% items = {weighted_concentration:.1%} of accesses")
        print(f"4. The startup heuristic uses vocabulary scale; this trace reports actual accessed IDs")
        print("="*80 + "\n")

        return {
            "table_analyses": table_analyses,
            "weighted_mean_ws_mb": weighted_mean_ws_bytes / (1024**2),
            "weighted_p95_ws_mb": weighted_p95_ws_bytes / (1024**2),
            "weighted_coverage_ratio": weighted_coverage,
            "weighted_concentration": weighted_concentration,
            "total_capacity_mb": total_capacity_bytes / (1024**2),
        }




def run_ranking_gr_simulate(
    checkpoint_dir: str,
    check_auc: bool = False,
    disable_contextual_features: bool = False,
    disable_kvcache: bool = False,
    max_bs: int = 1,
    max_batches: int = 0,
    warmup_batches: int = 0,
):
    if max_batches < 0 or warmup_batches < 0:
        raise ValueError("max_batches and warmup_batches must be non-negative")
    sweep_seed, sweep_repeat = _sweep_seed()
    # Seed before model construction or dataset generation. Each request is
    # reseeded below so model-construction RNG consumption cannot affect the
    # candidate stream used for pairing.
    seed_sweep_rngs(sweep_seed, sweep_repeat, -1)
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = max_bs
    max_num_candidates = dataset_args.max_num_candidates if dataset_args.max_num_candidates > 0 else 100
    total_max_seqlen = dataset_args.max_history_seqlen * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    with torch.inference_mode():
        # Get model and optionally the controller for dynamic allocation
        hotstate_enabled = get_hotstate_enabled()
        use_kvcache = not disable_kvcache

        total_hbm_bytes = get_hbm_budget_bytes()

        return_controller = hotstate_enabled and total_hbm_bytes > 0 and use_kvcache

        if return_controller:
            model, hotstate_controller = get_inference_hstu_model(
                emb_configs,
                max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                checkpoint_dir,
                use_kvcache,
                return_controller=True,
                max_num_candidates=max_num_candidates,
            )
        else:
            model = get_inference_hstu_model(
                emb_configs,
                max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                checkpoint_dir,
                use_kvcache,
                return_controller=False,
                max_num_candidates=max_num_candidates,
            )
            hotstate_controller = None

        if check_auc:
            eval_module = get_multi_event_metric_module(
                num_classes=model.get_num_class(),
                num_tasks=model.get_num_tasks(),
                metric_types=model.get_metric_types(),
            )

        dataset = InferenceDataset(
            seq_logs_file=dataproc._inference_sequence_file,
            batch_logs_file=dataproc._inference_batch_file,
            batch_size=max_batch_size,
            max_seqlen=total_max_seqlen,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names
            if not disable_contextual_features
            else [],
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,
            # Raw video IDs are opaque DynamicEmb keys. Keep the original
            # candidate key space used by the static baseline.
            item_vocab_size=VIDEO_ID_INPUT_KEY_SPACE,
            userid_name="user_id",
            date_name="date",
            sequence_endptr_name="interval_indptr",
            timestamp_names=["date", "interval_end_ts"],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        warmup_batches_ctr = 0
        measured_batches_ctr = 0
        start_time = time.time()
        cur_date = None

        # ============================================================
        # INSTRUMENTATION: Trace actual embedding access patterns
        # ============================================================
        access_tracer = EmbeddingAccessTracer()
        enable_access_tracing = True  # Set to True to collect trace data
        while True:
            try:
                uids, dates, seq_endptrs = next(dataloader_iter)
                is_warmup = warmup_batches_ctr < warmup_batches
                rng_batch_idx = (
                    -(warmup_batches_ctr + 1)
                    if is_warmup
                    else measured_batches_ctr
                )
                seed_sweep_rngs(sweep_seed, sweep_repeat, rng_batch_idx)
                if dates[0] != cur_date:
                    # if cur_date is not None:
                    # eval_metric_dict = eval_module.compute()
                    # print(
                    #     f"[eval]:\n    "
                    #     + stringify_dict(
                    #         eval_metric_dict, prefix="Metrics", sep="\n    "
                    #     )
                    # )
                    # model.clear_kv_cache()
                    if cur_date is not None:
                        break
                    cur_date = dates[0]

                batch = dataset.get_input_batch(
                    uids,
                    dates,
                    seq_endptrs,
                    torch.zeros_like(seq_endptrs),
                    with_contextual_features=True,
                    with_ranking_labels=False,
                )
                total_history_lengths = seq_endptrs * 2 + num_contextual_features

                if batch is not None:
                    # ============================================================
                    # INSTRUMENTATION: Record embedding access for this batch
                    # Extract IDs from batch.features (KeyedJaggedTensor)
                    # ============================================================
                    if enable_access_tracing and hasattr(batch, 'features'):
                        try:
                            access_tracer.record_batch_access(batch.features, emb_configs)
                        except Exception as e:
                            print(f"[AccessTracer] Failed to record batch {num_batches_ctr}: {e}")

                    trace_path_env = None
                    controller_trace_path = None
                    if is_warmup:
                        trace_path_env = os.environ.pop("HOTSTATE_TRACE_PATH", None)
                        if hotstate_controller is not None:
                            controller_trace_path = hotstate_controller.trace_path
                            hotstate_controller.trace_path = None
                    try:
                        if not disable_kvcache and (
                            hotstate_controller is not None
                            or os.getenv("HOTSTATE_TRACE_PATH")
                        ):
                            logits, _record = run_inference_request(
                                model,
                                batch,
                                uids,
                                total_history_lengths,
                                controller=hotstate_controller,
                                batch_idx=measured_batches_ctr,
                                state_budget_bytes=total_hbm_bytes or None,
                                static_split=os.getenv(
                                    "HOTSTATE_STATIC_SPLIT", "static_default"
                                ),
                            )
                        elif not disable_kvcache:
                            logits = model.forward_with_kvcache(
                                batch, uids, total_history_lengths
                            )
                        else:
                            logits = model.forward_nokvcache(batch)
                    finally:
                        if is_warmup:
                            if trace_path_env is not None:
                                os.environ["HOTSTATE_TRACE_PATH"] = trace_path_env
                            if hotstate_controller is not None:
                                hotstate_controller.trace_path = controller_trace_path

                num_batches_ctr += 1
                if batch is None:
                    continue
                if is_warmup:
                    warmup_batches_ctr += 1
                    continue
                measured_batches_ctr += 1
                if max_batches and measured_batches_ctr >= max_batches:
                    break
                # if num_batches_ctr == 1000:
                #     break
            except StopIteration:
                break
        end_time = time.time()
        print("Total #batch:", measured_batches_ctr)
        print("Total #warmup batch:", warmup_batches_ctr)
        print("Total time(s):", end_time - start_time)

        # ============================================================
        # INSTRUMENTATION: Analyze and report embedding access patterns
        # ============================================================
        if enable_access_tracing:
            try:
                trace_analysis = access_tracer.analyze_and_report()
                if trace_analysis:
                    print("\n[AccessTracer] Trace analysis completed successfully")
                    print("   Trace metrics are diagnostics for configured policy sweeps.")
            except Exception as e:
                print(f"[AccessTracer] Analysis failed: {e}")
                import traceback
                traceback.print_exc()

        # ================================================================
        # GAP 3: Report Transfer Scheduler Activity Summary
        # ================================================================
        if hotstate_controller is not None:
            try:
                hotstate_controller.report_residency_limit_summary()
                scheduler_summary = hotstate_controller.report_scheduler_summary()

                # ================================================================
                # GAP 4: Report Cross-State Value Comparison
                # ================================================================
                cross_state_summary = hotstate_controller.report_cross_state_comparison()

                # ================================================================
                # GAP 5: Report Semantic Policy Adherence
                # ================================================================
                semantic_summary = hotstate_controller.report_semantic_adherence()
            except Exception as e:
                print(f"[HotState] Failed to generate summary reports: {e}")


def run_ranking_gr_evaluate(
    checkpoint_dir: str,
    disable_contextual_features: bool = False,
    disable_kvcache: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 1
    total_max_seqlen = (
        dataset_args.max_num_candidates + dataset_args.max_history_seqlen
    ) * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    def strip_padding_batch(batch, unpadded_batch_size):
        batch.batch_size = unpadded_batch_size
        kjt_dict = batch.features.to_dict()
        for k in kjt_dict:
            kjt_dict[k] = JaggedTensor.from_dense_lengths(
                kjt_dict[k].to_padded_dense()[: batch.batch_size],
                kjt_dict[k].lengths()[: batch.batch_size].long(),
            )
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        batch.num_candidates = batch.num_candidates[: batch.batch_size]
        return batch

    with torch.inference_mode():
        use_kvcache = not disable_kvcache
        hotstate_enabled = get_hotstate_enabled()
        total_hbm_bytes = get_hbm_budget_bytes()
        return_controller = hotstate_enabled and total_hbm_bytes > 0 and use_kvcache

        if return_controller:
            model, hotstate_controller = get_inference_hstu_model(
                emb_configs,
                max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                checkpoint_dir,
                use_kvcache,
                return_controller=True,
            )
        else:
            model = get_inference_hstu_model(
                emb_configs,
                max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                checkpoint_dir,
                use_kvcache,
                return_controller=False,
            )
            hotstate_controller = None

        eval_module = get_multi_event_metric_module(
            num_classes=model.get_num_class(),
            num_tasks=model.get_num_tasks(),
            metric_types=model.get_metric_types(),
        )

        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_history_seqlen=dataset_args.max_history_seqlen,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=model.get_num_tasks(),
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
            load_candidate_action=False,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        batch_idx = 0
        # torch.cuda.profiler.start()
        while True:
            try:
                batch = next(dataloader_iter)

                batch = batch.to(device=torch.cuda.current_device())
                d = batch.features.to_dict()
                user_ids = d["user_id"].values().cpu().long()
                if user_ids.shape[0] != batch.batch_size:
                    batch = strip_padding_batch(batch, user_ids.shape[0])
                total_history_lengths = (
                    torch.sum(
                        batch.features.lengths().view(-1, batch.batch_size), 0
                    ).view(-1)
                    - batch.num_candidates
                )
                total_history_lengths = total_history_lengths.cpu()

                if not disable_kvcache and (hotstate_controller is not None or os.getenv("HOTSTATE_TRACE_PATH")):
                    logits, _record = run_inference_request(
                        model, batch, user_ids, total_history_lengths,
                        controller=hotstate_controller,
                        batch_idx=batch_idx,
                        state_budget_bytes=total_hbm_bytes or None,
                        static_split=os.getenv("HOTSTATE_STATIC_SPLIT", "static_default"),
                    )
                elif not disable_kvcache:
                    logits = model.forward_with_kvcache(
                        batch, user_ids, total_history_lengths
                    )
                else:
                    logits = model.forward_nokvcache(batch)
                eval_module(logits, batch.labels.values())
                batch_idx += 1
            except StopIteration:
                break
        # torch.cuda.profiler.stop()

        eval_metric_dict = eval_module.compute()
        print(
            f"[eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=RunningMode, choices=list(RunningMode), required=True
    )
    parser.add_argument("--disable_auc", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--disable_kvcache", action="store_true")
    parser.add_argument("--max_bs", type=int, default=1)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Stop simulate mode after this many batches; 0 processes the full date window.",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=0,
        help="Run this many untraced requests before measurement in simulate mode.",
    )

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    if args.mode == RunningMode.EVAL:
        if args.disable_auc:
            print("disable_auc is ignored in Eval mode.")
        if args.disable_context:
            print("disable_context is ignored in Eval mode.")
        run_ranking_gr_evaluate(
            checkpoint_dir=args.checkpoint_dir, disable_kvcache=args.disable_kvcache
        )
    elif args.mode == RunningMode.SIMULATE:
        if args.disable_kvcache:
            print("disable_kvcache is ignored in Eval mode.")
        run_ranking_gr_simulate(
            checkpoint_dir=args.checkpoint_dir,
            check_auc=not args.disable_auc,
            disable_contextual_features=args.disable_context,
            disable_kvcache=args.disable_kvcache,
            max_bs=args.max_bs,
            max_batches=args.max_batches,
            warmup_batches=args.warmup_batches,
        )
    print("Finished.")
