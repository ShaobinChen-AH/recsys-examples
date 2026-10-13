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
from modules.hotstate.physical_budget import get_hbm_budget_bytes, get_hotstate_enabled, GIB
from modules.hotstate.hotstate_controller import HotStateController
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RankingArgs

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr


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
    HASH_SIZE = 1000_000
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
                vocab_size=HASH_SIZE,
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


def _estimate_embedding_value_density(emb_configs):
    """Estimate expected value density (bytes per state) for embedding states.

    Returns a value in range [0.1, 1.0] representing relative embedding pressure.
    Higher = more value in embedding states.
    """
    # Count total logical embedding bytes and unique keys
    total_emb_bytes = 0
    unique_keys = 0
    for config in emb_configs:
        if config.use_dynamicemb:
            # Dynamic embedding: sparse, multiple small accesses
            total_emb_bytes += config.vocab_size * config.dim * 4  # fp32
            unique_keys += config.vocab_size

    if unique_keys == 0:
        return 0.5  # Default if no dynamic embeddings

    # Embeddings are valuable when vocab size is large (sparse access patterns)
    # Normalize by typical sequence length (assume ~4096 tokens)
    sparsity_ratio = min(1.0, unique_keys / 4096.0)
    return 0.3 + 0.5 * sparsity_ratio  # Range [0.3, 0.8]


def _estimate_kv_value_density(max_batch_size, total_max_seqlen, num_layers):
    """Estimate expected value density for KV cache states.

    Returns a value in range [0.1, 1.0] representing relative KV pressure.
    Higher = more value in KV states.
    """
    # KV cache is valuable when sequence is long (large recompute cost)
    # Normalize by typical sequence (assume ~4096 tokens)
    seq_pressure = min(1.0, total_max_seqlen / 4096.0)

    # KV cache is valuable when batch is small (few concurrent sequences)
    batch_pressure = 1.0 / max(1.0, math.sqrt(max_batch_size))

    return 0.4 + 0.4 * seq_pressure + 0.2 * batch_pressure  # Range [0.4, 1.0]


def _compute_adaptive_hbm_split(
    emb_configs, max_batch_size, total_max_seqlen, num_layers, available_hbm_bytes
):
    """Compute adaptive HBM split based on workload characteristics.

    Returns (embedding_hbm_bytes, kv_hbm_bytes) based on estimated value density.
    """
    emb_density = _estimate_embedding_value_density(emb_configs)
    kv_density = _estimate_kv_value_density(max_batch_size, total_max_seqlen, num_layers)

    # Normalize densities to probabilities
    total_density = emb_density + kv_density
    embedding_ratio = emb_density / total_density if total_density > 0 else 0.5
    kv_ratio = kv_density / total_density if total_density > 0 else 0.5

    embedding_hbm_bytes = int(available_hbm_bytes * embedding_ratio)
    kv_hbm_bytes = available_hbm_bytes - embedding_hbm_bytes

    return embedding_hbm_bytes, kv_hbm_bytes


def get_inference_hstu_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
    use_kvcache,
    return_controller=False,
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

    if total_hbm_bytes > 0:
        # Reserve safety margin for activations and fragmentation
        DEFAULT_SAFETY_MARGIN_BYTES = 64 * 1024 * 1024  # 64 MiB
        available_hbm_bytes = total_hbm_bytes - DEFAULT_SAFETY_MARGIN_BYTES

        if hotstate_enabled:
            # HotState: adaptive split based on workload characteristics
            embedding_hbm_bytes, kv_hbm_bytes = _compute_adaptive_hbm_split(
                emb_configs=emb_configs,
                max_batch_size=max_batch_size,
                total_max_seqlen=total_max_seqlen,
                num_layers=network_args.num_layers,
                available_hbm_bytes=available_hbm_bytes,
            )
        else:
            # Fixed split: 60% embedding, 40% KV
            embedding_hbm_bytes = int(available_hbm_bytes * 0.6)
            kv_hbm_bytes = int(available_hbm_bytes * 0.4)
    else:
        # No budget constraint
        embedding_hbm_bytes = 0
        kv_hbm_bytes = 0

    # Compute KV cache blocks from budget
    if kv_hbm_bytes > 0:
        # KV page size: num_layers * 2 * page_size * num_kv_heads * head_dim * element_bytes
        num_layers = network_args.num_layers
        num_kv_heads = network_args.num_attention_heads  # Assuming num_kv_heads == num_attention_heads
        head_dim = network_args.kv_channels
        page_size = 32
        element_bytes = 2  # bfloat16/fp16

        kv_page_bytes = num_layers * 2 * page_size * num_kv_heads * head_dim * element_bytes
        blocks_in_primary_pool = max(256, kv_hbm_bytes // kv_page_bytes)
    else:
        blocks_in_primary_pool = 10240  # Default

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
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()
    model.load_checkpoint(checkpoint_dir)
    model.eval()

    # Optionally create HotStateController for dynamic allocation
    hotstate_controller = None
    print(f"[DEBUG] Controller check: return_controller={return_controller}, hotstate_enabled={hotstate_enabled}, total_hbm_bytes={total_hbm_bytes}")
    if return_controller and hotstate_enabled and total_hbm_bytes > 0:
        try:
            kv_module = getattr(model, '_kv_cache_manager', None)
            embedding_module = getattr(model, 'dense_module', None)
            print(f"[DEBUG] Modules: kv_module={kv_module is not None}, embedding_module={embedding_module is not None}")
            if kv_module is not None:
                hotstate_controller = HotStateController(
                    total_hbm_bytes=total_hbm_bytes,
                    kv_module=kv_module,
                    embedding_module=embedding_module,
                    configured_state_budget_bytes=embedding_hbm_bytes + kv_hbm_bytes,
                )
                # Attach controller to model for easy access
                model._hotstate_controller = hotstate_controller
        except Exception as e:
            print(f"[HotState] Failed to create controller: {e}")

    if return_controller:
        return model, hotstate_controller
    return model


def run_ranking_gr_simulate(
    checkpoint_dir: str,
    check_auc: bool = False,
    disable_contextual_features: bool = False,
    disable_kvcache: bool = False,
    max_bs: int = 1,
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

    max_batch_size = max_bs
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    with torch.inference_mode():
        # Get model and optionally the controller for dynamic allocation
        hotstate_enabled = get_hotstate_enabled()
        total_hbm_bytes = get_hbm_budget_bytes()
        return_controller = hotstate_enabled and total_hbm_bytes > 0

        if return_controller:
            model, hotstate_controller = get_inference_hstu_model(
                emb_configs,
                max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                checkpoint_dir,
                True,
                return_controller=True,
            )
        else:
            model = get_inference_hstu_model(
                emb_configs,
                max_batch_size,
                num_contextual_features,
                total_max_seqlen,
                checkpoint_dir,
                True,
                return_controller=False,
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
            item_vocab_size=10_000_000,
            userid_name="user_id",
            date_name="date",
            sequence_endptr_name="interval_indptr",
            timestamp_names=["date", "interval_end_ts"],
        )

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        num_batches_ctr = 0
        start_time = time.time()
        cur_date = None
        scheduler_telemetry_interval = 100  # Log scheduler state every N batches
        cross_state_interval = 200  # Log cross-state comparison every N batches
        semantic_interval = 300  # Log semantic policies every N batches
        while True:
            try:
                uids, dates, seq_endptrs = next(dataloader_iter)
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
                    # GAP 3: Transfer Scheduler Telemetry
                    # Log scheduler coordination metrics periodically
                    # ============================================================
                    if hotstate_controller is not None and num_batches_ctr % scheduler_telemetry_interval == 0:
                        try:
                            coordination = hotstate_controller.get_transfer_coordination_metrics()
                            value_scoring = hotstate_controller.get_value_scoring_metrics()
                            scheduler_state = hotstate_controller.get_scheduler_state()

                            print(
                                f"[Batch {num_batches_ctr}] Scheduler Coordination: "
                                f"Transfers={{In-Flight: {scheduler_state.get('in_flight_transfers', 0)}, "
                                f"Queued: {scheduler_state.get('queued_transfers', 0)}}} | "
                                f"Coordination={{Overlapped: {coordination.get('overlapped_transfers', 0)}, "
                                f"Ratio: {coordination.get('coordination_ratio', 0):.1%}}} | "
                                f"Value Scoring={{Emb: {value_scoring.get('embedding_avg_value_density', 0):.4f}, "
                                f"KV: {value_scoring.get('kv_avg_value_density', 0):.4f}}} "
                                f"ms/byte"
                            )
                        except Exception as e:
                            print(f"[HotState] Scheduler telemetry failed: {e}")

                    # ============================================================
                    # GAP 4: Cross-State Value Comparison and Unified Eviction
                    # Log comparison metrics periodically
                    # ============================================================
                    if hotstate_controller is not None and num_batches_ctr % cross_state_interval == 0:
                        try:
                            normalized_densities = hotstate_controller.compute_normalized_value_densities()
                            candidates = hotstate_controller.build_unified_eviction_candidates()

                            print(
                                f"[Batch {num_batches_ctr}] Cross-State Comparison: "
                                f"Unified Pool={{Total: {candidates.get('total_candidates', 0)}, "
                                f"Embedding: {candidates.get('embedding_candidates', 0)}, "
                                f"KV: {candidates.get('kv_candidates', 0)}}} | "
                                f"Value Densities={{Emb: {normalized_densities.get('embedding_value_density_ms_per_byte', 0):.6f}, "
                                f"KV: {normalized_densities.get('kv_value_density_ms_per_byte', 0):.6f}}} "
                                f"ms/byte | More Valuable: "
                                f"{'Embedding' if normalized_densities.get('embedding_is_more_valuable') else 'KV'}"
                            )
                        except Exception as e:
                            print(f"[HotState] Cross-state comparison failed: {e}")

                    # ============================================================
                    # GAP 5: Semantic Differences and Consistency Policies
                    # Log semantic policy adherence periodically
                    # ============================================================
                    if hotstate_controller is not None and num_batches_ctr % semantic_interval == 0:
                        try:
                            semantic_plan = hotstate_controller.compute_semantic_allocation()
                            embedding_policy = hotstate_controller.get_semantic_policy("embedding")
                            kv_policy = hotstate_controller.get_semantic_policy("kv_cache")

                            print(
                                f"[Batch {num_batches_ctr}] Semantic Policies: "
                                f"Embedding={{Consistency: {embedding_policy.get('consistency_class', 'N/A')}, "
                                f"Regenerable: {embedding_policy.get('regenerable', False)}, "
                                f"Staleness Tolerance: {embedding_policy.get('staleness_tolerance_epochs', 0)} epochs}} | "
                                f"KV={{Consistency: {kv_policy.get('consistency_class', 'N/A')}, "
                                f"Session-Bound: {kv_policy.get('session_bound', False)}, "
                                f"Authoritative: {kv_policy.get('authoritative_copy_required', False)}}} | "
                                f"KV Min Budget: {semantic_plan.get('kv_ratio', 0):.1%}"
                            )
                        except Exception as e:
                            print(f"[HotState] Semantic policy logging failed: {e}")

                    # ============================================================
                    # GAP 2: Continuous Budget Monitoring and Enforcement
                    # Check if allocated budgets are being respected
                    # ============================================================
                    if hotstate_controller is not None:
                        # Validate budget enforcement periodically
                        validation = hotstate_controller.validate_budget_enforcement()

                        # Check if emergency reallocation is needed
                        emergency_realloc = hotstate_controller.should_trigger_emergency_reallocation()

                        # Decide allocation frequency based on enforcement success
                        hotstate_controller.update_reallocation_frequency()
                        realloc_interval = hotstate_controller._reallocation_interval

                        # Trigger reallocation if:
                        # 1. Regular interval check
                        # 2. Emergency condition (budget violations)
                        should_realloc = (
                            (num_batches_ctr % realloc_interval == 0) or
                            emergency_realloc
                        )

                        if should_realloc:
                            try:
                                # Use corrective allocation if violations detected, else adaptive
                                if emergency_realloc:
                                    allocation_plan = hotstate_controller.compute_corrective_allocation()
                                    print(
                                        f"[Batch {num_batches_ctr}] EMERGENCY REALLOCATION: "
                                        f"{allocation_plan.get('decision_reason', 'unknown')} | "
                                        f"Success Rate: {validation.get('enforcement_success_rate', 0):.1%}"
                                    )
                                else:
                                    allocation_plan = hotstate_controller.compute_adaptive_allocation()

                                hotstate_controller.enforce_adaptive_allocation(allocation_plan)

                                # Log budget enforcement metrics periodically
                                if num_batches_ctr % (realloc_interval * 5) == 0:
                                    drift = validation["current_drift"]
                                    print(
                                        f"[Batch {num_batches_ctr}] Budget Enforcement: "
                                        f"Emb Drift={drift.get('embedding_drift_ratio', 0):.1%} "
                                        f"KV Drift={drift.get('kv_drift_ratio', 0):.1%} | "
                                        f"Success Rate={validation.get('enforcement_success_rate', 0):.1%} | "
                                        f"Interval={realloc_interval}"
                                    )
                            except Exception as e:
                                print(f"[HotState] Budget monitoring check failed: {e}")
                        else:
                            # Between reallocation checks, still validate budget
                            if validation.get("enforcement_success_rate", 1.0) < 0.5 and num_batches_ctr % 10 == 0:
                                drift = validation["current_drift"]
                                if drift.get("has_embedding_violation") or drift.get("has_kv_violation"):
                                    print(
                                        f"[Batch {num_batches_ctr}] WARNING: Budget violation detected "
                                        f"(next check in {realloc_interval} batches)"
                                    )

                    # ============================================================
                    # Call HotState controller BEFORE each inference batch
                    # This invokes: transfer scheduling + dynamic allocation
                    # ============================================================
                    hotstate_result = None
                    if hotstate_controller is not None:
                        try:
                            # Run the control epoch: dynamic allocation + transfer planning
                            # This calls execute_residency_plan() and stage_kv_request()
                            hotstate_result = hotstate_controller.before_batch(
                                batch=batch,
                                user_ids=uids,
                                total_history_lengths=total_history_lengths,
                                batch_idx=num_batches_ctr,
                            )

                            # Log transfer scheduler decisions (GAP 3: Verify scheduler is active)
                            if hotstate_result and num_batches_ctr % (scheduler_telemetry_interval * 2) == 0:
                                admitted_emb = len(hotstate_result.get("admitted_keys", []))
                                evicted_emb = len(hotstate_result.get("evicted_keys", []))
                                if admitted_emb > 0 or evicted_emb > 0:
                                    print(
                                        f"[Batch {num_batches_ctr}] Transfer Scheduler Decisions: "
                                        f"Admitted Emb={admitted_emb}, Evicted Emb={evicted_emb}, "
                                        f"(Coordinated embedding+KV transfers)"
                                    )
                        except Exception as e:
                            print(f"[HotState] before_batch failed: {e}")
                            import traceback
                            traceback.print_exc()

                    if not disable_kvcache:
                        logits = model.forward_with_kvcache(
                            batch,
                            uids,
                            total_history_lengths,
                        )
                    else:
                        model.forward_nokvcache(batch)
                    # eval_module(logits, batch.labels)

                num_batches_ctr += 1
                # if num_batches_ctr == 1000:
                #     break
            except StopIteration:
                break
        end_time = time.time()
        print("Total #batch:", num_batches_ctr)
        print("Total time(s):", end_time - start_time)

        # ================================================================
        # GAP 3: Report Transfer Scheduler Activity Summary
        # ================================================================
        if hotstate_controller is not None:
            try:
                budget_summary = hotstate_controller.report_budget_enforcement_summary()
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
        return_controller = hotstate_enabled and total_hbm_bytes > 0

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

                if not disable_kvcache:
                    logits = model.forward_with_kvcache(
                        batch, user_ids, total_history_lengths
                    )
                else:
                    logits = model.forward_nokvcache(batch)
                eval_module(logits, batch.labels.values())
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
        )
    print("Finished.")
