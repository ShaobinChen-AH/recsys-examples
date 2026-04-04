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

import os
import time
from itertools import accumulate
from typing import Dict, List, Optional

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
    pad_vbe_kjt_lengths,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    KJTList,
    ShardingType,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from ..dynamicemb_config import DynamicEmbKernel, DynamicEmbScoreStrategy
from ..planner.rw_sharding import RwSequenceDynamicEmbeddingSharding

try:
    from dynamicemb_extensions import expand_table_ids_cuda
except ImportError:
    expand_table_ids_cuda = None

try:
    from dynamicemb_extensions import segmented_unique_cuda
except ImportError:
    segmented_unique_cuda = None

try:
    from dynamicemb_extensions import compute_dedup_lengths_cuda
except ImportError:
    compute_dedup_lengths_cuda = None


def _host_trace_enabled() -> bool:
    return os.environ.get("ASYNC_OVERLAP_HOST_TRACE", "0") == "1"


def _disable_index_dedup_for_experiment() -> bool:
    return os.environ.get("ASYNC_OVERLAP_DISABLE_INDEX_DEDUP", "0") == "1"


def _host_trace_should_print() -> bool:
    if not _host_trace_enabled():
        return False
    target = os.environ.get("ASYNC_OVERLAP_TRACE_STEP", "50")
    return os.environ.get("ASYNC_OVERLAP_CURRENT_STEP") == target


def _host_trace(msg: str, start_ns: Optional[int] = None) -> int:
    now = time.perf_counter_ns()
    if _host_trace_should_print():
        if start_ns is None:
            print(f"[host-trace] {msg}", flush=True)
        else:
            print(f"[host-trace] {msg}: {(now - start_ns) / 1e6:.3f} ms", flush=True)
    return now


def _expand_table_ids_fallback(
    offsets: torch.Tensor,
    table_offsets_in_feature: torch.Tensor,
    num_tables: int,
    local_batch_size: int,
    num_elements: int,
) -> torch.Tensor:
    """Pure-tensor fallback for older dynamicemb_extensions builds."""
    device = offsets.device
    if num_elements == 0:
        return torch.empty(0, dtype=torch.int32, device=device)

    total_buckets = offsets.numel() - 1
    bucket_ids = torch.arange(total_buckets, dtype=torch.int64, device=device)
    feature_ids = torch.div(bucket_ids, local_batch_size, rounding_mode="floor")

    if table_offsets_in_feature is not None and table_offsets_in_feature.numel() > 0:
        table_ids_per_bucket = torch.bucketize(
            feature_ids,
            table_offsets_in_feature[1:-1],
            right=False,
        ).to(torch.int32)
    else:
        table_ids_per_bucket = feature_ids.to(torch.int32)

    lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        table_ids_per_bucket,
        lengths,
        output_size=num_elements,
    )


def _segmented_unique_fallback(
    keys: torch.Tensor,
    table_ids: torch.Tensor,
    num_tables: int,
    input_frequencies: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-tensor fallback for segmented_unique_cuda.

    Keeps semantics close enough for tracing/debugging on older extension builds.
    """
    device = keys.device
    if keys.numel() == 0:
        empty_keys = torch.empty_like(keys)
        empty_idx = torch.empty(0, dtype=torch.uint64, device=device)
        table_offsets = torch.zeros(num_tables + 1, dtype=torch.int64, device=device)
        empty_freq = torch.empty(0, dtype=torch.int64, device=device)
        return (
            torch.zeros(1, dtype=torch.int64, device=device),
            empty_keys,
            empty_idx,
            table_offsets,
            empty_freq,
        )

    # Unique on (table_id, key) pairs. torch.unique returns lexicographically
    # sorted pairs, which still groups rows by table and preserves valid reverse
    # indices for later lookup/unbucketize.
    pair_dtype = torch.int64 if keys.dtype == torch.uint64 else keys.dtype
    pairs = torch.stack([table_ids.to(torch.int64), keys.to(pair_dtype)], dim=1)
    unique_pairs, reverse_idx = torch.unique(
        pairs,
        return_inverse=True,
        dim=0,
        sorted=True,
    )

    unique_keys = unique_pairs[:, 1].to(keys.dtype)
    table_counts = torch.bincount(
        unique_pairs[:, 0].to(torch.int64),
        minlength=num_tables,
    )
    table_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(table_counts, dim=0),
        ]
    )
    num_uniques = table_offsets[-1:].clone()

    if input_frequencies is None:
        freq_counters = torch.empty(0, dtype=torch.int64, device=device)
    else:
        weights = (
            torch.ones_like(reverse_idx, dtype=torch.int64)
            if input_frequencies.numel() == 0
            else input_frequencies.to(torch.int64)
        )
        freq_counters = torch.zeros(
            unique_keys.numel(), dtype=torch.int64, device=device
        )
        freq_counters.scatter_add_(0, reverse_idx.to(torch.int64), weights)

    return (
        num_uniques,
        unique_keys,
        reverse_idx.to(torch.uint64),
        table_offsets,
        freq_counters,
    )


def _compute_dedup_lengths_fallback(
    unique_offsets: torch.Tensor,
    table_offsets_in_feature: torch.Tensor,
    num_tables: int,
    local_batch_size: int,
    new_lengths_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor fallback for older dynamicemb_extensions builds.

    The CUDA helper evenly distributes each table's deduplicated keys across that
    table's (feature, batch) buckets. This fallback keeps the same semantics
    without requiring a rebuilt extension.
    """
    device = unique_offsets.device
    if new_lengths_size == 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )

    unique_counts = unique_offsets[1:] - unique_offsets[:-1]
    features_per_table = table_offsets_in_feature[1:] - table_offsets_in_feature[:-1]
    buckets_per_table = features_per_table * local_batch_size

    # Map each (feature, batch) bucket to its owning table, entirely on device.
    bucket_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(buckets_per_table, dim=0),
        ]
    )
    bucket_ids = torch.arange(new_lengths_size, dtype=torch.int64, device=device)
    table_ids = torch.bucketize(bucket_ids, bucket_offsets[1:-1], right=False)
    local_bucket_ids = bucket_ids - bucket_offsets[table_ids]

    base = torch.div(unique_counts, buckets_per_table, rounding_mode="floor")
    remainder = torch.remainder(unique_counts, buckets_per_table)
    new_lengths = (
        base[table_ids] + (local_bucket_ids < remainder[table_ids]).to(torch.int64)
    ).to(torch.int32)
    new_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.cumsum(new_lengths.to(torch.int64), dim=0),
        ]
    )
    return new_lengths, new_offsets


class DynamicEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Extended EmbeddingCollectionContext that includes frequency_counters for LFU strategy."""

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        seq_vbe_ctx: Optional[List] = None,
        frequency_counters: Optional[List[torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            sharding_contexts, input_features, reverse_indices, seq_vbe_ctx
        )
        self.frequency_counters: List[torch.Tensor] = frequency_counters or []


class _LazyDynamicEmbeddingCollectionForward(
    LazyAwaitable[Dict[str, JaggedTensor]]
):
    """Defers TorchRec's eager input-dist waits until the caller explicitly waits."""

    def __init__(
        self,
        module: "ShardedDynamicEmbeddingCollection",
        ctx: DynamicEmbeddingCollectionContext,
        input_dist_request: Awaitable[Awaitable[KJTList]],
    ) -> None:
        super().__init__()
        self._module = module
        self._ctx = ctx
        self._input_dist_request = input_dist_request

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        t_wait1 = _host_trace("dynemb.forward.input_dist_wait1.enter")
        with record_function("## dynemb_forward_wait_splits ##"):
            with torch.cuda.nvtx.range("dynemb_forward_wait_splits"):
                tensors_awaitable = self._input_dist_request.wait()
        _host_trace("dynemb.forward.input_dist_wait1.return", t_wait1)

        t_wait2 = _host_trace("dynemb.forward.input_dist_wait2.enter")
        with record_function("## dynemb_forward_wait_tensors ##"):
            with torch.cuda.nvtx.range("dynemb_forward_wait_tensors"):
                dist_input = tensors_awaitable.wait()
        _host_trace("dynemb.forward.input_dist_wait2.return", t_wait2)

        t_compute = _host_trace("dynemb.forward.compute_output_dist.enter")
        with record_function("## dynemb_forward_compute_output_dist ##"):
            with torch.cuda.nvtx.range("dynemb_forward_compute_output_dist"):
                output_awaitable = self._module.compute_and_output_dist(
                    self._ctx, dist_input
                )
        _host_trace("dynemb.forward.compute_output_dist.return", t_compute)

        t_output = _host_trace("dynemb.forward.output_wait.enter")
        with record_function("## dynemb_forward_wait_output ##"):
            with torch.cuda.nvtx.range("dynemb_forward_wait_output"):
                result = output_awaitable.wait()
        _host_trace("dynemb.forward.output_wait.return", t_output)
        return result


class ShardedDynamicEmbeddingCollection(ShardedEmbeddingCollection):
    supported_compute_kernels: List[str] = [
        kernel.value for kernel in EmbeddingComputeKernel
    ] + [DynamicEmbKernel]

    def __init__(
        self,
        *args,
        score_strategy: Optional[DynamicEmbScoreStrategy] = None,
        has_admit_strategy: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the global score strategy
        self._score_strategy = score_strategy
        self._is_lfu_enabled = (
            (score_strategy == DynamicEmbScoreStrategy.LFU) if score_strategy else False
        )
        self._has_admit_strategy = has_admit_strategy
        self._disable_index_dedup_for_overlap = _disable_index_dedup_for_experiment()
        if self._disable_index_dedup_for_overlap:
            self._use_index_dedup = False

    def _effective_use_index_dedup(self) -> bool:
        return self._use_index_dedup and not getattr(
            self,
            "_disable_index_dedup_for_overlap",
            _disable_index_dedup_for_experiment(),
        )

    @classmethod
    def create_embedding_sharding(
        cls,
        sharding_type: str,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        """
        override this function to provide customized EmbeddingSharding
        """
        if sharding_type == ShardingType.ROW_WISE.value:
            return RwSequenceDynamicEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        else:
            return super().create_embedding_sharding(
                sharding_type=sharding_type,
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )

    def _create_lookups(self) -> None:
        effective_use_index_dedup = self._effective_use_index_dedup()
        for sharding in self._sharding_type_to_sharding.values():
            if isinstance(sharding, RwSequenceDynamicEmbeddingSharding):
                for config in sharding._grouped_embedding_configs:
                    if (
                        config.compute_kernel
                        is EmbeddingComputeKernel.CUSTOMIZED_KERNEL
                        and config.pooling is not None
                    ):
                        config.fused_params["use_index_dedup"] = (
                            effective_use_index_dedup
                        )
            self._lookups.append(sharding.create_lookup())

    def _create_hash_size_info(
        self,
        feature_names: List[str],
        ctx: Optional[EmbeddingCollectionContext] = None,
    ) -> None:
        super()._create_hash_size_info(feature_names)

        # _is_lfu_enabled is already set in __init__ from score_strategy parameter

        for i, sharding in enumerate(self._sharding_type_to_sharding.values()):
            nonfuse_table_feature_offsets: List[int] = []
            for table in sharding.embedding_tables():
                nonfuse_table_feature_offsets.append(table.num_features())

            nonfuse_table_feature_offsets_cumsum: List[int] = [0] + list(
                accumulate(nonfuse_table_feature_offsets)
            )

            # Register buffers for this shard
            self.register_buffer(
                f"_nonfuse_table_feature_offsets_host_{i}",
                torch.tensor(
                    nonfuse_table_feature_offsets_cumsum,
                    device="cpu",
                    dtype=torch.int64,
                ),
                persistent=False,
            )

            self.register_buffer(
                f"_nonfuse_table_feature_offsets_device_{i}",
                torch.tensor(
                    nonfuse_table_feature_offsets_cumsum,
                    device=self._device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )

    def forward(
        self, *input, **kwargs
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        """Issue input-dist only; defer all waits to the returned LazyAwaitable."""
        t_ctx = _host_trace("dynemb.forward.create_context.enter")
        ctx = self.create_context()
        _host_trace("dynemb.forward.create_context.return", t_ctx)

        t_issue = _host_trace("dynemb.forward.issue_input_dist.enter")
        with record_function("## dynemb_forward_issue_input_dist ##"):
            with torch.cuda.nvtx.range("dynemb_forward_issue_input_dist"):
                input_dist_request = self.input_dist(ctx, *input, **kwargs)
        _host_trace("dynemb.forward.issue_input_dist.return", t_issue)

        return _LazyDynamicEmbeddingCollectionForward(
            module=self,
            ctx=ctx,
            input_dist_request=input_dist_request,
        )

    def _dedup_indices(
        self,
        ctx: DynamicEmbeddingCollectionContext,
        input_feature_splits: List[KeyedJaggedTensor],
    ) -> List[KeyedJaggedTensor]:
        """Deduplicate indices using segmented_unique_cuda."""
        with record_function("## dedup_ec_indices ##"):
            features_by_shards = []
            for i, input_feature in enumerate(input_feature_splits):
                hash_size_offset = self.get_buffer(f"_hash_size_offset_tensor_{i}")
                d_table_offset = self.get_buffer(
                    f"_nonfuse_table_feature_offsets_device_{i}"
                )
                input_feature._values = input_feature._values.contiguous()

                table_num = d_table_offset.numel() - 1
                total_B = input_feature.offsets().numel() - 1
                features = hash_size_offset.numel() - 1
                local_batchsize = total_B // features

                indices = input_feature.values()
                offsets = input_feature.offsets().to(torch.int64)
                num_elements = indices.numel()

                # Handle empty input
                if num_elements == 0:
                    dedup_features = KeyedJaggedTensor(
                        keys=input_feature.keys(),
                        lengths=input_feature.lengths(),
                        offsets=input_feature.offsets(),
                        values=indices,
                    )
                    ctx.input_features.append(input_feature)
                    ctx.reverse_indices.append(
                        torch.empty(0, dtype=torch.uint64, device=self._device)
                    )
                    features_by_shards.append(dedup_features)
                    continue

                # Generate table_ids from jagged offsets (fully on GPU, no sync)
                with record_function("## dynemb_dedup_expand_table_ids ##"):
                    with torch.cuda.nvtx.range("dynemb_dedup_expand_table_ids"):
                        if expand_table_ids_cuda is not None:
                            table_ids = expand_table_ids_cuda(
                                offsets,
                                d_table_offset,
                                table_num,
                                local_batchsize,
                                num_elements,
                            )
                        else:
                            table_ids = _expand_table_ids_fallback(
                                offsets,
                                d_table_offset,
                                table_num,
                                local_batchsize,
                                num_elements,
                            )

                # Prepare input_frequencies tensor to control frequency counting
                input_frequencies = None
                if self._is_lfu_enabled or self._has_admit_strategy:
                    input_frequencies = torch.empty(
                        0, dtype=torch.int64, device=self._device
                    )

                # Call segmented_unique_cuda
                with record_function("## dynemb_dedup_segmented_unique ##"):
                    with torch.cuda.nvtx.range("dynemb_dedup_segmented_unique"):
                        if segmented_unique_cuda is not None:
                            (
                                num_uniques,
                                unique_keys,
                                reverse_idx,
                                table_offsets,
                                freq_counters,
                            ) = segmented_unique_cuda(
                                indices,
                                table_ids,
                                table_num,
                                input_frequencies,
                            )
                        else:
                            (
                                num_uniques,
                                unique_keys,
                                reverse_idx,
                                table_offsets,
                                freq_counters,
                            ) = _segmented_unique_fallback(
                                indices,
                                table_ids,
                                table_num,
                                input_frequencies,
                            )

                # Compute new lengths and offsets using GPU kernel
                # new_lengths_size = total_B (total number of feature/batch buckets)
                with record_function("## dynemb_dedup_lengths ##"):
                    with torch.cuda.nvtx.range("dynemb_dedup_lengths"):
                        if compute_dedup_lengths_cuda is not None:
                            new_lengths, new_offsets = compute_dedup_lengths_cuda(
                                table_offsets,
                                d_table_offset,
                                table_num,
                                local_batchsize,
                                total_B,
                            )
                        else:
                            new_lengths, new_offsets = _compute_dedup_lengths_fallback(
                                table_offsets,
                                d_table_offset,
                                table_num,
                                local_batchsize,
                                total_B,
                            )

                # Get unique values for the KJT
                # .item() implicitly syncs GPU to CPU
                with record_function("## dynemb_dedup_num_uniques_sync ##"):
                    with torch.cuda.nvtx.range("dynemb_dedup_num_uniques_sync"):
                        total_unique = num_uniques.item()
                unique_keys = unique_keys[:total_unique]

                dedup_features = KeyedJaggedTensor(
                    keys=input_feature.keys(),
                    lengths=new_lengths,
                    offsets=new_offsets,
                    values=unique_keys,
                )
                ctx.input_features.append(input_feature)
                ctx.reverse_indices.append(reverse_idx)

                if self._is_lfu_enabled or self._has_admit_strategy:
                    frequency_counters = freq_counters[:total_unique].to(torch.uint64)
                    ctx.frequency_counters.append(frequency_counters)

                features_by_shards.append(dedup_features)
        return features_by_shards

    def input_dist(
        self,
        ctx: DynamicEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        t_input_dist = _host_trace("dynemb.input_dist.enter")
        if self._has_uninitialized_input_dist:
            with record_function("## dynemb_input_dist_init ##"):
                with torch.cuda.nvtx.range("dynemb_input_dist_init"):
                    self._create_input_dist(input_feature_names=features.keys(), ctx=ctx)
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            t_prepare = _host_trace("dynemb.input_dist.prepare.enter")
            with record_function("## dynemb_input_dist_prepare ##"):
                with torch.cuda.nvtx.range("dynemb_input_dist_prepare"):
                    unpadded_features = None
                    if features.variable_stride_per_key():
                        with record_function("## dynemb_input_dist_pad_vbe ##"):
                            with torch.cuda.nvtx.range("dynemb_input_dist_pad_vbe"):
                                unpadded_features = features
                                features = pad_vbe_kjt_lengths(unpadded_features)

                    if self._features_order:
                        with record_function("## dynemb_input_dist_permute ##"):
                            with torch.cuda.nvtx.range("dynemb_input_dist_permute"):
                                features = features.permute(
                                    self._features_order,
                                    self._features_order_tensor,
                                )
                    with record_function("## dynemb_input_dist_split ##"):
                        with torch.cuda.nvtx.range("dynemb_input_dist_split"):
                            features_by_shards = features.split(self._feature_splits)
            _host_trace("dynemb.input_dist.prepare.return", t_prepare)
            effective_use_index_dedup = self._effective_use_index_dedup()
            if effective_use_index_dedup:
                t_dedup = _host_trace("dynemb.input_dist.dedup.enter")
                features_by_shards = self._dedup_indices(ctx, features_by_shards)
                _host_trace("dynemb.input_dist.dedup.return", t_dedup)
            elif self._disable_index_dedup_for_overlap:
                _host_trace("dynemb.input_dist.dedup.skipped_by_env")
                with record_function("## dynemb_input_dist_skip_dedup ##"):
                    with torch.cuda.nvtx.range("dynemb_input_dist_skip_dedup"):
                        pass

            awaitables = []
            t_launch = _host_trace("dynemb.input_dist.launch.enter")
            with record_function("## dynemb_input_dist_launch ##"):
                with torch.cuda.nvtx.range("dynemb_input_dist_launch"):
                    for i, (input_dist, features) in enumerate(
                        zip(self._input_dists, features_by_shards)
                    ):
                        # Attach frequency counters as weights if LFU strategy is enabled
                        if (
                            effective_use_index_dedup
                            and (self._is_lfu_enabled or self._has_admit_strategy)
                            and len(ctx.frequency_counters) > i
                        ):
                            frequency_counters = ctx.frequency_counters[i]
                            features._weights = frequency_counters.float()
                        else:
                            features._weights = None

                        t_launch_one = _host_trace(
                            f"dynemb.input_dist.launch_one[{i}].enter"
                        )
                        awaitables.append(input_dist(features))
                        _host_trace(
                            f"dynemb.input_dist.launch_one[{i}].return", t_launch_one
                        )
                        ctx.sharding_contexts.append(
                            SequenceShardingContext(
                                features_before_input_dist=features,
                                unbucketize_permute_tensor=(
                                    input_dist.unbucketize_permute_tensor
                                    if hasattr(input_dist, "unbucketize_permute_tensor")
                                    else None
                                ),
                            )
                        )
            _host_trace("dynemb.input_dist.launch.return", t_launch)
            if unpadded_features is not None:
                self._compute_sequence_vbe_context(ctx, unpadded_features)
        _host_trace("dynemb.input_dist.return", t_input_dist)
        return KJTListSplitsAwaitable(awaitables, ctx)

    # def create_context(self) -> DynamicEmbeddingCollectionContext:
    #     return DynamicEmbeddingCollectionContext(sharding_contexts=[])

    def create_context(self) -> DynamicEmbeddingCollectionContext:
        # pre-allocate frequency_counters list, ensure all ranks have the same structure
        frequency_counters = (
            [] if not (self._is_lfu_enabled or self._has_admit_strategy) else None
        )
        return DynamicEmbeddingCollectionContext(
            sharding_contexts=[], frequency_counters=frequency_counters
        )


class DynamicEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    """
    DynamicEmbeddingCollectionSharder extends the EmbeddingCollectionSharder class from the TorchREC repo.

    TorchREC performs deduplication in static embedding collections using fuse unique, but fuse unique is not
    suitable for dynamic embedding. Therefore, DynamicEmbeddingCollectionSharder inherits from the
    EmbeddingCollectionSharder class and overrides the shard method to create ShardedDynamicEmbeddingCollection
    and override its input_dist method.

    The usage is completely consistent with TorchREC's EmbeddingCollectionSharder.
    """

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingCollection:
        # Extract global score_strategy from params (only once, as it's a global configuration)
        # Strategy is expected to be consistent across all tables
        global_score_strategy = None
        has_admit_strategy = False
        if global_score_strategy is None:
            for param_name, param_sharding in params.items():
                if (
                    hasattr(param_sharding, "dynamicemb_options")
                    and param_sharding.dynamicemb_options
                ):
                    if param_sharding.dynamicemb_options.score_strategy is not None:
                        global_score_strategy = (
                            param_sharding.dynamicemb_options.score_strategy
                        )

                    if param_sharding.dynamicemb_options.admit_strategy is not None:
                        has_admit_strategy = True

                    break

        # Pass score_strategy directly as a parameter to ShardedDynamicEmbeddingCollection
        return ShardedDynamicEmbeddingCollection(
            module,
            params,
            env,
            self.fused_params,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            use_index_dedup=self._use_index_dedup,
            score_strategy=global_score_strategy,  # Pass as direct parameter
            has_admit_strategy=has_admit_strategy,
        )
