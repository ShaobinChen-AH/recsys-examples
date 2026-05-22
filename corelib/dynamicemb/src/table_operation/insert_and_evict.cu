/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
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
******************************************************************************/

#include "kernels.cuh"
#include "table.cuh"

namespace dyn_emb {

__global__ void update_counter_with_layout_kernel(
    int32_t *__restrict__ counter, int64_t total_capacity,
    int64_t const *__restrict__ slot_indices, int64_t n, int32_t delta,
    int64_t const *__restrict__ table_ids,
    int64_t const *__restrict__ table_bucket_offsets, int64_t bucket_capacity,
    int64_t main_capacity, int64_t const *__restrict__ overflow_output_offsets,
    int64_t overflow_bucket_capacity, int64_t num_tables) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t slot = slot_indices[i];
  if (slot < 0)
    return;
  int64_t tid = (table_ids != nullptr) ? table_ids[i] : 0;
  int64_t flat = -1;
  if (overflow_output_offsets != nullptr) {
    int64_t per_table_cap = overflow_output_offsets[tid];
    if (slot < per_table_cap) {
      flat = table_bucket_offsets[tid] * bucket_capacity + slot;
    } else {
      flat = main_capacity + tid * overflow_bucket_capacity +
             (slot - per_table_cap);
    }
  } else {
    if (table_bucket_offsets != nullptr && num_tables > 1) {
      flat = table_bucket_offsets[tid] * bucket_capacity + slot;
    } else {
      flat = slot;
    }
  }
  if (flat >= 0 && flat < total_capacity) {
    counter[flat] += delta;
  }
}

template <typename Table, ScorePolicyType PolicyTypeV, bool OutputScoreV,
          int CompactTileSize, bool EnableOverflowV = false>
void launch_table_insert_and_evict_kernel(
    Table table, int64_t *table_bucket_offsets_ptr, int *bucket_sizes_ptr,
    int64_t num_total, typename Table::KeyType *keys_ptr,
    int64_t *table_ids_ptr, InsertResult *insert_results_ptr,
    IndexType *indices_ptr, ScoreType **score_input_ptrs,
    int64_t **score_output_ptrs, ScorePolicyType *extra_policies_arr,
    typename Table::KeyType **table_key_slots_ptr,
    CounterType *evict_counter_ptr, typename Table::KeyType *evicted_keys_ptr,
    int64_t **evicted_scores_ptrs, IndexType *evicted_indices_ptr,
    int64_t *evicted_table_ids_ptr, int32_t *counter_ptr, cudaStream_t stream,
    Table ovf_table = Table(), int *ovf_bucket_sizes_ptr = nullptr,
    int32_t *ovf_counter_ptr = nullptr,
    int64_t *ovf_output_offsets_ptr = nullptr) {
  constexpr int BLOCK_SIZE = 256;
  using KernelTraits =
      InsertKernelTraits<BLOCK_SIZE, 1, 1, CompactTileSize, 8, PolicyTypeV,
                         OutputScoreV, EnableOverflowV>;

  table_insert_and_evict_kernel<Table, KernelTraits>
      <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table, table_bucket_offsets_ptr, bucket_sizes_ptr, num_total,
          keys_ptr, table_ids_ptr, insert_results_ptr, indices_ptr,
          score_input_ptrs, score_output_ptrs, extra_policies_arr,
          table_key_slots_ptr,
          evict_counter_ptr, evicted_keys_ptr, evicted_scores_ptrs,
          evicted_indices_ptr, evicted_table_ids_ptr, counter_ptr, ovf_table,
          ovf_bucket_sizes_ptr, ovf_counter_ptr, ovf_output_offsets_ptr);

  table_unlock_kernel<Table>
      <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
          table, num_total, keys_ptr, table_key_slots_ptr);
}

void table_insert_and_evict_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor indices,
    std::optional<at::Tensor> insert_results,
    std::optional<at::Tensor> score_output, at::Tensor num_evicted,
    at::Tensor evicted_keys, at::Tensor evicted_indices,
    at::Tensor evicted_scores, at::Tensor evicted_table_ids,
    at::Tensor counter, int num_scores,
    const std::vector<std::optional<at::Tensor>> &score_inputs,
    const std::vector<std::optional<at::Tensor>> &score_outputs,
    const std::vector<ScorePolicyType> &extra_policies,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
    std::optional<at::Tensor> ovf_bucket_sizes,
    std::optional<at::Tensor> ovf_counter,
    std::optional<at::Tensor> ovf_output_offsets) {

  auto key_type = get_data_type(keys);

  ScoreType *score_input_ptr = nullptr;
  at::Tensor score_input_tensor;
  if (score_input.has_value() && score_input.value().defined()) {
    at::Tensor in = score_input.value();
    if (in.scalar_type() == torch::kUInt64) {
      score_input_ptr = get_pointer<ScoreType>(score_input);
    } else {
      score_input_tensor = in.view(torch::kUInt64);
      score_input_ptr = score_input_tensor.data_ptr<ScoreType>();
    }
  }

  int64_t *score_output_ptr = nullptr;
  if (score_output.has_value() && score_output.value().defined()) {
    score_output_ptr = score_output.value().data_ptr<int64_t>();
  }

  auto indices_ptr = indices.data_ptr<IndexType>();
  auto insert_results_ptr = get_pointer<InsertResult>(insert_results);
  auto bucket_sizes_ = get_pointer<int>(bucket_sizes);

  auto evict_counter_ = get_pointer<CounterType>(num_evicted);
  auto evicted_scores_ptr = evicted_scores.data_ptr<int64_t>();
  auto evicted_indices_ = get_pointer<IndexType>(evicted_indices);
  auto evicted_table_ids_ptr = evicted_table_ids.data_ptr<int64_t>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto ovf_bucket_sizes_ = get_pointer<int>(ovf_bucket_sizes);
  auto ovf_counter_ptr =
      ovf_counter.has_value() && ovf_counter.value().defined()
          ? ovf_counter.value().data_ptr<int32_t>()
          : nullptr;
  auto ovf_output_offsets_ptr =
      ovf_output_offsets.has_value() && ovf_output_offsets.value().defined()
          ? ovf_output_offsets.value().data_ptr<int64_t>()
          : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  auto table_key_slots = at::zeros(
      num_total, at::TensorOptions().dtype(at::kLong).device(keys.device()));

  bool output_score = (score_output_ptr != nullptr);
  bool use_overflow =
      ovf_storage.has_value() && ovf_storage.value().defined();

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ = get_pointer<KeyType>(keys);
    auto evicted_keys_ = get_pointer<KeyType>(evicted_keys);
    auto table_key_slots_ = get_pointer<KeyType *>(table_key_slots);

    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    // Build score pointer arrays
    ScoreType *score_input_ptrs[kMaxNumScores] = {};
    int64_t *score_output_ptrs[kMaxNumScores] = {};
    int64_t *evicted_scores_ptrs[kMaxNumScores] = {};
    ScorePolicyType extra_policies_arr[kMaxNumScores] = {
        ScorePolicyType::Const};
    score_input_ptrs[0] = score_input_ptr;
    score_output_ptrs[0] = score_output_ptr;
    evicted_scores_ptrs[0] = evicted_scores_ptr;
    for (int s = 0; s < num_scores - 1 && s < (int)extra_policies.size(); ++s)
      extra_policies_arr[s] = extra_policies[s];
    std::vector<at::Tensor> extra_input_tensors(kMaxNumScores);
    std::vector<at::Tensor> extra_evicted_scores(kMaxNumScores);
    for (int s = 1; s < num_scores; ++s) {
      if (s < (int)score_inputs.size() && score_inputs[s].has_value()) {
        at::Tensor in = score_inputs[s].value();
        if (in.scalar_type() == torch::kUInt64) {
          score_input_ptrs[s] = in.data_ptr<ScoreType>();
        } else {
          extra_input_tensors[s] = in.view(torch::kUInt64);
          score_input_ptrs[s] = extra_input_tensors[s].data_ptr<ScoreType>();
        }
      }
      if (s < (int)score_outputs.size() &&
          score_outputs[s].has_value() && score_outputs[s].value().defined()) {
        score_output_ptrs[s] =
            score_outputs[s].value().data_ptr<int64_t>();
      }
      // Allocate extra evicted score tensors
      extra_evicted_scores[s] = torch::empty(
          {num_total}, keys.options().dtype(torch::kInt64));
      evicted_scores_ptrs[s] = extra_evicted_scores[s].data_ptr<int64_t>();
    }

    DISPATCH_NUM_SCORES(num_scores, NumScoresV, [&] {
      using Bucket = LinearBucket<KeyType, NumScoresV>;
      using Table = LinearBucketTable<Bucket>;

      auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                         num_buckets, bucket_capacity);

      Table ovf_table;
      if (use_overflow) {
        at::Tensor ovf = ovf_storage.value();
        int64_t ovf_bucket_bytes = ovf_bucket_capacity * total_size;
        int64_t ovf_num_buckets =
            ovf.numel() * ovf.element_size() / ovf_bucket_bytes;
        ovf_table = Table(reinterpret_cast<uint8_t *>(ovf.data_ptr()),
                          ovf_num_buckets, ovf_bucket_capacity);
      }

      DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
        auto launch = [&](auto compact_tile_size,
                          auto enable_overflow) {
          if (output_score) {
            launch_table_insert_and_evict_kernel<
                Table, PolicyTypeV, true, compact_tile_size.value,
                enable_overflow.value>(
                table, table_bucket_offsets_ptr, bucket_sizes_, num_total,
                keys_, table_ids_ptr, insert_results_ptr, indices_ptr,
                score_input_ptrs, score_output_ptrs, extra_policies_arr,
                table_key_slots_,
                evict_counter_, evicted_keys_, evicted_scores_ptrs,
                evicted_indices_, evicted_table_ids_ptr, counter_ptr, stream,
                ovf_table, ovf_bucket_sizes_, ovf_counter_ptr,
                ovf_output_offsets_ptr);
          } else {
            launch_table_insert_and_evict_kernel<
                Table, PolicyTypeV, false, compact_tile_size.value,
                enable_overflow.value>(
                table, table_bucket_offsets_ptr, bucket_sizes_, num_total,
                keys_, table_ids_ptr, insert_results_ptr, indices_ptr,
                score_input_ptrs, nullptr, extra_policies_arr, table_key_slots_, evict_counter_,
                evicted_keys_, evicted_scores_ptrs, evicted_indices_,
                evicted_table_ids_ptr, counter_ptr, stream, ovf_table,
                ovf_bucket_sizes_, ovf_counter_ptr, ovf_output_offsets_ptr);
          }
        };

        struct CT32 {
          static constexpr int value = 32;
        };
        struct CT1 {
          static constexpr int value = 1;
        };
        struct OVTrue {
          static constexpr bool value = true;
        };
        struct OVFalse {
          static constexpr bool value = false;
        };

        if (num_total % 32 == 0) {
          if (use_overflow) {
            launch(CT32{}, OVTrue{});
          } else {
            launch(CT32{}, OVFalse{});
          }
        } else {
          if (use_overflow) {
            launch(CT1{}, OVTrue{});
          } else {
            launch(CT1{}, OVFalse{});
          }
        }
      });
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
table_insert_and_evict(at::Tensor table_storage,
                       at::Tensor table_bucket_offsets, int64_t bucket_capacity,
                       at::Tensor bucket_sizes, at::Tensor keys,
                       at::Tensor table_ids,
                       std::optional<at::Tensor> score_input,
                       ScorePolicyType policy_type, at::Tensor counter,
                       std::optional<at::Tensor> insert_results,
                       std::optional<at::Tensor> score_output,
                       std::optional<at::Tensor> ovf_storage,
                       int64_t ovf_bucket_capacity,
                       std::optional<at::Tensor> ovf_bucket_sizes,
                       std::optional<at::Tensor> ovf_counter,
                       std::optional<at::Tensor> ovf_output_offsets,
                        int num_scores,
                        std::vector<std::optional<at::Tensor>> score_inputs,
                        std::vector<std::optional<at::Tensor>> score_outputs,
                        std::vector<ScorePolicyType> extra_policies) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    at::Tensor indices = torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor num_evicted =
        torch::zeros({1}, keys.options().dtype(torch::kInt64));
    at::Tensor evicted_keys =
        torch::empty({0}, keys.options().dtype(keys.scalar_type()));
    at::Tensor evicted_indices =
        torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor evicted_scores =
        torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor evicted_table_ids =
        torch::empty({0}, keys.options().dtype(torch::kLong));
    return std::make_tuple(indices, num_evicted, evicted_keys, evicted_indices,
                           evicted_scores, evicted_table_ids);
  }

  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor num_evicted =
      torch::zeros({1}, keys.options().dtype(torch::kInt64));
  at::Tensor evicted_keys =
      torch::empty({num_total}, keys.options().dtype(keys.scalar_type()));
  at::Tensor evicted_indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor evicted_scores =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor evicted_table_ids =
      torch::empty({num_total}, keys.options().dtype(torch::kLong));

  table_insert_and_evict_single_score(
      table_storage, table_bucket_offsets, bucket_capacity, bucket_sizes, keys,
      table_ids, score_input, policy_type, indices, insert_results,
      score_output, num_evicted, evicted_keys, evicted_indices, evicted_scores,
      evicted_table_ids, counter, num_scores, score_inputs, score_outputs,
      extra_policies, ovf_storage, ovf_bucket_capacity, ovf_bucket_sizes,
      ovf_counter, ovf_output_offsets);

  return std::make_tuple(indices, num_evicted, evicted_keys, evicted_indices,
                         evicted_scores, evicted_table_ids);
}

void table_update_counter_with_layout(
    at::Tensor counter, at::Tensor slot_indices, int32_t delta,
    at::Tensor table_bucket_offsets, int64_t bucket_capacity,
    int64_t main_capacity, int64_t num_tables,
    c10::optional<at::Tensor> table_ids,
    c10::optional<at::Tensor> overflow_output_offsets,
    int64_t overflow_bucket_capacity) {

  int64_t n = slot_indices.size(0);
  if (n == 0)
    return;

  int64_t total_capacity = counter.numel();
  auto counter_ptr = counter.data_ptr<int32_t>();
  auto slot_indices_ptr = slot_indices.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  int64_t const *table_ids_ptr = (table_ids.has_value() && table_ids->defined())
                                     ? table_ids->data_ptr<int64_t>()
                                     : nullptr;
  int64_t const *overflow_output_offsets_ptr =
      (overflow_output_offsets.has_value() &&
       overflow_output_offsets->defined())
          ? overflow_output_offsets->data_ptr<int64_t>()
          : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int BLOCK_SIZE = 256;
  update_counter_with_layout_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                      BLOCK_SIZE, 0, stream>>>(
      counter_ptr, total_capacity, slot_indices_ptr, n, delta, table_ids_ptr,
      table_bucket_offsets_ptr, bucket_capacity, main_capacity,
      overflow_output_offsets_ptr, overflow_bucket_capacity, num_tables);

  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
