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

static void table_lookup_single_score(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
    std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
    at::Tensor score_output, at::Tensor founds, at::Tensor indices,
    int num_scores,
    const std::vector<std::optional<at::Tensor>> &score_inputs,
    const std::vector<at::Tensor> &score_outputs,
    const std::vector<ScorePolicyType> &extra_policies,
    std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
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

  auto score_output_ptr = score_output.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<IndexType>();
  auto founds_ptr = founds.data_ptr<bool>();
  auto table_ids_ptr = table_ids.data_ptr<int64_t>();
  auto table_bucket_offsets_ptr = table_bucket_offsets.data_ptr<int64_t>();
  int64_t *ovf_output_offsets_ptr =
      ovf_output_offsets.has_value() && ovf_output_offsets.value().defined()
          ? ovf_output_offsets.value().data_ptr<int64_t>()
          : nullptr;

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    auto keys_ptr = get_pointer<KeyType>(keys);

    int64_t total_size =
        sizeof(KeyType) + sizeof(DigestType) + num_scores * sizeof(ScoreType);
    int64_t bucket_bytes = bucket_capacity * total_size;
    int64_t num_buckets =
        table_storage.numel() * table_storage.element_size() / bucket_bytes;

    // Build score pointer arrays for kernel
    ScoreType *score_input_ptrs[kMaxNumScores] = {};
    int64_t *score_output_ptrs[kMaxNumScores] = {};
    ScorePolicyType extra_policies_arr[kMaxNumScores] = {
        ScorePolicyType::Const};
    score_input_ptrs[0] = score_input_ptr;
    score_output_ptrs[0] = score_output_ptr;
    for (int s = 0; s < num_scores - 1 && s < (int)extra_policies.size(); ++s)
      extra_policies_arr[s] = extra_policies[s];
    std::vector<at::Tensor> score_input_tensors(kMaxNumScores);
    for (int s = 1; s < num_scores; ++s) {
      if (s < (int)score_inputs.size() && score_inputs[s].has_value()) {
        at::Tensor in = score_inputs[s].value();
        if (in.scalar_type() == torch::kUInt64) {
          score_input_ptrs[s] = in.data_ptr<ScoreType>();
        } else {
          score_input_tensors[s] = in.view(torch::kUInt64);
          score_input_ptrs[s] = score_input_tensors[s].data_ptr<ScoreType>();
        }
      }
      if (s < (int)score_outputs.size() && score_outputs[s].defined()) {
        score_output_ptrs[s] = score_outputs[s].data_ptr<int64_t>();
      }
    }

    bool use_overflow =
        ovf_storage.has_value() && ovf_storage.value().defined();

    DISPATCH_NUM_SCORES(num_scores, NumScoresV, [&] {
      using Bucket = LinearBucket<KeyType, NumScoresV>;
      using Table = LinearBucketTable<Bucket>;

      auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                         num_buckets, bucket_capacity);

      Table ovf_table;
      if (use_overflow) {
        at::Tensor ovf = ovf_storage.value();
        int64_t ovf_bucket_bytes =
            bucket_capacity * total_size; // same total_size
        int64_t ovf_num_buckets =
            ovf.numel() * ovf.element_size() / ovf_bucket_bytes;
        ovf_table = Table(reinterpret_cast<uint8_t *>(ovf.data_ptr()),
                          ovf_num_buckets, bucket_capacity);
      }

      DISPATCH_SCORE_POLICY(policy_type, PolicyTypeV, [&] {
        if (use_overflow) {
          table_lookup_kernel<Table, 1, PolicyTypeV, true>
              <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                 stream>>>(table, table_bucket_offsets_ptr, num_total, keys_ptr,
                           table_ids_ptr, founds_ptr, indices_ptr,
                           score_input_ptrs, score_output_ptrs,
                           extra_policies_arr, ovf_table,
                           ovf_output_offsets_ptr);
        } else {
          table_lookup_kernel<Table, 1, PolicyTypeV, false>
              <<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                 stream>>>(table, table_bucket_offsets_ptr, num_total, keys_ptr,
                           table_ids_ptr, founds_ptr, indices_ptr,
                           score_input_ptrs, score_output_ptrs,
                           extra_policies_arr, Table(),
                           nullptr);
        }
      });
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
             std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
             std::optional<at::Tensor> ovf_storage, int64_t ovf_bucket_capacity,
             std::optional<at::Tensor> ovf_output_offsets, int num_scores,
             std::vector<std::optional<at::Tensor>> score_inputs,
             std::vector<ScorePolicyType> extra_policies) {

  int64_t num_total = keys.size(0);
  if (num_total == 0) {
    at::Tensor score_output =
        torch::empty({0}, keys.options().dtype(torch::kInt64));
    at::Tensor founds = torch::empty({0}, keys.options().dtype(torch::kBool));
    at::Tensor indices = torch::empty({0}, keys.options().dtype(torch::kInt64));
    return std::make_tuple(score_output, founds, indices);
  }

  at::Tensor score_output0 =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));
  at::Tensor founds =
      torch::empty({num_total}, keys.options().dtype(torch::kBool));
  at::Tensor indices =
      torch::empty({num_total}, keys.options().dtype(torch::kInt64));

  // Allocate extra score output tensors
  std::vector<at::Tensor> score_outputs;
  for (int s = 1; s < num_scores; ++s) {
    score_outputs.push_back(
        torch::empty({num_total}, keys.options().dtype(torch::kInt64)));
  }

  table_lookup_single_score(
      table_storage, table_bucket_offsets, bucket_capacity, keys, table_ids,
      score_input, policy_type, score_output0, founds, indices, num_scores,
      score_inputs, score_outputs, extra_policies, ovf_storage,
      ovf_bucket_capacity, ovf_output_offsets);

  // TODO: return multi-score outputs when needed by the caller.
  return std::make_tuple(score_output0, founds, indices);
}

} // namespace dyn_emb
