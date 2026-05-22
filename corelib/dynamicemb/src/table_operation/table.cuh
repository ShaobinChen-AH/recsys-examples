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

#include "../check.h"
#include "torch_utils.h"
#include "types.cuh"

#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda/std/tuple>

#ifdef DEMB_USE_PYBIND11
#include <torch/extension.h>
#endif
#include <torch/torch.h>

#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#define DISPATCH_KEY_TYPE(DATA_TYPE, HINT, ...)                                \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::Int64, int64_t, HINT, __VA_ARGS__)          \
    CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
  default:                                                                     \
    throw std::runtime_error("Not supported key type.");                       \
  }

#define DISPATCH_SCORE_TYPE(DATA_TYPE, HINT, ...)                              \
  switch (DATA_TYPE) {                                                         \
    CASE_TYPE_USING_HINT(DataType::UInt64, uint64_t, HINT, __VA_ARGS__)        \
    CASE_TYPE_USING_HINT(DataType::UInt32, uint32_t, HINT, __VA_ARGS__)        \
  default:                                                                     \
    throw std::runtime_error("Not supported score type.");                     \
  }

#define DISPATCH_SCORE_POLICY(SCORE_POLICY, HINT, ...)                         \
  switch (SCORE_POLICY) {                                                      \
    CASE_ENUM_USING_HINT(ScorePolicyType::Const, HINT, __VA_ARGS__)            \
    CASE_ENUM_USING_HINT(ScorePolicyType::Assign, HINT, __VA_ARGS__)           \
    CASE_ENUM_USING_HINT(ScorePolicyType::Accumulate, HINT, __VA_ARGS__)       \
    CASE_ENUM_USING_HINT(ScorePolicyType::GlobalTimer, HINT, __VA_ARGS__)      \
  default:                                                                     \
    throw std::runtime_error("Not supported score policy.");                   \
  }

#define DISPATCH_NUM_SCORES(NUM_SCORES, HINT, ...)                             \
  switch (NUM_SCORES) {                                                        \
    case 1: {                                                                  \
      constexpr int HINT = 1;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 2: {                                                                  \
      constexpr int HINT = 2;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 3: {                                                                  \
      constexpr int HINT = 3;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 4: {                                                                  \
      constexpr int HINT = 4;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 5: {                                                                  \
      constexpr int HINT = 5;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 6: {                                                                  \
      constexpr int HINT = 6;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 7: {                                                                  \
      constexpr int HINT = 7;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 8: {                                                                  \
      constexpr int HINT = 8;                                                  \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
  default:                                                                     \
    throw std::runtime_error("Unsupported num_scores (1-8): " +                \
                             std::to_string(NUM_SCORES));                      \
  }

namespace dyn_emb {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
table_lookup(at::Tensor table_storage, at::Tensor table_bucket_offsets,
             int64_t bucket_capacity, at::Tensor keys, at::Tensor table_ids,
             std::optional<at::Tensor> score_input, ScorePolicyType policy_type,
             std::optional<at::Tensor> ovf_storage = std::nullopt,
             int64_t ovf_bucket_capacity = 0,
             std::optional<at::Tensor> ovf_output_offsets = std::nullopt,
             int num_scores = 1,
             std::vector<std::optional<at::Tensor>> score_inputs = {},
             std::vector<ScorePolicyType> extra_policies = {});

at::Tensor table_insert(at::Tensor table_storage,
                        at::Tensor table_bucket_offsets,
                        int64_t bucket_capacity, at::Tensor bucket_sizes,
                        at::Tensor keys, at::Tensor table_ids,
                        std::optional<at::Tensor> score_input,
                        ScorePolicyType policy_type, at::Tensor counter,
                        std::optional<at::Tensor> insert_results = std::nullopt,
                        std::optional<at::Tensor> score_output = std::nullopt,
                        int num_scores = 1,
                        std::vector<std::optional<at::Tensor>> score_inputs = {},
                         std::vector<std::optional<at::Tensor>> score_outputs =
                             {},
                         std::vector<ScorePolicyType> extra_policies = {});

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
table_insert_and_evict(
    at::Tensor table_storage, at::Tensor table_bucket_offsets,
    int64_t bucket_capacity, at::Tensor bucket_sizes, at::Tensor keys,
    at::Tensor table_ids, std::optional<at::Tensor> score_input,
    ScorePolicyType policy_type, at::Tensor counter,
    std::optional<at::Tensor> insert_results = std::nullopt,
    std::optional<at::Tensor> score_output = std::nullopt,
    std::optional<at::Tensor> ovf_storage = std::nullopt,
    int64_t ovf_bucket_capacity = 0,
    std::optional<at::Tensor> ovf_bucket_sizes = std::nullopt,
    std::optional<at::Tensor> ovf_counter = std::nullopt,
    std::optional<at::Tensor> ovf_output_offsets = std::nullopt,
    int num_scores = 1,
     std::vector<std::optional<at::Tensor>> score_inputs = {},
     std::vector<std::optional<at::Tensor>> score_outputs = {},
     std::vector<ScorePolicyType> extra_policies = {});

void table_erase(at::Tensor table_storage, at::Tensor table_bucket_offsets,
                 int64_t bucket_capacity, at::Tensor bucket_sizes,
                 at::Tensor keys, at::Tensor table_ids,
                 std::optional<at::Tensor> indices, int num_scores = 1);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
table_export_batch(at::Tensor table_storage, int64_t bucket_capacity,
                   int64_t batch, int64_t offset, torch::Dtype key_dtype,
                   std::optional<ScoreType> threshold = std::nullopt,
                   int64_t table_begin = 0, int num_scores = 1);

at::Tensor table_count_matched(at::Tensor table_storage, torch::Dtype key_dtype,
                               int64_t bucket_capacity, ScoreType threshold,
                               int64_t begin = -1, int64_t end = -1,
                               int num_scores = 1);

std::vector<at::Tensor> table_partition(at::Tensor storage,
                                        std::vector<torch::Dtype> dtypes,
                                        int64_t bucket_capacity,
                                        int64_t num_buckets);

std::vector<at::Tensor> tensor_partition(at::Tensor input,
                                         std::vector<int64_t> byte_range,
                                         std::vector<torch::Dtype> dtypes);

std::vector<at::Tensor> bucketize_keys(at::Tensor keys, at::Tensor table_ids,
                                       at::Tensor table_bucket_offsets,
                                       int64_t num_buckets,
                                       int64_t bucket_capacity);

void table_update_counter_with_layout(
    at::Tensor counter, at::Tensor slot_indices, int32_t delta,
    at::Tensor table_bucket_offsets, int64_t bucket_capacity,
    int64_t main_capacity, int64_t num_tables,
    c10::optional<at::Tensor> table_ids,
    c10::optional<at::Tensor> overflow_output_offsets,
    int64_t overflow_bucket_capacity);

at::Tensor no_eviction_assign_scores(at::Tensor no_eviction_next_index_dev,
                                     at::Tensor table_ids);

} // namespace dyn_emb
