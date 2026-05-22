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

#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace dyn_emb {

using ScoreType = uint64_t;
// Maximum number of scores per slot supported at compile time.
// When more scores are needed, increase this constant and recompile.
static constexpr int kMaxNumScores = 8;

enum class ScorePolicyType : uint8_t {
  Const = 0,
  Assign = 1,
  Accumulate = 2,
  GlobalTimer = 3,
};

template <ScorePolicyType PolicyType> struct ScorePolicy {

  static __device__ __forceinline__ ScoreType get(ScoreType *scores,
                                                  int64_t index) {
    if constexpr (PolicyType == ScorePolicyType::Const) {
      return ScoreType();
    } else if constexpr (PolicyType == ScorePolicyType::GlobalTimer) {
      ScoreType score;
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(score));
      return score;
    } else {
      return scores[index];
    }
  }

  static __device__ __forceinline__ ScoreType
  score_for_compare(ScoreType score) {
    return UINT64_MAX;
  }

  // Updates table slot and returns the output score.
  static __device__ __forceinline__ ScoreType update(ScoreType *table_score,
                                                     ScoreType score) {
    if constexpr (PolicyType == ScorePolicyType::Const) {
      return *table_score;
    } else if constexpr (PolicyType == ScorePolicyType::Accumulate) {
      score += *table_score;
      *table_score = score;
      return score;
    } else {
      *table_score = score;
      return score;
    }
  }
};

// Compile-time policy array: dispatches to the correct ScorePolicy at each
// score index.  PolicyType<0>..PolicyType<N-1> must be the actual policies.
template <ScorePolicyType... Policies> struct ScorePolicyArray {
  static constexpr int kNumPolicies = sizeof...(Policies);

  // Get the input score for index i from score_inputs[s] (the s-th score
  // array).  Const => return 0; GlobalTimer => return GPU timer; otherwise
  // return the value at score_inputs[s][i].
  template <int S>
  static __device__ __forceinline__ ScoreType
  get(ScoreType *const *score_inputs, int64_t i) {
    constexpr ScorePolicyType p = get_policy<S>();
    return ScorePolicy<p>::get(score_inputs[S], i);
  }

  // Update table_scores[s] with the given score value.
  template <int S>
  static __device__ __forceinline__ ScoreType update(ScoreType *table_scores,
                                                     ScoreType score) {
    constexpr ScorePolicyType p = get_policy<S>();
    return ScorePolicy<p>::update(table_scores, score);
  }

  // Returns the policy at compile-time index S.
  template <int S> static constexpr ScorePolicyType get_policy() {
    constexpr ScorePolicyType arr[] = {Policies...};
    return arr[S];
  }
};

// Helper to build ScorePolicyArray from a runtime policy array via switch
// dispatch.  The caller matches num_scores to the correct instantiation.
template <ScorePolicyType P0, ScorePolicyType P1 = ScorePolicyType::Const,
          ScorePolicyType P2 = ScorePolicyType::Const,
          ScorePolicyType P3 = ScorePolicyType::Const,
          ScorePolicyType P4 = ScorePolicyType::Const,
          ScorePolicyType P5 = ScorePolicyType::Const,
          ScorePolicyType P6 = ScorePolicyType::Const,
          ScorePolicyType P7 = ScorePolicyType::Const>
struct MakeScorePolicyArray {
  using type = ScorePolicyArray<P0, P1, P2, P3, P4, P5, P6, P7>;
};

// Convenience: given a runtime num_scores and an array of ScorePolicyType,
// returns the effective primary (eviction) score policy.  The primary score is
// always at index 0.
inline ScorePolicyType primary_policy(int num_scores,
                                      const ScorePolicyType *policies) {
  (void)num_scores;
  return policies[0];
}

} // namespace dyn_emb
