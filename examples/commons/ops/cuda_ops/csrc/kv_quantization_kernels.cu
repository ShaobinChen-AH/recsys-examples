/******************************************************************************
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
#
# CUDA Kernels for KV Cache Quantization with Random Rotation + LVQ
# 
******************************************************************************/

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Helper: Convert between bfloat16/half and float
__device__ __forceinline__ float to_float(nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float to_float(nv_half x) {
    return __half2float(x);
}

__device__ __forceinline__ nv_bfloat16 to_bfloat16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ nv_half to_half(float x) {
    return __float2half(x);
}

// ============================================
// 4-bit Quantization/Packing Kernels
// ============================================

// Pack 4-bit indices into uint8 (2 values per byte)
// input: [N] uint8 values in range [0, 15]
// output: [N/2] packed uint8
template <typename IdType>
__global__ void Pack4BitKernel(
    const IdType* __restrict__ input,
    uint8_t* __restrict__ output,
    const int32_t n) {
    
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t pack_idx = idx * 2;
    
    if (pack_idx + 1 < n) {
        uint8_t high = static_cast<uint8_t>(input[pack_idx]) & 0x0F;
        uint8_t low = static_cast<uint8_t>(input[pack_idx + 1]) & 0x0F;
        output[idx] = (high << 4) | low;
    } else if (pack_idx < n) {
        // Last element if n is odd
        output[idx] = (static_cast<uint8_t>(input[pack_idx]) & 0x0F) << 4;
    }
}

// Unpack uint8 into 4-bit indices
// input: [N/2] packed uint8
// output: [N] uint8 values in range [0, 15]
template <typename IdType>
__global__ void Unpack4BitKernel(
    const uint8_t* __restrict__ input,
    IdType* __restrict__ output,
    const int32_t n) {
    
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t unpack_idx = idx * 2;
    
    if (unpack_idx + 1 < n) {
        uint8_t packed = input[idx];
        output[unpack_idx] = static_cast<IdType>((packed >> 4) & 0x0F);
        output[unpack_idx + 1] = static_cast<IdType>(packed & 0x0F);
    } else if (unpack_idx < n) {
        uint8_t packed = input[idx];
        output[unpack_idx] = static_cast<IdType>((packed >> 4) & 0x0F);
    }
}

// ============================================
// LVQ Quantization Kernel (Per-head)
// ============================================

// Quantize KV data to 4-bit with per-head scale/zero-point
// input: [num_pages, 2, page_size, num_heads, head_dim]
// output_indices: [num_pages, 2, page_size, num_heads, head_dim/2] packed 4-bit
// output_scales: [num_pages, 2, num_heads]
// output_zero_points: [num_pages, 2, num_heads]
template <typename DType>
__global__ void LVQQuantizeKernel(
    const DType* __restrict__ input,
    uint8_t* __restrict__ output_indices,
    float* __restrict__ output_scales,
    float* __restrict__ output_zero_points,
    const int32_t num_pages,
    const int32_t page_size,
    const int32_t num_heads,
    const int32_t head_dim) {
    
    // Each block processes one page, one k/v position (0 or 1), and one head
    const int32_t page_idx = blockIdx.x;
    const int32_t kv_idx = blockIdx.y;  // 0 for K, 1 for V
    const int32_t head_idx = blockIdx.z;
    
    if (page_idx >= num_pages || kv_idx >= 2 || head_idx >= num_heads) return;
    
    // Calculate strides
    const int32_t page_stride = 2 * page_size * num_heads * head_dim;
    const int32_t kv_stride = page_size * num_heads * head_dim;
    const int32_t seq_stride = num_heads * head_dim;
    const int32_t head_stride = head_dim;
    
    // Shared memory for finding min/max
    __shared__ float s_min;
    __shared__ float s_max;
    
    // Step 1: Find min and max for this head
    float local_min = 1e10f;
    float local_max = -1e10f;
    
    const int32_t base_offset = page_idx * page_stride + kv_idx * kv_stride + head_idx * head_stride;
    
    for (int32_t seq_idx = threadIdx.x; seq_idx < page_size; seq_idx += blockDim.x) {
        for (int32_t dim_idx = 0; dim_idx < head_dim; ++dim_idx) {
            const int32_t offset = base_offset + seq_idx * seq_stride + dim_idx;
            float val = to_float(input[offset]);
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Warp reduction for min/max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_min = fminf(local_min, __shfl_xor_sync(0xFFFFFFFF, local_min, offset));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
    }
    
    if (threadIdx.x == 0) {
        s_min = local_min;
        s_max = local_max;
    }
    __syncthreads();
    
    // Step 2: Compute scale and zero point
    const float min_val = s_min;
    const float max_val = s_max;
    const float scale = (max_val - min_val) / 15.0f;  // 4-bit: 16 levels
    const float zero_point = min_val;
    
    // Store scale and zero point
    if (threadIdx.x == 0) {
        const int32_t meta_offset = page_idx * 2 * num_heads + kv_idx * num_heads + head_idx;
        output_scales[meta_offset] = scale;
        output_zero_points[meta_offset] = zero_point;
    }
    
    // Step 3: Quantize values to 4-bit and pack
    const int32_t indices_page_stride = 2 * page_size * num_heads * ((head_dim + 1) / 2);
    const int32_t indices_kv_stride = page_size * num_heads * ((head_dim + 1) / 2);
    const int32_t indices_seq_stride = num_heads * ((head_dim + 1) / 2);
    const int32_t indices_head_stride = (head_dim + 1) / 2;
    
    for (int32_t seq_idx = 0; seq_idx < page_size; ++seq_idx) {
        for (int32_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
            const int32_t input_offset = base_offset + seq_idx * seq_stride + dim_idx;
            float val = to_float(input[input_offset]);
            
            // Quantize: (val - zero_point) / scale
            int32_t idx = static_cast<int32_t>(roundf((val - zero_point) / scale));
            idx = max(0, min(15, idx));  // Clamp to [0, 15]
            
            // Pack two 4-bit values into one byte
            const int32_t pack_idx = dim_idx / 2;
            const int32_t is_high = dim_idx % 2;  // 0 for high nibble, 1 for low nibble
            
            const int32_t output_offset = page_idx * indices_page_stride 
                                        + kv_idx * indices_kv_stride 
                                        + seq_idx * indices_seq_stride 
                                        + head_idx * indices_head_stride 
                                        + pack_idx;
            
            if (is_high == 0) {
                // High nibble (first 4 bits)
                atomicOr((unsigned int*)&output_indices[output_offset], 
                        static_cast<unsigned int>(idx << 4));
            } else {
                // Low nibble (last 4 bits)
                atomicOr((unsigned int*)&output_indices[output_offset], 
                        static_cast<unsigned int>(idx & 0x0F));
            }
        }
    }
}

// ============================================
// LVQ Dequantization Kernel (Per-head)
// ============================================

// Dequantize 4-bit indices back to original precision
// input_indices: [num_pages, 2, page_size, num_heads, head_dim/2] packed 4-bit
// input_scales: [num_pages, 2, num_heads]
// input_zero_points: [num_pages, 2, num_heads]
// output: [num_pages, 2, page_size, num_heads, head_dim]
template <typename DType>
__global__ void LVQDequantizeKernel(
    const uint8_t* __restrict__ input_indices,
    const float* __restrict__ input_scales,
    const float* __restrict__ input_zero_points,
    DType* __restrict__ output,
    const int32_t num_pages,
    const int32_t page_size,
    const int32_t num_heads,
    const int32_t head_dim) {
    
    // Each block processes one page, one k/v position, and one head
    const int32_t page_idx = blockIdx.x;
    const int32_t kv_idx = blockIdx.y;
    const int32_t head_idx = blockIdx.z;
    
    if (page_idx >= num_pages || kv_idx >= 2 || head_idx >= num_heads) return;
    
    // Load scale and zero point
    const int32_t meta_offset = page_idx * 2 * num_heads + kv_idx * num_heads + head_idx;
    const float scale = input_scales[meta_offset];
    const float zero_point = input_zero_points[meta_offset];
    
    // Strides
    const int32_t indices_page_stride = 2 * page_size * num_heads * ((head_dim + 1) / 2);
    const int32_t indices_kv_stride = page_size * num_heads * ((head_dim + 1) / 2);
    const int32_t indices_seq_stride = num_heads * ((head_dim + 1) / 2);
    const int32_t indices_head_stride = (head_dim + 1) / 2;
    
    const int32_t page_stride = 2 * page_size * num_heads * head_dim;
    const int32_t kv_stride = page_size * num_heads * head_dim;
    const int32_t seq_stride = num_heads * head_dim;
    const int32_t head_stride = head_dim;
    
    // Dequantize
    for (int32_t seq_idx = 0; seq_idx < page_size; ++seq_idx) {
        for (int32_t dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += blockDim.x) {
            // Unpack 4-bit value
            const int32_t pack_idx = dim_idx / 2;
            const int32_t is_high = dim_idx % 2;
            
            const int32_t input_offset = page_idx * indices_page_stride 
                                       + kv_idx * indices_kv_stride 
                                       + seq_idx * indices_seq_stride 
                                       + head_idx * indices_head_stride 
                                       + pack_idx;
            
            uint8_t packed = input_indices[input_offset];
            int32_t idx;
            if (is_high == 0) {
                idx = (packed >> 4) & 0x0F;
            } else {
                idx = packed & 0x0F;
            }
            
            // Dequantize: idx * scale + zero_point
            float val = static_cast<float>(idx) * scale + zero_point;
            
            // Store
            const int32_t output_offset = page_idx * page_stride 
                                        + kv_idx * kv_stride 
                                        + seq_idx * seq_stride 
                                        + head_idx * head_stride 
                                        + dim_idx;
            
            if constexpr (std::is_same_v<DType, nv_bfloat16>) {
                output[output_offset] = to_bfloat16(val);
            } else {
                output[output_offset] = to_half(val);
            }
        }
    }
}

__device__ __forceinline__ nv_bfloat16 float_to_dtype(float val, nv_bfloat16) {
    return to_bfloat16(val);
}

__device__ __forceinline__ nv_half float_to_dtype(float val, nv_half) {
    return to_half(val);
}


// ============================================
// Random Rotation Kernels
// ============================================

// Apply random rotation matrix to KV data
// Using group-wise rotation for efficiency
template <typename DType>
__global__ void RandomRotationKernel(
    const DType* __restrict__ input,
    DType* __restrict__ output,
    const float* __restrict__ rotation_matrix,  // [group_size, group_size]
    const int32_t num_pages,
    const int32_t page_size,
    const int32_t num_heads,
    const int32_t head_dim,
    const int32_t group_size,
    const bool inverse) {  // true for inverse rotation (transpose)
    
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t total_elements = num_pages * 2 * page_size * num_heads * head_dim;
    
    if (tid >= total_elements) return;
    
    // Decode indices
    const int32_t dim_idx = tid % head_dim;
    const int32_t head_idx = (tid / head_dim) % num_heads;
    const int32_t seq_idx = (tid / (head_dim * num_heads)) % page_size;
    const int32_t kv_idx = (tid / (head_dim * num_heads * page_size)) % 2;
    const int32_t page_idx = tid / (head_dim * num_heads * page_size * 2);
    
    // Calculate group
    const int32_t group_idx = dim_idx / group_size;
    const int32_t within_group_idx = dim_idx % group_size;
    
    // Apply rotation: output[dim] = sum_i input[i] * R[i, dim] (or R^T)
    float result = 0.0f;
    const int32_t base_offset = page_idx * 2 * page_size * num_heads * head_dim 
                              + kv_idx * page_size * num_heads * head_dim 
                              + seq_idx * num_heads * head_dim 
                              + head_idx * head_dim 
                              + group_idx * group_size;
    
    for (int32_t i = 0; i < group_size; ++i) {
        const int32_t matrix_idx = inverse ? (within_group_idx * group_size + i) 
                                           : (i * group_size + within_group_idx);
        float rot_val = rotation_matrix[group_idx * group_size * group_size + matrix_idx];
        float input_val = to_float(input[base_offset + i]);
        result += input_val * rot_val;
    }
    
    DType dummy;
    output[tid] = float_to_dtype(result, dummy);
}

// ============================================
// C++ API Wrappers
// ============================================

template <typename DType, typename IdType>
cudaError_t QuantizeKVCache4Bit(
    const DType* input,
    uint8_t* output_indices,
    float* output_scales,
    float* output_zero_points,
    int32_t num_pages,
    int32_t page_size,
    int32_t num_heads,
    int32_t head_dim,
    cudaStream_t stream) {
    
    dim3 grid(num_pages, 2, num_heads);
    dim3 block(min(256, page_size * head_dim));
    
    LVQQuantizeKernel<DType><<<grid, block, 0, stream>>>(
        input, output_indices, output_scales, output_zero_points,
        num_pages, page_size, num_heads, head_dim);
    
    return cudaGetLastError();
}

template <typename DType>
cudaError_t DequantizeKVCache4Bit(
    const uint8_t* input_indices,
    const float* input_scales,
    const float* input_zero_points,
    DType* output,
    int32_t num_pages,
    int32_t page_size,
    int32_t num_heads,
    int32_t head_dim,
    cudaStream_t stream) {
    
    dim3 grid(num_pages, 2, num_heads);
    dim3 block(min(256, page_size * head_dim));
    
    LVQDequantizeKernel<DType><<<grid, block, 0, stream>>>(
        input_indices, input_scales, input_zero_points, output,
        num_pages, page_size, num_heads, head_dim);
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t QuantizeKVCache4Bit<nv_bfloat16, int32_t>(
    const nv_bfloat16*, uint8_t*, float*, float*, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template cudaError_t QuantizeKVCache4Bit<nv_half, int32_t>(
    const nv_half*, uint8_t*, float*, float*, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template cudaError_t DequantizeKVCache4Bit<nv_bfloat16>(
    const uint8_t*, const float*, const float*, nv_bfloat16*, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

template cudaError_t DequantizeKVCache4Bit<nv_half>(
    const uint8_t*, const float*, const float*, nv_half*, int32_t, int32_t, int32_t, int32_t, cudaStream_t);

