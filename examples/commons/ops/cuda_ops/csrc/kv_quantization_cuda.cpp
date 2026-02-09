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
# PyBind11 interface for KV Cache Quantization CUDA Kernels
# 
******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

// Forward declarations of CUDA functions
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
    cudaStream_t stream);

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
    cudaStream_t stream);

void quantize_kv_cache_4bit(
    at::Tensor input,
    at::Tensor output_indices,
    at::Tensor output_scales,
    at::Tensor output_zero_points) {
    
    // Validate inputs
    TORCH_CHECK(input.ndimension() == 5, "input must be 5D: [num_pages, 2, page_size, num_heads, head_dim]");
    TORCH_CHECK(output_indices.ndimension() == 5, "output_indices must be 5D");
    TORCH_CHECK(output_scales.ndimension() == 3, "output_scales must be 3D: [num_pages, 2, num_heads]");
    TORCH_CHECK(output_zero_points.ndimension() == 3, "output_zero_points must be 3D: [num_pages, 2, num_heads]");
    
    int32_t num_pages = input.size(0);
    int32_t page_size = input.size(2);
    int32_t num_heads = input.size(3);
    int32_t head_dim = input.size(4);
    
    auto device = input.device();
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream();
    
    auto input_dtype = input.scalar_type();
    
    cudaError_t status;
    switch (input_dtype) {
        case at::ScalarType::BFloat16:
            status = QuantizeKVCache4Bit<nv_bfloat16, int32_t>(
                static_cast<const nv_bfloat16*>(input.data_ptr()),
                static_cast<uint8_t*>(output_indices.data_ptr()),
                static_cast<float*>(output_scales.data_ptr()),
                static_cast<float*>(output_zero_points.data_ptr()),
                num_pages, page_size, num_heads, head_dim, stream);
            break;
        case at::ScalarType::Half:
            status = QuantizeKVCache4Bit<nv_half, int32_t>(
                static_cast<const nv_half*>(input.data_ptr()),
                static_cast<uint8_t*>(output_indices.data_ptr()),
                static_cast<float*>(output_scales.data_ptr()),
                static_cast<float*>(output_zero_points.data_ptr()),
                num_pages, page_size, num_heads, head_dim, stream);
            break;
        default:
            TORCH_CHECK(false, "quantize_kv_cache_4bit: unsupported dtype ", input_dtype);
    }
    
    TORCH_CHECK(status == cudaSuccess,
                "quantize_kv_cache_4bit failed: ", cudaGetErrorString(status));
}

void dequantize_kv_cache_4bit(
    at::Tensor input_indices,
    at::Tensor input_scales,
    at::Tensor input_zero_points,
    at::Tensor output) {
    
    // Validate inputs
    TORCH_CHECK(input_indices.ndimension() == 5, "input_indices must be 5D");
    TORCH_CHECK(input_scales.ndimension() == 3, "input_scales must be 3D: [num_pages, 2, num_heads]");
    TORCH_CHECK(input_zero_points.ndimension() == 3, "input_zero_points must be 3D: [num_pages, 2, num_heads]");
    TORCH_CHECK(output.ndimension() == 5, "output must be 5D: [num_pages, 2, page_size, num_heads, head_dim]");
    
    int32_t num_pages = output.size(0);
    int32_t page_size = output.size(2);
    int32_t num_heads = output.size(3);
    int32_t head_dim = output.size(4);
    
    auto device = output.device();
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream();
    
    auto output_dtype = output.scalar_type();
    
    cudaError_t status;
    switch (output_dtype) {
        case at::ScalarType::BFloat16:
            status = DequantizeKVCache4Bit<nv_bfloat16>(
                static_cast<const uint8_t*>(input_indices.data_ptr()),
                static_cast<const float*>(input_scales.data_ptr()),
                static_cast<const float*>(input_zero_points.data_ptr()),
                static_cast<nv_bfloat16*>(output.data_ptr()),
                num_pages, page_size, num_heads, head_dim, stream);
            break;
        case at::ScalarType::Half:
            status = DequantizeKVCache4Bit<nv_half>(
                static_cast<const uint8_t*>(input_indices.data_ptr()),
                static_cast<const float*>(input_scales.data_ptr()),
                static_cast<const float*>(input_zero_points.data_ptr()),
                static_cast<nv_half*>(output.data_ptr()),
                num_pages, page_size, num_heads, head_dim, stream);
            break;
        default:
            TORCH_CHECK(false, "dequantize_kv_cache_4bit: unsupported dtype ", output_dtype);
    }
    
    TORCH_CHECK(status == cudaSuccess,
                "dequantize_kv_cache_4bit failed: ", cudaGetErrorString(status));
}

PYBIND11_MODULE(kv_quantization_ops, m) {
    m.def("quantize_4bit", &quantize_kv_cache_4bit, 
          "Quantize KV cache to 4-bit using LVQ");
    m.def("dequantize_4bit", &dequantize_kv_cache_4bit, 
          "Dequantize 4-bit KV cache back to bfloat16/float16");
}

