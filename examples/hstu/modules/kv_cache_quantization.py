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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class KVCacheQuantizer(nn.Module):
    """
    KV Cache Quantizer implementing Random Rotation + Layer-wise Vector Quantization (LVQ).
    
    This quantizer reduces KV cache memory usage and CPU-GPU transfer bandwidth
    by quantizing KV cache to 4-bit with random rotation preprocessing.
    """
    
    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        quantization_bits: int = 4,
        use_random_rotation: bool = True,
        rotation_group_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the KV Cache Quantizer.
        
        Args:
            head_dim: Dimension per attention head
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer layers
            quantization_bits: Number of bits for quantization (default: 4)
            use_random_rotation: Whether to apply random rotation before quantization
            rotation_group_size: Size of groups for random rotation
            dtype: Data type for rotation matrices
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.quantization_bits = quantization_bits
        self.use_random_rotation = use_random_rotation
        self.rotation_group_size = rotation_group_size
        self.dtype = dtype
        
        # Calculate number of quantization levels
        self.num_levels = 2 ** quantization_bits
        
        # Initialize rotation matrices if needed
        self.rotation_matrices = None
        if use_random_rotation:
            self.rotation_matrices = self._init_rotation_matrices()
        
        # Codebooks for LVQ (per-layer, per-head)
        self.codebooks = nn.ParameterList()
        for _ in range(num_layers):
            layer_codebooks = nn.ParameterList()
            for _ in range(num_heads):
                # Initialize codebook with uniform grid
                codebook = torch.linspace(-1, 1, self.num_levels, dtype=dtype)
                layer_codebooks.append(nn.Parameter(codebook))
            self.codebooks.append(layer_codebooks)
    
    def _init_rotation_matrices(self) -> list:
        """
        Initialize random orthogonal rotation matrices using QR decomposition.
        
        Returns:
            List of rotation matrices for each layer and head group
        """
        rotation_matrices = []
        num_groups = (self.head_dim + self.rotation_group_size - 1) // self.rotation_group_size
        
        for layer_idx in range(self.num_layers):
            layer_matrices = []
            for head_idx in range(self.num_heads):
                head_matrices = []
                for group_idx in range(num_groups):
                    # Generate random orthogonal matrix using QR decomposition
                    size = min(self.rotation_group_size, 
                              self.head_dim - group_idx * self.rotation_group_size)
                    random_matrix = torch.randn(size, size, dtype=torch.float32)
                    q, r = torch.linalg.qr(random_matrix)
                    # Ensure determinant is 1 (proper rotation)
                    if torch.det(q) < 0:
                        q[:, 0] *= -1
                    head_matrices.append(q)
                layer_matrices.append(head_matrices)
            rotation_matrices.append(layer_matrices)
        
        return rotation_matrices
    
    def _apply_rotation(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply random rotation to input tensor.
        
        Args:
            x: Input tensor of shape [..., num_heads, head_dim]
            layer_idx: Layer index for selecting rotation matrix
            
        Returns:
            Rotated tensor of same shape
        """
        if not self.use_random_rotation or self.rotation_matrices is None:
            return x
        
        num_groups = (self.head_dim + self.rotation_group_size - 1) // self.rotation_group_size
        rotated = torch.empty_like(x)
        
        for head_idx in range(self.num_heads):
            head_rotation_matrices = self.rotation_matrices[layer_idx][head_idx]
            for group_idx in range(num_groups):
                start_idx = group_idx * self.rotation_group_size
                end_idx = min((group_idx + 1) * self.rotation_group_size, self.head_dim)
                
                # Apply rotation: x_rotated = R @ x
                group_data = x[..., head_idx, start_idx:end_idx]
                rotation_matrix = head_rotation_matrices[group_idx].to(device = x.device, dtype = x.dtype)
                rotated[..., head_idx, start_idx:end_idx] = torch.matmul(
                    group_data, rotation_matrix.T
                )
        
        return rotated
    
    def _apply_inverse_rotation(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Apply inverse random rotation (transpose of rotation matrix).
        
        Args:
            x: Input tensor of shape [..., num_heads, head_dim]
            layer_idx: Layer index for selecting rotation matrix
            
        Returns:
            Unrotated tensor of same shape
        """
        if not self.use_random_rotation or self.rotation_matrices is None:
            return x
        
        num_groups = (self.head_dim + self.rotation_group_size - 1) // self.rotation_group_size
        unrotated = torch.empty_like(x)
        
        for head_idx in range(self.num_heads):
            head_rotation_matrices = self.rotation_matrices[layer_idx][head_idx]
            for group_idx in range(num_groups):
                start_idx = group_idx * self.rotation_group_size
                end_idx = min((group_idx + 1) * self.rotation_group_size, self.head_dim)
                
                # Apply inverse rotation: x = R^T @ x_rotated
                group_data = x[..., head_idx, start_idx:end_idx]
                rotation_matrix = head_rotation_matrices[group_idx].to(device = x.device, dtype = x.dtype)
                unrotated[..., head_idx, start_idx:end_idx] = torch.matmul(
                    group_data, rotation_matrix
                )
        
        return unrotated
    
    def quantize(
        self, 
        kv_data: torch.Tensor, 
        layer_idx: int,
        per_head_scale: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize KV cache data using LVQ.
        
        Args:
            kv_data: KV cache tensor of shape [num_pages, 2, page_size, num_heads, head_dim]
            layer_idx: Layer index
            per_head_scale: Whether to compute scale per head
            
        Returns:
            quantized_indices: Quantized indices [num_pages, 2, page_size, num_heads, head_dim]
            scales: Scale factors [num_heads] or scalar
            zero_points: Zero points [num_heads] or scalar
        """
        batch_size, k2v, seq_len, num_heads, head_dim = kv_data.shape
        
        # Apply random rotation
        rotated_data = self._apply_rotation(kv_data, layer_idx)
        
        # Compute per-head scales and zero points
        if per_head_scale:
            # Reshape to compute per-head statistics
            flat_data = rotated_data.view(batch_size * k2v * seq_len, num_heads, head_dim)
            
            # Compute min/max per head
            data_min = flat_data.min(dim=0)[0].min(dim=-1)[0]  # [num_heads]
            data_max = flat_data.max(dim=0)[0].max(dim=-1)[0]  # [num_heads]
            
            # Expand dimensions for broadcasting
            data_min = data_min.view(1, num_heads, 1)
            data_max = data_max.view(1, num_heads, 1)
        else:
            data_min = rotated_data.min()
            data_max = rotated_data.max()
        
        # Compute scales and zero points
        scales = (data_max - data_min) / (self.num_levels - 1)
        zero_points = data_min
        
        # Quantize to indices
        normalized = (rotated_data - zero_points) / scales
        quantized_indices = torch.clamp(
            torch.round(normalized).long(),
            0,
            self.num_levels - 1
        )
        
        # Pack indices to reduce memory (4-bit packing)
        if self.quantization_bits == 4:
            quantized_indices = self._pack_4bit(quantized_indices)
        
        return quantized_indices, scales, zero_points
    
    def dequantize(
        self,
        quantized_indices: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Dequantize KV cache data.
        
        Args:
            quantized_indices: Quantized indices from quantize()
            scales: Scale factors
            zero_points: Zero points
            layer_idx: Layer index
            
        Returns:
            Dequantized KV cache tensor
        """
        # Unpack if 4-bit
        if self.quantization_bits == 4:
            quantized_indices = self._unpack_4bit(quantized_indices)
        
        # Dequantize
        dequantized = quantized_indices.float() * scales + zero_points
        
        # Cast to original dtype
        dequantized = dequantized.to(self.dtype)
        
        # Apply inverse rotation
        dequantized = self._apply_inverse_rotation(dequantized, layer_idx)
        
        return dequantized
    
    def _pack_4bit(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Pack 4-bit indices into 8-bit integers to save memory.
        
        Args:
            indices: Tensor of indices in range [0, 15]
            
        Returns:
            Packed tensor with half the size in last dimension
        """
        # Ensure indices are contiguous and in correct shape
        original_shape = indices.shape
        indices = indices.view(-1)
        
        # Pad if odd number of elements
        if indices.numel() % 2 == 1:
            indices = torch.cat([indices, torch.zeros(1, dtype=indices.dtype, device=indices.device)])
        
        # Pack two 4-bit values into one 8-bit value
        high = indices[0::2] << 4
        low = indices[1::2] & 0x0F
        packed = high | low
        
        # Reshape back
        new_shape = list(original_shape[:-1]) + [original_shape[-1] // 2]
        return packed.view(new_shape).to(torch.uint8)
    
    def _unpack_4bit(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack 8-bit integers into 4-bit indices.
        
        Args:
            packed: Packed tensor
            
        Returns:
            Unpacked indices tensor
        """
        original_shape = packed.shape
        packed = packed.view(-1).long()
        
        # Unpack
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        
        # Interleave
        indices = torch.empty(packed.numel() * 2, dtype=torch.long, device=packed.device)
        indices[0::2] = high
        indices[1::2] = low
        
        # Reshape back (accounting for doubled last dimension)
        new_shape = list(original_shape[:-1]) + [original_shape[-1] * 2]
        return indices.view(new_shape)
    
    def to(self, device):
        """Move quantizer to device."""
        if self.rotation_matrices is not None:
            for layer_matrices in self.rotation_matrices:
                for head_matrices in layer_matrices:
                    for i, matrix in enumerate(head_matrices):
                        head_matrices[i] = matrix.to(device)
        
        for layer_codebooks in self.codebooks:
            for i, codebook in enumerate(layer_codebooks):
                layer_codebooks[i] = codebook.to(device)
        
        return self


def create_kv_quantizer(
    hstu_config,
    kv_cache_config,
) -> Optional[KVCacheQuantizer]:
    """
    Factory function to create KV Cache Quantizer from configs.
    
    Args:
        hstu_config: InferenceHSTUConfig
        kv_cache_config: KVCacheConfig
        
    Returns:
        KVCacheQuantizer instance or None if quantization disabled
    """
    if not getattr(kv_cache_config, 'enable_kv_quantization', False):
        return None
    
    return KVCacheQuantizer(
        head_dim=hstu_config.head_dim,
        num_heads=hstu_config.num_heads,
        num_layers=hstu_config.num_layers,
        quantization_bits=getattr(kv_cache_config, 'kv_quantization_bits', 4),
        use_random_rotation=getattr(kv_cache_config, 'use_random_rotation', True),
        rotation_group_size=getattr(kv_cache_config, 'rotation_group_size', 128),
        dtype=torch.bfloat16 if hstu_config.bf16 else torch.float16 if hstu_config.fp16 else torch.float32,
    )

