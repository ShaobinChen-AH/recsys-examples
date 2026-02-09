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
from typing import List, Optional, Tuple

import tensorrt_llm
import torch
from configs import InferenceHSTUConfig, KVCacheConfig

KVCacheManagerImpl = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
KvCacheConfigCpp = tensorrt_llm.bindings.KvCacheConfig
DataType = tensorrt_llm.bindings.DataType


class HSTUHostKVStorageImpl:
    def __init__(self):
        pass

    def get_user_kvdata_info(self, user_id: int):
        pass

    def append_kvdata(
        self, user_id: int, start_pos: int, length: int, kv_data: List[torch.Tensor]
    ):
        pass

    def get_kv_data(
        self, user_id: int, length: int, layer_idx: int, output_buffer: torch.Tensor
    ) -> torch.Tensor:
        pass

    def evict_all_kvdata(self):
        pass


class DummyHSTUHostKVStorageImpl(HSTUHostKVStorageImpl):
    def __init__(self, num_layers, page_size, offload_chunksize):
        super(HSTUHostKVStorageImpl, self).__init__()
        self._num_layers = num_layers
        self._page_size = page_size
        self._offload_chunksize = offload_chunksize

        self.sequence_start_pos = dict()
        self.sequence_length = dict()
        self.kv_data_storage = [dict() for _ in range(self._num_layers)]

    def get_user_kvdata_info(self, user_id: int):
        seq_start_pos = self.sequence_start_pos.get(user_id, 0)
        seq_length = self.sequence_length.get(user_id, 0)
        return (seq_start_pos, seq_length)

    def append_kvdata(
        self, user_id: int, start_pos: int, length: int, kv_data: List[torch.Tensor]
    ):
        old_start_pos, old_length = self.get_user_kvdata_info(user_id)
        # assert old_start_pos + old_length == start_pos, \
        #     "{0} new kvdata starting position is {1}, unmatching current {2} ~ {3}".format(
        #         user_id, start_pos, old_start_pos, old_start_pos + old_length)
        # assert self._num_layers == len(
        #     kv_data), "the given kv_data has wrong number of layers"

        if user_id not in self.sequence_start_pos:
            self.sequence_start_pos[user_id] = start_pos
            self.sequence_length[user_id] = 0
        self.sequence_length[user_id] += length

        for layer_idx in range(self._num_layers):
            storage = self.kv_data_storage[layer_idx]
            if user_id not in storage:
                storage[user_id] = list()
            data_item = kv_data[layer_idx]
            if isinstance(data_item, dict):
                # 量化数据：存储为 dict
                storage[user_id].append(data_item)
            else:
                storage[user_id].append(data_item)

    def get_kv_data(
        self, user_id: int, length: int, layer_idx: int, output_buffer: torch.Tensor
    ) -> torch.Tensor:
        kv_data_list = []
        current_length = 0
        for data_chunk in self.kv_data_storage[layer_idx][user_id]:
            if data_chunk.shape[0] * self._page_size + current_length > length:
                slice_length = (length - current_length) // self._page_size
                kv_data_list.append(data_chunk[:slice_length, ...])
                break
            else:
                kv_data_list.append(data_chunk)
            current_length += kv_data_list[-1].shape[0] * self._page_size
        # assert sum([t.shape[0] for t in kv_data_list]) == output_buffer.shape[0]
        return torch.cat(kv_data_list, dim=0, out=output_buffer)

    def evict_all_kvdata(self):
        self.sequence_start_pos.clear()
        self.sequence_length.clear()
        for layer_idx in range(self._num_layers):
            self.kv_data_storage[layer_idx].clear()

    def get_kv_data_quantized(
            self,
            user_id: int,
            length: int,
            layer_idx: int,
            indices_buffer: torch.Tensor,
            scales_buffer: torch.Tensor,
            zero_points_buffer: torch.Tensor,
        ):
        """获取量化格式的 KV 数据"""
        current_length = 0
        buffer_idx = 0

        for data_chunk in self.kv_data_storage[layer_idx][user_id]:
            # 量化数据是 dict 格式
            if isinstance(data_chunk, dict):
                indices_chunk = data_chunk['indices']
                scales_chunk = data_chunk['scales']
                zero_points_chunk = data_chunk['zero_points']
            else:
                # 非量化数据不应该调用这个函数
                raise ValueError("Calling get_kv_data_quantized on non-quantized data")

            chunk_length = indices_chunk.shape[0] * self._page_size

            if current_length + chunk_length > length:
                # 只取需要的部分
                pages_needed = (length - current_length) // self._page_size
                indices_buffer[buffer_idx:buffer_idx + pages_needed, ...].copy_(indices_chunk[:pages_needed, ...])
                scales_buffer[buffer_idx:buffer_idx + pages_needed, ...].copy_(scales_chunk[:pages_needed, ...])
                zero_points_buffer[buffer_idx:buffer_idx + pages_needed, ...].copy_(zero_points_chunk[:pages_needed, ...])
                break
            else:
                indices_buffer[buffer_idx:buffer_idx + indices_chunk.shape[0], ...].copy_(indices_chunk)
                scales_buffer[buffer_idx:buffer_idx + scales_chunk.shape[0], ...].copy_(scales_chunk)
                zero_points_buffer[buffer_idx:buffer_idx + zero_points_chunk.shape[0], ...].copy_(zero_points_chunk)
                buffer_idx += indices_chunk.shape[0]

            current_length += chunk_length



class HSTUHostKVStorageManager:
    def __init__(
        self, hstu_config: InferenceHSTUConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        self.num_layers = hstu_config.num_layers
        self.head_dim = hstu_config.head_dim
        self.num_heads = hstu_config.num_heads
        self.page_size = kv_cache_config.page_size
        self.num_cache_pages = kv_cache_config.blocks_in_primary_pool

        self.max_seq_len = hstu_config.max_seq_len
        if kv_cache_config.max_attention_window is None:
            self.max_attention_window = hstu_config.max_seq_len
        else:
            self.max_attention_window = max(kv_cache_config.max_attention_window)

        self.offload_chunksize = kv_cache_config.offload_chunksize
        self.max_batch_size = hstu_config.max_batch_size

        self.enable_quantization = getattr(kv_cache_config, 'enable_kv_quantization', False)
        if self.enable_quantization:
            self.quantization_bits = getattr(kv_cache_config, 'kv_quantization_bits', 4)
            self.compression_ratio = 16 // self.quantization_bits

        self.impl: HSTUHostKVStorageImpl = DummyHSTUHostKVStorageImpl(
            self.num_layers, self.page_size, self.offload_chunksize
        )

        self.kv_cache_dtype = (
            torch.bfloat16
            if hstu_config.bf16
            else torch.float16
            if hstu_config.fp16
            else torch.float32
        )

        if self.enable_quantization:
            # 量化模式：存储 uint8 (4-bit packed)
            buffer_dtype = torch.uint8
            buffer_head_dim = (self.head_dim + 1) // 2  # 4-bit packed
        else:
            buffer_dtype = self.kv_cache_dtype
            buffer_head_dim = self.head_dim

        self.static_kvdata_buffer_ = torch.empty(
            (
                self.num_layers,
                (self.max_batch_size * self.max_seq_len) // self.page_size,
                2,
                self.page_size,
                self.num_heads,
                self.head_dim,
            ),
            dtype=self.kv_cache_dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        ).uniform_(-0.05, 0.05)

        if self.enable_quantization:
            self.static_kvdata_scales_ = torch.empty(
                (
                    self.num_layers,
                    (self.max_batch_size * self.max_seq_len) // self.page_size,
                    2,
                    self.num_heads,
                ),
                dtype=torch.float32,
                device=torch.device("cpu"),
                pin_memory=True,
            )
            self.static_kvdata_zero_points_ = torch.empty(
                (
                    self.num_layers,
                    (self.max_batch_size * self.max_seq_len) // self.page_size,
                    2,
                    self.num_heads,
                ),
                dtype=torch.float32,
                device=torch.device("cpu"),
                pin_memory=True,
            )
        else:
            self.static_kvdata_scales_ = None
            self.static_kvdata_zero_points_ = None

    def fetch_kv_data(
        self, user_id: int, length: int, layer_idx: int, output_buffer: torch.Tensor
    ):
        if self.enable_quantization and scales_buffer is not None and zero_points_buffer is not None:
            # 量化模式下需要提取三个 buffer
            self.impl.get_kv_data_quantized(
                user_id, length, layer_idx, output_buffer, scales_buffer, zero_points_buffer
            )
        else:
            self.impl.get_kv_data(user_id, length, layer_idx, output_buffer)

    def get_user_kvdata_info(self, user_id: int) -> Tuple[int, int]:
        return self.impl.get_user_kvdata_info(user_id)

    def lookup_kvdata(
        self,
        user_ids: torch.Tensor,
        cached_start_pos: torch.Tensor,
        cached_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        onload_history_seq_length = torch.zeros_like(
            user_ids, dtype=torch.int32, device=torch.device("cpu")
        )
        onload_history_seqlen_offsets = torch.zeros(
            (user_ids.shape[0] + 1,), dtype=torch.int32, device=torch.device("cpu")
        )
        for idx in range(user_ids.shape[0]):
            user_id = user_ids[idx].item()
            cache_len = cached_lengths[idx].item()
            onload_history_seq_length[idx] = self.get_onload_history_seqlen(
                user_id, cached_start_pos[idx].item() if cache_len > 0 else -1
            )
        torch.cumsum(
            onload_history_seq_length, 0, out=onload_history_seqlen_offsets[1:]
        )
        onload_length = onload_history_seqlen_offsets[-1].item()
        if onload_length == 0:
            return (onload_length, None, None)

        # copy lookup results into buffer (compress into one)
        for idx in range(user_ids.shape[0]):
            user_id = user_ids[idx].item()
            length = onload_history_seq_length[idx].item()
            if length == 0:
                continue

            start_page_idx = onload_history_seqlen_offsets[idx].item() // self.page_size
            end_page_idx = (
                onload_history_seqlen_offsets[idx + 1].item() // self.page_size
            )
            for layer_idx in range(self.num_layers):
                self.fetch_kv_data(
                    user_id,
                    length,
                    layer_idx,
                    self.static_kvdata_buffer_[
                        layer_idx, start_page_idx:end_page_idx, ...
                    ],
                    self.static_kvdata_scales_[layer_idx, start_page_idx:end_page_idx, ...] if self.enable_quantization else None,
                    self.static_kvdata_zero_points_[layer_idx, start_page_idx:end_page_idx, ...] if self.enable_quantization else None,
                )

        onload_kv_page_ids = torch.arange(
            start=self.num_cache_pages,
            end=onload_length // self.page_size + self.num_cache_pages,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        onload_kv_page_indptr = (onload_history_seqlen_offsets / self.page_size).to(
            dtype=torch.int32, device=torch.cuda.current_device()
        )

        return onload_length, onload_kv_page_ids, onload_kv_page_indptr

    def append_kvdata(
        self,
        offloaded_kv_data: List[torch.Tensor],  # (total_num_pages, *single_page_shape)
        user_ids: torch.Tensor,
        offload_start_pos: torch.Tensor,
        offload_page_indptr: torch.Tensor,
        offloaded_scales: List[torch.Tensor] = None,  # === 新增 ===
        offloaded_zero_points: List[torch.Tensor] = None,
    ):
        for idx in range(len(user_ids)):
            uid = user_ids[idx].item()
            start_pos = offload_start_pos[idx].item()
            page_start = offload_page_indptr[idx].item()
            page_end = offload_page_indptr[idx + 1].item()
            length = (page_end - page_start) * self.page_size
            if length == 0:
                continue

            kv_data_per_layer = []
            for layer_idx in range(self.num_layers):
                layer_data = {
                    'indices': offloaded_kv_data[layer_idx][page_start:page_end, ...]
                    .detach()
                    .clone()
                }
                # === 新增：如果有量化参数，也存储 ===
                if offloaded_scales is not None and offloaded_zero_points is not None:
                    layer_data['scales'] = offloaded_scales[layer_idx][page_start:page_end, ...].detach().clone()
                    layer_data['zero_points'] = offloaded_zero_points[layer_idx][page_start:page_end, ...].detach().clone()

                kv_data_per_layer.append(layer_data)

            self.impl.append_kvdata(
                uid,
                start_pos,
                length,
                kv_data_per_layer,
            )

    def evict_all_kvdata(self):
        self.impl.evict_all_kvdata()

    def get_lookup_buffer(self) -> torch.Tensor:
        if self.enable_quantization:
            return {
                'indices': self.static_kvdata_buffer_,
                'scales': self.static_kvdata_scales_,
                'zero_points': self.static_kvdata_zero_points_,
            }
        else:
            return self.static_kvdata_buffer_

    def get_lookup_scales(self):
        return self.static_kvdata_scales_ if self.enable_quantization else None

    def get_lookup_zero_points(self):
        return self.static_kvdata_zero_points_ if self.enable_quantization else None

    def get_onload_history_seqlen(self, user_id: int, cached_start_pos: int) -> int:
        (offloaded_start_pos, offloaded_length) = self.get_user_kvdata_info(user_id)
        return (
            min(offloaded_length, cached_start_pos - offloaded_start_pos)
            if cached_start_pos > -1
            else offloaded_length
        )
