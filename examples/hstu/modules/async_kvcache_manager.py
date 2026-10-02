import math
import threading
import time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import paged_kvcache_ops
import torch
from configs import KVCacheMetadata
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class AsyncHSTUKVCacheManager:
    def __init__(
        self,
        num_layers,
        num_kv_heads,
        kv_headdim,
        num_tokens_per_page,
        num_primary_cache_pages,
        num_onload_buffer_pages,
        num_reserved_buffer_pages,
        num_tokens_per_chunk,
        max_num_sequences,
        max_sequence_length,
        max_batch_size,
        max_queued_offload_tokens,
        num_onload_buffer_chunks=1,
        num_offload_buffer_chunks=8,
        num_memcpy_workers=8,
        enable_nvcomp=False,
    ):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.onload_worker = ThreadPoolExecutor(max_workers=1)

        self.num_layers = num_layers
        self.num_heads = num_kv_heads
        self.head_dim = kv_headdim
        self.page_size = num_tokens_per_page
        self.num_primary_cache_pages = num_primary_cache_pages
        self.num_onload_buffer_pages = num_onload_buffer_pages
        self.num_reserved_buffer_pages = num_reserved_buffer_pages
        self.chunk_size = num_tokens_per_chunk
        self.num_onload_buffer_chunks = int(num_onload_buffer_chunks)
        self.num_offload_buffer_chunks = int(num_offload_buffer_chunks)
        self.enable_nvcomp = bool(enable_nvcomp)
        self.max_num_sequences = max_num_sequences
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.max_num_pages_per_seq = math.ceil(
            self.max_sequence_length / self.page_size
        )

        self.num_cache_pages = num_primary_cache_pages + num_onload_buffer_pages
        self.cache_table = torch.empty(
            [
                num_layers,
                self.num_cache_pages,
                2,
                self.page_size,
                self.num_heads,
                self.head_dim,
            ],
            dtype=torch.bfloat16,
            device=torch.cuda.current_device(),
        )

        self.host_kv_mgr = paged_kvcache_ops.HostKVStorageImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.chunk_size,
        )
        self.gpu_kvcache_mgr = paged_kvcache_ops.GPUKVCacheMangerImpl(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.page_size,
            self.num_primary_cache_pages,
            self.num_onload_buffer_pages,
            self.num_reserved_buffer_pages,
            self.chunk_size,
            self.max_num_sequences,
            self.max_num_sequences,
            self.cache_table,
            self.host_kv_mgr,
            max_queued_offload_tokens,
            self.num_onload_buffer_chunks,
            self.num_offload_buffer_chunks,
            num_memcpy_workers,
            self.enable_nvcomp,
        )

        self.static_page_ids_gpu_buffer = torch.empty(
            [
                self.max_batch_size * self.max_num_pages_per_seq,
            ],
            dtype=torch.int32,
        ).cuda()
        self.static_offload_page_ids_gpu_buffer = torch.empty(
            [
                self.max_batch_size * self.max_num_pages_per_seq,
            ],
            dtype=torch.int32,
        ).cuda()
        self.static_metadata_gpu_buffer = torch.empty(
            [
                self.max_batch_size * 5
                + 4
                + self.max_batch_size * self.max_sequence_length * 2,
            ],
            dtype=torch.int32,
        ).cuda()
        self.static_onload_handle = paged_kvcache_ops.KVOnloadHandle(self.num_layers)
        self.static_empty_offload_handle = paged_kvcache_ops.KVOffloadHandle()

        self.cache_table_list = [
            self.cache_table[idx] for idx in range(self.num_layers)
        ]

        self.last_origin_cached_lengths = None
        self.last_new_tokens = None
        self.last_num_offload_pages = None
        self.last_max_seqlen = None
        self.last_offload_submission = {
            "generation": 0,
            "accepted": False,
            "tokens_by_user": {},
            "submitted_at": None,
            "completion_target": None,
        }
        self._prepare_generation = 0
        self._staged_prepare = None
        self._staged_prepare_lock = threading.Lock()
        self.last_consumed_prepare_generation = None
        self._last_offload_completion_target = 0
        self._active_prepare = None

    def physical_hbm_breakdown(self) -> dict:
        """Return bytes allocated by the KV state manager on the GPU.

        ``cache_table`` owns both primary and onload pages.  The C++ manager
        also owns one CUDA copy buffer per configured onload/offload chunk;
        those buffers are raw pointers rather than PyTorch tensors, so their
        sizes are reconstructed from the same chunk geometry used by C++.
        The three static metadata tensors are persistent device allocations
        and are included explicitly.
        """
        if self.enable_nvcomp:
            raise RuntimeError(
                "physical KV HBM accounting is not supported with NVCOMP enabled; "
                "disable NVCOMP for the fixed physical-HBM envelope"
            )
        element_bytes = int(self.cache_table.element_size())
        cache_table_bytes = int(self.cache_table.numel() * element_bytes)
        chunk_bytes = (
            int(self.chunk_size)
            * 2
            * int(self.num_heads)
            * int(self.head_dim)
            * element_bytes
        )
        onload_buffer_bytes = self.num_onload_buffer_chunks * chunk_bytes
        offload_buffer_bytes = self.num_offload_buffer_chunks * chunk_bytes
        metadata_bytes = sum(
            int(t.numel() * t.element_size())
            for t in (
                self.static_page_ids_gpu_buffer,
                self.static_offload_page_ids_gpu_buffer,
                self.static_metadata_gpu_buffer,
            )
        )
        return {
            "cache_table_bytes": cache_table_bytes,
            "onload_device_buffer_bytes": int(onload_buffer_bytes),
            "offload_device_buffer_bytes": int(offload_buffer_bytes),
            "static_cuda_metadata_bytes": int(metadata_bytes),
            "physical_hbm_bytes": int(
                cache_table_bytes
                + onload_buffer_bytes
                + offload_buffer_bytes
                + metadata_bytes
            ),
        }

    def physical_hbm_bytes(self) -> int:
        return int(self.physical_hbm_breakdown()["physical_hbm_bytes"])

    def prepare_kvcache_async(
        self,
        batch_size,
        user_ids,
        total_history_lengths,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        static_metadata_gpu_buffer,
        static_onload_handle,
    ):
        origin_cached_lengths = self.gpu_kvcache_mgr.get_total_cache_length(user_ids)
        new_tokens = sum(
            [
                total_history_lengths[idx] - origin_cached_lengths[idx]
                for idx in range(batch_size)
            ]
        )

        self.last_origin_cached_lengths = list(origin_cached_lengths)
        self.last_new_tokens = int(new_tokens)

        offload_uids_buffer = torch.empty(
            [
                batch_size,
            ],
            dtype=torch.int64,
        )
        metadata_host_buffer = torch.empty(
            [
                batch_size * 7 + 7,
            ],
            dtype=torch.int,
            pin_memory=True,
        )
        # metadata_gpu_buffer = torch.empty([batch_size * 5 + 4 + new_tokens * 2,], dtype=torch.int, device = torch.cuda.current_device())

        kvcache_metadata_fut = self.executor.submit(
            paged_kvcache_ops.prepare_kvcache,
            self.gpu_kvcache_mgr,
            self.host_kv_mgr,
            user_ids,
            total_history_lengths,
            static_page_ids_gpu_buffer,
            static_offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer,
            static_metadata_gpu_buffer,
        )

        static_onload_handle.reset()
        onload_fut = self.onload_worker.submit(
            self.gpu_kvcache_mgr.onload_kvcache, user_ids, static_onload_handle
        )

        return [
            origin_cached_lengths,
            new_tokens,
            offload_uids_buffer,
            metadata_host_buffer,
            static_metadata_gpu_buffer,
            kvcache_metadata_fut,
            onload_fut,
        ]

    def stage_prepare_kvcache(self, user_ids, total_history_lengths):
        """Start the exact KV prepare/onload operation consumed by forward.

        Only one request may use the manager's static metadata buffers and
        static onload handle at a time.  Staging therefore owns those buffers
        until :meth:`consume_staged_prepare` is called.
        """
        normalized_users = tuple(int(uid) for uid in user_ids)
        normalized_lengths = tuple(int(length) for length in total_history_lengths)
        with self._staged_prepare_lock:
            if self._staged_prepare is not None:
                if (
                    self._staged_prepare["user_ids"] == normalized_users
                    and self._staged_prepare["total_history_lengths"]
                    == normalized_lengths
                ):
                    return int(self._staged_prepare["generation"])
                raise RuntimeError(
                    "a different KV request is already staged and unconsumed"
                )
            result = self.prepare_kvcache_async(
                len(normalized_users),
                list(normalized_users),
                list(normalized_lengths),
                self.static_page_ids_gpu_buffer,
                self.static_offload_page_ids_gpu_buffer,
                self.static_metadata_gpu_buffer,
                self.static_onload_handle,
            )
            self._prepare_generation += 1
            self._staged_prepare = {
                "generation": int(self._prepare_generation),
                "user_ids": normalized_users,
                "total_history_lengths": normalized_lengths,
                "result": result,
            }
            return int(self._prepare_generation)

    def consume_staged_prepare(self, user_ids, total_history_lengths):
        """Return and clear a matching staged request, or ``None``."""
        normalized_users = tuple(int(uid) for uid in user_ids)
        normalized_lengths = tuple(int(length) for length in total_history_lengths)
        with self._staged_prepare_lock:
            staged = self._staged_prepare
            if staged is None:
                return None
            if (
                staged["user_ids"] != normalized_users
                or staged["total_history_lengths"] != normalized_lengths
            ):
                raise RuntimeError(
                    "staged KV request does not match the request being executed"
                )
            self._staged_prepare = None
            self.last_consumed_prepare_generation = int(staged["generation"])
            self._active_prepare = {
                "generation": int(staged["generation"]),
                "result": staged["result"],
            }
            return int(staged["generation"]), staged["result"]

    def complete_active_prepare(self) -> None:
        """Release bookkeeping after a forward consumed the prepared buffers."""
        with self._staged_prepare_lock:
            self._active_prepare = None

    def active_prepare_generation(self):
        with self._staged_prepare_lock:
            if self._active_prepare is None:
                return None
            return int(self._active_prepare["generation"])

    def abort_active_prepare(self) -> None:
        """Drain a consumed prepare after a forward failure.

        The static onload handle and metadata buffers are shared by all
        requests.  Waiting for their worker futures before reuse prevents a
        failed request from publishing a partial onload to the next request.
        """
        with self._staged_prepare_lock:
            active = self._active_prepare
            self._active_prepare = None
            staged = self._staged_prepare
            self._staged_prepare = None
        candidate = active or staged
        if candidate is None:
            return
        result = candidate.get("result", candidate)
        for index in (5, 6):
            try:
                result[index].result()
            except Exception:
                pass
        self.static_onload_handle.reset()

    def cancel_staged_prepare(self) -> None:
        """Cancel an unconsumed staged request and drain its workers."""
        self.abort_active_prepare()

    def prepare_kvcache_wait(
        self,
        onload_fut,
        kvcache_metadata_fut,
        batch_size,
        new_tokens,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,  # input static
        static_onload_handle,
    ):
        kvcache_metadata_fut.result()
        return self.get_kvcache_metadata_from_buffer(
            batch_size,
            new_tokens,
            static_page_ids_gpu_buffer,
            static_offload_page_ids_gpu_buffer,
            offload_uids_buffer,
            metadata_host_buffer,
            metadata_gpu_buffer,
            static_onload_handle,
        )

    def offload_kvcache(self, kvcache_metadata):
        num_offload_pages = len(kvcache_metadata.offload_page_ids)
        if num_offload_pages == 0:
            kvcache_metadata.kv_offload_handle.set_no_offload()
            self.last_offload_submission = {
                "generation": int(self.last_offload_submission["generation"]),
                "accepted": False,
                "tokens_by_user": {},
                "submitted_at": None,
                "completion_target": None,
            }
            return None

        accepted = self.gpu_kvcache_mgr.offload_kvcache(
            kvcache_metadata.kv_offload_handle,
            kvcache_metadata.offload_user_ids,
            kvcache_metadata.offload_page_ids,
            kvcache_metadata.new_offload_startpos,
            kvcache_metadata.new_offload_lengths,
        )
        if not accepted:
            # The native manager rejects before it creates per-layer CUDA
            # events.  Prevent later attention layers from marking an
            # uninitialized handle ready.
            kvcache_metadata.kv_offload_handle.set_no_offload()
        tokens_by_user = {
            int(uid): int(length)
            for uid, length in zip(
                kvcache_metadata.offload_user_ids.tolist(),
                kvcache_metadata.new_offload_lengths.tolist(),
            )
            if int(length) > 0
        }
        completed_count = int(self.gpu_kvcache_mgr.get_completed_offload_count())
        if accepted:
            self._last_offload_completion_target = max(
                int(self._last_offload_completion_target), completed_count
            ) + 1
        self.last_offload_submission = {
            "generation": int(self.last_offload_submission["generation"]) + 1,
            "accepted": bool(accepted),
            "tokens_by_user": tokens_by_user if accepted else {},
            "submitted_at": time.perf_counter() if accepted else None,
            "completion_target": (
                int(self._last_offload_completion_target) if accepted else None
            ),
        }
        return bool(accepted)

    def get_kvcache_metadata_from_buffer(
        self,
        batch_size,
        new_tokens,
        static_page_ids_gpu_buffer,
        static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,  # input static
        static_onload_handle,
    ):
        # assert int(metadata_host_buffer[batch_size * 4 + 2]) == new_tokens
        offload_handle = self.static_empty_offload_handle
        if int(metadata_host_buffer[batch_size * 7 + 5]) > 0:
            offload_handle = paged_kvcache_ops.KVOffloadHandle(
                self.num_layers, self.gpu_kvcache_mgr, True
            )

        self.last_num_offload_pages = int(metadata_host_buffer[batch_size * 7 + 5])
        self.last_max_seqlen = int(
            torch.max(metadata_host_buffer[batch_size * 2 + 1 : batch_size * 3 + 1]).item()
        )

        return KVCacheMetadata(
            kv_indices=static_page_ids_gpu_buffer[
                : metadata_host_buffer[batch_size * 7 + 4]
            ],
            kv_indptr=metadata_gpu_buffer[: batch_size + 1],
            kv_last_page_len=metadata_gpu_buffer[batch_size + 1 : batch_size * 2 + 1],
            total_history_lengths=metadata_gpu_buffer[
                batch_size * 2 + 1 : batch_size * 3 + 1
            ],
            total_history_offsets=metadata_gpu_buffer[
                batch_size * 3 + 1 : batch_size * 4 + 2
            ],
            batch_indices=metadata_gpu_buffer[
                batch_size * 5 + 4 : batch_size * 5 + 4 + new_tokens
            ],
            position=metadata_gpu_buffer[
                batch_size * 5 + 4 + new_tokens : batch_size * 5 + 4 + new_tokens * 2
            ],
            new_history_nnz=new_tokens,
            new_history_nnz_cuda=metadata_gpu_buffer[
                batch_size * 4 + 2 : batch_size * 4 + 3
            ],
            kv_cache_table=self.cache_table_list,
            kv_onload_handle=static_onload_handle,
            kv_offload_handle=offload_handle,
            offload_user_ids=offload_uids_buffer[
                : metadata_host_buffer[batch_size * 7 + 6]
            ],
            offload_page_ids=static_offload_page_ids_gpu_buffer[
                : int(metadata_host_buffer[batch_size * 7 + 5])
            ].clone(),
            new_offload_startpos=metadata_host_buffer[
                batch_size * 5 + 4 : batch_size * 6 + 4
            ],
            new_offload_lengths=metadata_host_buffer[
                batch_size * 6 + 4 : batch_size * 7 + 4
            ],
            max_seqlen=torch.max(
                metadata_host_buffer[batch_size * 2 + 1 : batch_size * 3 + 1]
            ).item(),
        )

    def strip_cached_tokens(self, batch, origin_num_cached):
        torch.cuda.nvtx.range_push("strip_cached_tokens")

        num_context = len(batch.contextual_feature_names)

        num_cached = torch.clamp_min(origin_num_cached - num_context, 0).to(torch.int32)
        num_cached_action = num_cached // 2
        num_cached_item = num_cached - num_cached_action
        num_hist_cached = torch.concat([num_cached_item, num_cached_action], dim=0)

        old_offsets = batch.features.offsets().cpu()
        old_lengths = batch.features.lengths().cpu()

        item_offset = num_context * batch.batch_size
        item_offset + batch.batch_size

        new_lengths = torch.zeros_like(old_lengths)
        new_lengths[:item_offset] = torch.where(
            (origin_num_cached == 0).view(-1, batch.batch_size),
            old_lengths[:item_offset].view(-1, batch.batch_size),
            new_lengths[:item_offset].view(-1, batch.batch_size),
        ).view(-1)
        new_lengths[item_offset:] = old_lengths[item_offset:] - num_hist_cached

        startpos = (
            old_offsets[item_offset : item_offset + 2 * batch.batch_size]
            + num_hist_cached
        )
        endpos = old_offsets[item_offset + 1 :]

        old_values = batch.features.values()
        new_hist_value = [
            old_values[startpos[idx] : endpos[idx]]
            for idx in range(2 * batch.batch_size)
        ]

        new_context_value = [
            old_values[idx : idx + 1]
            for idx in range(num_context * batch.batch_size)
            if int(new_lengths[idx]) > 0
        ]

        new_features = KeyedJaggedTensor(
            values=torch.cat(new_context_value + new_hist_value, dim=0),
            lengths=new_lengths.cuda(),
            keys=batch.features.keys(),
        )

        torch.cuda.nvtx.range_pop()
        return replace(batch, features=new_features)
