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
import random
import time

import paged_kvcache_ops
import pytest
import torch
from commons.datasets.hstu_batch import FeatureConfig
from configs import KVCacheMetadata, get_inference_hstu_config, get_kvcache_config
from modules.async_kvcache_manager import AsyncHSTUKVCacheManager


def get_test_kvcache_mgr(
    num_layers,
    blocks_in_primary_pool,
    page_size,
    offload_chunksize,
    enable_nvcomp=False,
):
    # requires to test sequentientially
    max_batch_size = 8
    max_seqlen = 10240

    item_fea_name, item_vocab_size = "item_feat", 100
    action_fea_name, action_vocab_size = "act_feat", 128
    feature_configs = [
        FeatureConfig(
            feature_names=[item_fea_name, action_fea_name],
            max_item_ids=[item_vocab_size - 1, action_vocab_size - 1],
            max_sequence_length=max_seqlen,
            is_jagged=False,
        ),
    ]

    hidden_dim_size = 512
    num_heads = 4
    head_dim = 128
    inference_dtype = torch.bfloat16

    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seqlen,
        dtype=inference_dtype,
    )

    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
    )

    async_kvcache_mgr = AsyncHSTUKVCacheManager(
        hstu_config.num_layers,
        hstu_config.num_heads,
        hstu_config.head_dim,
        kv_cache_config.page_size,
        kv_cache_config.blocks_in_primary_pool,
        math.ceil(
            hstu_config.max_batch_size
            * hstu_config.max_seq_len
            / kv_cache_config.page_size
        ),
        4 * hstu_config.max_batch_size * hstu_config.max_seq_len,
        kv_cache_config.offload_chunksize,
        -1,
        hstu_config.max_seq_len,
        hstu_config.max_batch_size,
        4 * hstu_config.max_batch_size * hstu_config.max_seq_len,
        1,
        8,
        8,
        enable_nvcomp,
    )

    # randomnize kvcache data
    for idx in range(num_layers):
        async_kvcache_mgr.cache_table[idx].uniform_(-0.5, 0.5)

    return async_kvcache_mgr


def init_random_kvcache_status():
    pass


def reset_kvcache_status():
    pass


def get_test_userids_and_metadata(
    min_user_id, seq_lengths, num_layers, gpu_kvcache_mgr
):
    batch_size = len(seq_lengths)
    page_size = 32
    chunk_size = 1024
    blocks_in_primary_pool = 10240

    uids = list(range(min_user_id * 2, (min_user_id + batch_size) * 2))
    random.shuffle(uids)
    user_ids = torch.tensor(uids[:batch_size]).long().cuda()
    offload_user_ids = user_ids.clone().cpu()

    num_pages = torch.floor(seq_lengths / chunk_size).int() * int(
        chunk_size / page_size
    )
    num_pages = num_pages.cpu().tolist()
    offload_page_ids = torch.cat(
        [
            torch.randint(blocks_in_primary_pool, (int(num_pages[idx]),))
            for idx in range(batch_size)
        ],
        0,
    ).int()

    kv_offload_handle = paged_kvcache_ops.KVOffloadHandle(
        num_layers, gpu_kvcache_mgr, True
    )

    # zero start
    new_offload_startpos = torch.zeros((batch_size,)).int().cpu()
    new_offload_lengths = torch.floor(seq_lengths / chunk_size).int().cpu() * int(
        chunk_size
    )

    kvcache_metadata = KVCacheMetadata(
        offload_user_ids=offload_user_ids,
        offload_page_ids=offload_page_ids,
        kv_offload_handle=kv_offload_handle,
        new_offload_startpos=new_offload_startpos,
        new_offload_lengths=new_offload_lengths,
    )

    return user_ids, kvcache_metadata


def test_kvcache_offload_onload():
    num_layers = 4
    blocks_in_primary_pool = 10240
    page_size = 32
    offload_chunksize = 1024

    with torch.inference_mode():
        async_kvcache_mgr = get_test_kvcache_mgr(
            num_layers, blocks_in_primary_pool, page_size, offload_chunksize
        )

        uid_min_limit = 0
        for batch_size, seq_len in [
            (3, 5000),
            (5, 10000),
            (6, 8000),
        ]:
            uids, kv_metadata = get_test_userids_and_metadata(
                uid_min_limit,
                torch.tensor([seq_len] * batch_size).int().cuda(),
                num_layers,
                async_kvcache_mgr.gpu_kvcache_mgr,
            )
            async_kvcache_mgr.offload_kvcache(kv_metadata)

            for layer_idx in range(num_layers):
                kv_metadata.kv_offload_handle.mark_ready(layer_idx)

            while async_kvcache_mgr.gpu_kvcache_mgr.is_busy_offloading():
                pass

            async_kvcache_mgr.static_onload_handle.reset()
            async_kvcache_mgr.gpu_kvcache_mgr.onload_kvcache(
                uids.tolist(), async_kvcache_mgr.static_onload_handle
            )

            for layer_idx in range(num_layers):
                async_kvcache_mgr.static_onload_handle.wait_host(layer_idx)

            # check data
            total_onload_pages = len(kv_metadata.offload_page_ids)
            origin_kvdata = async_kvcache_mgr.cache_table[
                :, kv_metadata.offload_page_ids, ...
            ]
            onload_kvdata = async_kvcache_mgr.cache_table[
                :,
                blocks_in_primary_pool : blocks_in_primary_pool + total_onload_pages,
                ...,
            ]
            assert torch.allclose(onload_kvdata, origin_kvdata)

            torch.cuda.synchronize()

            uid_min_limit += batch_size


@pytest.fixture(scope="module")
def suffix_release_kvcache_mgr():
    """Build a small native cache for suffix-release integration tests."""
    if not torch.cuda.is_available():
        pytest.skip("suffix-release integration tests require CUDA")

    manager = AsyncHSTUKVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        kv_headdim=64,
        num_tokens_per_page=32,
        num_primary_cache_pages=64,
        num_onload_buffer_pages=64,
        num_reserved_buffer_pages=0,
        num_tokens_per_chunk=1024,
        max_num_sequences=1,
        max_sequence_length=2048,
        max_batch_size=1,
        max_queued_offload_tokens=4096,
        num_onload_buffer_chunks=1,
        num_offload_buffer_chunks=1,
        num_memcpy_workers=4,
        enable_nvcomp=False,
    )
    yield manager
    manager.executor.shutdown(wait=True)
    manager.onload_worker.shutdown(wait=True)


def _prepare_suffix_release_request(manager, uid, total_history_length):
    """Run native page allocation/metadata preparation for one user."""
    gpu_mgr = manager.gpu_kvcache_mgr
    old_cached_length = int(gpu_mgr.get_total_cache_length([int(uid)])[0])
    offload_uids_buffer = torch.empty((1,), dtype=torch.int64)
    metadata_host_buffer = torch.empty((14,), dtype=torch.int32, pin_memory=True)

    paged_kvcache_ops.prepare_kvcache(
        gpu_mgr,
        manager.host_kv_mgr,
        [int(uid)],
        [int(total_history_length)],
        manager.static_page_ids_gpu_buffer,
        manager.static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        manager.static_metadata_gpu_buffer,
    )
    torch.cuda.synchronize()

    new_tokens = int(total_history_length) - old_cached_length
    metadata = manager.get_kvcache_metadata_from_buffer(
        1,
        new_tokens,
        manager.static_page_ids_gpu_buffer,
        manager.static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        manager.static_metadata_gpu_buffer,
        manager.static_onload_handle,
    )
    return metadata


def _append_suffix_release_test_data(manager, metadata, key_value):
    """Write deterministic K/V values through the public append kernel."""
    new_tokens = int(metadata.new_history_nnz)
    if new_tokens == 0:
        return None, None

    device = manager.cache_table.device
    keys = torch.full(
        (new_tokens, manager.num_heads, manager.head_dim),
        float(key_value),
        dtype=manager.cache_table.dtype,
        device=device,
    )
    values = torch.full_like(keys, -float(key_value))
    paged_k_cache, paged_v_cache = manager.cache_table_list[0].unbind(dim=1)
    paged_kvcache_ops.append_kvcache(
        keys,
        values,
        metadata.batch_indices,
        metadata.position,
        torch.tensor([0, new_tokens], dtype=torch.int32, device=device),
        metadata.new_history_nnz_cuda,
        new_tokens,
        paged_k_cache,
        paged_v_cache,
        metadata.kv_indices,
        metadata.kv_indptr,
        metadata.kv_last_page_len,
        0,
        1,
    )
    torch.cuda.synchronize()
    return keys, values


def _complete_suffix_release_test_offload(manager, metadata):
    accepted = manager.offload_kvcache(metadata)
    assert accepted
    for layer_idx in range(manager.num_layers):
        metadata.kv_offload_handle.mark_ready(layer_idx)

    deadline = time.monotonic() + 30.0
    while manager.gpu_kvcache_mgr.is_busy_offloading():
        if time.monotonic() >= deadline:
            pytest.fail("native KV offload did not complete within 30 seconds")
        time.sleep(0.01)


def test_kvcache_suffix_release_waits_for_offload_and_preserves_host_data(
    suffix_release_kvcache_mgr,
):
    manager = suffix_release_kvcache_mgr
    gpu_mgr = manager.gpu_kvcache_mgr
    uid = 201
    metadata = _prepare_suffix_release_request(manager, uid, 1024)
    _append_suffix_release_test_data(manager, metadata, key_value=1)

    torch.cuda.synchronize()
    assert manager.offload_kvcache(metadata)
    try:
        assert gpu_mgr.is_user_offload_frozen(uid)
        assert gpu_mgr.get_user_page_count(uid) == 32
        assert not gpu_mgr.evict_user_suffix_pages(uid, 32, 0, 1024)
        assert gpu_mgr.get_user_page_count(uid) == 32
    finally:
        for layer_idx in range(manager.num_layers):
            metadata.kv_offload_handle.mark_ready(layer_idx)

    deadline = time.monotonic() + 30.0
    while gpu_mgr.is_busy_offloading():
        if time.monotonic() >= deadline:
            pytest.fail("native KV offload did not complete within 30 seconds")
        time.sleep(0.01)

    assert not gpu_mgr.is_user_offload_frozen(uid)
    assert manager.host_kv_mgr.get_kvdata_length(uid) == 1024
    host_kv = manager.host_kv_mgr.get_kvdata_tensor([uid], True)[0]
    assert torch.all(host_kv[:, :, 0, ...] == 1)
    assert torch.all(host_kv[:, :, 1, ...] == -1)

    assert gpu_mgr.evict_user_suffix_pages(uid, 32, 0, 1024)
    assert gpu_mgr.get_user_page_count(uid) == 0
    assert gpu_mgr.get_total_cache_length([uid]) == [1024]


def test_kvcache_suffix_release_preserves_durable_host_prefix(
    suffix_release_kvcache_mgr,
):
    manager = suffix_release_kvcache_mgr
    gpu_mgr = manager.gpu_kvcache_mgr
    uid = 202

    first_metadata = _prepare_suffix_release_request(manager, uid, 1024)
    _append_suffix_release_test_data(manager, first_metadata, key_value=2)
    _complete_suffix_release_test_offload(manager, first_metadata)
    assert manager.host_kv_mgr.get_kvdata_length(uid) == 1024

    # Add a non-offloaded HBM suffix after the durable host prefix.
    _prepare_suffix_release_request(manager, uid, 2048)
    assert gpu_mgr.get_user_page_count(uid) == 64
    assert gpu_mgr.get_user_cache_interval(uid) == [0, 2048, 64]
    empty_pages_before = gpu_mgr.get_empty_page_count()

    # Keeping 31 pages would expose only 992 tokens, shorter than the
    # authoritative 1024-token host prefix. The native API must reject it
    # without changing either the page mapping or the free-page pool.
    assert not gpu_mgr.evict_user_suffix_pages(uid, 33, 0, 2048)
    assert gpu_mgr.get_user_page_count(uid) == 64
    assert gpu_mgr.get_empty_page_count() == empty_pages_before
    assert gpu_mgr.get_user_cache_interval(uid) == [0, 2048, 64]

    # Releasing exactly the non-host-backed suffix leaves a complete host
    # prefix resident in the logical cache length.
    assert gpu_mgr.evict_user_suffix_pages(uid, 32, 0, 2048)
    assert gpu_mgr.get_user_page_count(uid) == 32
    assert gpu_mgr.get_user_cache_interval(uid) == [0, 1024, 32]
    assert gpu_mgr.get_total_cache_length([uid]) == [1024]


def test_kvcache_suffix_release_then_append_after_host_prefix(
    suffix_release_kvcache_mgr,
):
    manager = suffix_release_kvcache_mgr
    gpu_mgr = manager.gpu_kvcache_mgr
    uid = 203

    initial_metadata = _prepare_suffix_release_request(manager, uid, 1024)
    _append_suffix_release_test_data(manager, initial_metadata, key_value=3)
    _complete_suffix_release_test_offload(manager, initial_metadata)

    # Drop the full GPU mapping while retaining its durable host prefix.
    assert gpu_mgr.evict_user_suffix_pages(uid, 32, 0, 1024)
    assert gpu_mgr.get_user_page_count(uid) == 0
    assert gpu_mgr.get_total_cache_length([uid]) == [1024]

    # The next request must onload the prefix and allocate a new page whose
    # token positions begin exactly at the old logical end.
    metadata = _prepare_suffix_release_request(manager, uid, 1056)
    manager.static_onload_handle.reset()
    gpu_mgr.onload_kvcache([uid], manager.static_onload_handle)
    for layer_idx in range(manager.num_layers):
        manager.static_onload_handle.wait_host(layer_idx)

    assert gpu_mgr.get_user_cache_interval(uid)[0:2] == [1024, 1056]
    assert gpu_mgr.get_user_page_count(uid) == 1
    assert gpu_mgr.get_total_cache_length([uid]) == [1056]
    assert metadata.new_history_nnz == 32
    assert metadata.kv_indptr.tolist() == [0, 33]
    assert metadata.kv_last_page_len.tolist() == [32]
    assert metadata.position.cpu().tolist() == list(range(1024, 1056))

    keys, values = _append_suffix_release_test_data(manager, metadata, key_value=7)
    page_indices = metadata.kv_indices.cpu().tolist()
    assert len(page_indices) == 33
    assert page_indices[:32] == list(range(64, 96))
    assert page_indices[32] < manager.num_primary_cache_pages

    # Verify both the host prefix onload and the append target on the new
    # primary page, so metadata-only success cannot mask a bad page mapping.
    assert torch.all(manager.cache_table[0, 64:96, 0, ...] == 3)
    assert torch.all(manager.cache_table[0, 64:96, 1, ...] == -3)
    append_page = page_indices[32]
    assert torch.equal(
        manager.cache_table[0, append_page, 0, 0, 0], keys[0, 0]
    )
    assert torch.equal(
        manager.cache_table[0, append_page, 1, 31, 0], values[31, 0]
    )


if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(0)
    test_kvcache_offload_onload()




