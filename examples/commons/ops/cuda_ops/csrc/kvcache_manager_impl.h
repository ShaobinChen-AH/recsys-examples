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
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <barrier>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "nvcomp/ans.h"

namespace kvcache {

// File: kvcache_manager_impl.h add in public section, before the class's private section

enum class TransferDirection { ONLOAD = 0, OFFLOAD = 1 };

struct TransferCommand {
    int64_t uid;
    TransferDirection direction;
    int stream_group;       // 0=critical, 1=prefetch, 2=background
    float priority;         // 0.0 to 1.0, higher = more urgent
    int num_pages;
    int transfer_id;        // assigned by submit_transfer, returned to Python
    int submit_epoch;       // epoch when submitted (for aging/staleness)

    // Priority comparator: higher priority pops first in max-heap.
    // std::priority_queue uses operator< for ordering (largest comes out first
    // when using default std::less), so we invert: lower numeric value = higher
    // priority pops first.
    bool operator<(const TransferCommand& other) const {
        if (stream_group != other.stream_group) {
            return stream_group > other.stream_group;  // group 0 pops before group 1
        }
        if (priority != other.priority) {
            return priority < other.priority;          // higher priority pops first
        }
        return submit_epoch > other.submit_epoch;      // older (stale) pops last
    }
};

// File: kvcache_manager_impl.h add in private section

struct StreamGroup {
    int group_id;                                    // 0=critical, 1=prefetch, 2=background
    std::vector<cudaStream_t> streams;               // pool of CUDA streams
    std::priority_queue<TransferCommand> cmd_queue;  // ordered by TransferCommand::operator<
    std::unordered_map<int, TransferCommand> in_flight;  // stream_index active command
    std::queue<int> idle_streams;                    // indices of free streams
    int num_streams;

    StreamGroup(int gid, int count) : group_id(gid), num_streams(count) {
        streams.reserve(count);
        for (int i = 0; i < count; i++) {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            streams.push_back(s);
            idle_streams.push(i);
        }
    }

    ~StreamGroup() {
        for (int i = 0; i < num_streams; i++) {
            if (in_flight.count(i)) {
                cudaStreamSynchronize(streams[i]);
            }
            cudaStreamDestroy(streams[i]);
        }
    }

    bool has_idle_stream() const { return !idle_streams.empty(); }

    int acquire_stream(const TransferCommand& cmd) {
        int idx = idle_streams.front();
        idle_streams.pop();
        in_flight[idx] = cmd;
        return idx;
    }

    int release_stream(int idx);

    cudaStream_t stream(int idx) { return streams[idx]; }
};


class MultithreadMemcpyProcessor
{
public:
    MultithreadMemcpyProcessor(int num_workers);
    ~MultithreadMemcpyProcessor();

    inline const size_t num_workers() const;
    void memcpy(void* dst, void* src, size_t bytes, size_t bytes_part);

private:
    void memcpy_coworker_loop(const int idx);

private:
    int num_workers_;
    std::vector<std::thread> workers_;
    std::barrier<> start_barrier_;
    std::barrier<> end_barrier_;
    char *dst_;
    char *src_;
    size_t localbytes_;
    bool terminate_;
};

class PinnedDoubleBuffer {
public:
    PinnedDoubleBuffer(size_t buffer_bytes);
    ~PinnedDoubleBuffer();

public:
    std::vector<char *> ptr_;
    std::vector<cudaEvent_t> cuda_event_;
};

class KVCompressor {
public:
    KVCompressor(int max_num_chunks, size_t chunk_numel, size_t chunk_bytes);
    ~KVCompressor();

    void set_compress_input_buffer_ptrs(char *base_ptr, size_t num_chunks);  // call once
    void set_decompress_output_buffer_ptrs(char *base_ptr, size_t num_chunks, cudaStream_t stream);  // call multiples

    void compress(
        size_t *compressed_bytes_cpu,
        size_t num_chunks,
        cudaStream_t stream);
    void decompress(
        size_t *compressed_bytes_cpu,
        size_t num_chunks,
        cudaStream_t stream);

public:
    char *comp_out_buffer();
    char *comp_out_buffer(int index);

    char *decomp_in_buffer();
    char *decomp_in_buffer(int index);

private:
    int max_num_chunks_;
    size_t chunk_numel_;
    size_t chunk_bytes_;

    size_t max_comp_chunk_bytes_;

    char *comp_out_buffer_;
    void **comp_in_ptrs_;      // setup once
    size_t *comp_in_bytes_;    // setup once
    void **comp_out_ptrs_;     // to internal
    size_t *comp_out_bytes_;   // output
    // size_t *comp_out_bytes_cpu_;   // output

    char *decomp_in_buffer_;
    void **decomp_in_ptrs_;      // to internal
    size_t *decomp_in_bytes_;    // setup multiple times
    // size_t *decomp_in_bytes_cpu_;    // setup multiple times
    void **decomp_out_ptrs_;     // setup multiple times
    size_t *decomp_out_bytes_;   // may ignore
    size_t *decomp_buffer_bytes_;  // setup internal


    void *comp_tmp_buffer_;
    size_t comp_tmp_bytes_;
    void *decomp_tmp_buffer_;
    size_t decomp_tmp_bytes_;
    
    nvcompStatus_t *comp_status_;
    nvcompStatus_t *comp_status_cpu_;
    nvcompStatus_t *decomp_status_;
    nvcompStatus_t *decomp_status_cpu_;

    const nvcompBatchedANSOpts_t k_comp_opts_ = {nvcomp_rANS, uint8};
    const nvcompBatchedANSOpts_t k_decomp_opts_ = nvcompBatchedANSDefaultOpts;
};

class GPUKVCacheMangerImpl;
class HostKVStorageImpl;

class KVOnloadHandle {
public:
    KVOnloadHandle();
    KVOnloadHandle(int num_layers);
    ~KVOnloadHandle() ;

    void reset();
    void complete_host(int layer_idx);
    void complete_host(int layer_idx, cudaStream_t stream);
    void wait_host(int layer_idx);

public:
    int num_layers;
    std::vector<cudaEvent_t> event;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<int> host_complete;
    bool no_onload;
};

class KVOffloadHandle {
public:
    KVOffloadHandle();
    KVOffloadHandle(
        int num_layers,
        GPUKVCacheMangerImpl& gpu_kv_mgr,
        bool has_offload
    );

    void mark_ready(int layer_idx);
    void set_no_offload();

public:
    GPUKVCacheMangerImpl* gpu_kv_mgr;
    int num_layers;
    std::vector<cudaEvent_t> ready_event;
    int *host_ready;
    bool no_offload;
};

class HostKVStorageImpl
{
public:
    HostKVStorageImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int64_t num_tokens_per_chunk
    );
    ~HostKVStorageImpl();

    int64_t get_kvdata_length(int64_t user_id);

    void append_kvdata(
        int64_t user_id, int64_t start_position, int64_t length, 
        uint16_t *kvdata_buffer, size_t buffer_layer_stride);
    void append_kvdata(
        int64_t user_id, int64_t start_position, int64_t length, 
        uint16_t *kvdata_buffer, size_t buffer_layer_stride,
        size_t *kvdata_bytes, size_t bytes_layer_stride);

    std::vector<uint16_t*> get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx);
    std::vector<size_t> get_kvdata_bytes(int64_t user_id, int64_t length, int64_t layer_idx);

public:
    std::vector<at::Tensor> get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat = true);
    void init_random_kvdata(int64_t user_id, size_t num_tokens);

public:
    const int num_layers;
    const int num_kv_heads;
    const int kv_headdim;
    const int page_size;

    const int64_t chunk_size;
    size_t chunk_numel;
    size_t page_numel;
    size_t per_token_numel;
    size_t layer_numel;

    std::vector<std::unordered_map<int64_t, std::vector<uintptr_t>>> _uid_to_chunk_id;
    std::vector<std::unordered_map<int64_t, std::vector<size_t>>> _uid_to_chunk_bytes;
    std::unordered_map<int64_t, int64_t> _uid_to_length;
    std::unordered_map<int64_t, std::vector<uintptr_t>> _uid_to_mempool;
    std::mutex host_kvcache_mutex_;
};

class GPUKVCacheMangerImpl
{
public:
    GPUKVCacheMangerImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int num_primary_cache_pages,
        int num_onload_buffer_pages,
        int num_reserved_buffer_pages,
        int num_tokens_per_chunk,
        int max_num_sequences,
        int max_sequence_length,
        at::Tensor cache_table_tensor,
        HostKVStorageImpl& host_kv_mgr,
        size_t max_queued_offload_tokens,
        int onload_buffer_chunks = 1,
        int offload_buffer_chunks = 8,
        int num_memcpy_workers = 4,
        bool enable_nvcomp = false);
    ~GPUKVCacheMangerImpl();

    int64_t getUIdToEvict(std::unordered_set<int64_t> extra_freezed_uids);

    std::vector<int32_t>& alloc(int64_t uid, int new_total_length, std::unordered_set<int64_t> freezed_uids);
    std::vector<int32_t> get_total_cache_length(std::vector<int64_t>& uids);
    
    void evict(int64_t uid);
    void evict_all();
    void invalid(int64_t uid);
    bool retain(int64_t uid);

    uint16_t *get_cache_table(void);
    uint16_t *get_cache_table_by_layer(int layer_idx);

public:
    int submit_transfer(int64_t uid, int direction, int stream_group, float priority, int num_pages);
    bool is_transfer_complete(int transfer_id);

    void cancel_transfers_for_user(int64_t uid);

    int pending_in_group(int stream_group);

    void set_current_epoch(int epoch);

private:
    std::atomic<int> _next_transfer_id{1};
    int _current_epoch{0};
    std::mutex _transfer_mutex;
    std::condition_variable _transfer_cv;

    std::unordered_set<int> _completed_transfers;
    std::mutex _completed_mutex;

    void execute_transfer(const TransferCommand& cmd, int page_offset,
                      int num_pages, cudaStream_t stream);

public:
    void onload_kvcache(
        std::vector<int64_t>& user_ids, 
        KVOnloadHandle& onloadhandle);

    void offload_kvcache(
        KVOffloadHandle& offload_handle,
        at::Tensor offload_user_ids,      // host -> make static
        at::Tensor offload_page_ids,      // gpu -> make static
        at::Tensor new_offload_startpos,  // host
        at::Tensor new_offload_lengths);  // host

    bool is_busy_offloading();

public:
    void init_random_offload_status(int64_t user_id, size_t length);

private:
    void offload_loop();

public:
    int num_layers;
    int num_kv_heads;
    int kv_headdim;
    int num_tokens_per_page;
    int num_primary_cache_pages;
    int num_onload_buffer_pages;
    int num_reserved_buffer_pages;
    int num_tokens_per_chunk;
    int max_num_sequences;
    int max_sequence_length;

    size_t layer_stride;
    size_t k2v_stride;
    size_t page_stride;
    size_t per_token_kv_stride;

public:
    // kvcache bookkeeping
    std::list<int64_t> _lru_list;
    std::unordered_map<int64_t, 
                       typename std::list<int64_t>::iterator> _lru_lookup_table;
    std::queue<int64_t> _empty_pages;
    std::unordered_map<int64_t, std::vector<int32_t>> _uid_to_page_id;
    std::unordered_map<int64_t, int32_t> _uid_to_paged_cache_startpos;
    std::unordered_map<int64_t, int32_t> _uid_to_paged_cache_length;
    std::unordered_map<int64_t, int32_t> _uid_to_offloaded_length;

    // threadpool
    std::thread offload_worker;
    bool terminate_;
    std::atomic<bool> offload_busy_;

    // offloading shared objects
    std::queue<std::tuple<std::vector<int>, at::Tensor, std::vector<cudaEvent_t>, int*>> offload_task_queue;
    std::mutex offload_task_mutex_;
    std::condition_variable offload_task_cv_;

    // offloading limiter
    std::unordered_map<int64_t, int> queued_offload_lastpos;
    size_t queued_offload_tokens;
    std::mutex queued_offload_lastpos_mutex_;
    size_t queued_offload_limits;

    // internal device buffer
    uint16_t* onload_device_buffers;
    int num_onload_device_chunks;
    uint16_t* offload_device_buffers;
    int num_offload_device_chunks;

    // external offloading synchronization
    std::mutex offload_ready_mtx_;
    std::condition_variable offload_ready_cv_;

    // allocation-vs-offloading synchronization
    std::unordered_map<int64_t, int> offload_freezed_uids_;
    std::mutex offload_freezed_uids_mtx_;

    bool enable_nvcomp;
    KVCompressor compressor;

    cudaStream_t worker_stream;
    std::vector<StreamGroup> _stream_groups;

    PinnedDoubleBuffer onload_pin_buffer;
    PinnedDoubleBuffer offload_pin_buffer;

    MultithreadMemcpyProcessor onload_memcpy_workers;
    MultithreadMemcpyProcessor offload_memcpy_workers;

    HostKVStorageImpl *host_kv_mgr;

public:
    uint16_t *cache_table;
    c10::Device device;
public:
    int get_empty_page_count() const { return (int)_empty_pages.size(); }
    int get_user_page_count(int64_t uid);
    void set_active_page_limit(int new_limit);
private:
    std::queue<int64_t> _withheld_pages;   // pages temporarily removed from circulation
    int _active_page_limit;                 // current effective page budget
};

void prepare_kvcache(
    GPUKVCacheMangerImpl& gpu_mgr,
    HostKVStorageImpl& host_mgr,
    std::vector<int64_t>& user_ids,
    std::vector<int64_t>& total_hist_lens, // all histo w/o candi
    at::Tensor page_ids_gpu_buffer,
    at::Tensor offload_page_ids_gpu_buffer,
    at::Tensor offload_uids_buffer,
    at::Tensor metadata_host_buffer,
    at::Tensor metadata_gpu_buffer);

}  // namespace kvcache
