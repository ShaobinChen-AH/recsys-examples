import torch, sys, os
sys.path.insert(0, "/workspace/recsys-examples/examples/commons")
sys.path.insert(0, "/workspace/recsys-examples/examples/hstu")
sys.path.insert(0, "/workspace/recsys-examples/examples")

from configs import get_inference_hstu_config, get_kvcache_config, InferenceEmbeddingConfig, RankingConfig, KVCacheMetadata
from modules.inference_dense_module import InferenceDenseModule
from modules.jagged_data import JaggedData
import paged_kvcache_ops

B, L = 1, 128
HDIM, NHEADS, NLAYERS = 128, 4, 3

hstu_config = get_inference_hstu_config(
    hidden_size=HDIM, num_layers=NLAYERS, num_attention_heads=NHEADS,
    head_dim=HDIM, max_batch_size=1, max_seq_len=2048, dtype=torch.bfloat16,
)
kv_cache_config = get_kvcache_config(
    blocks_in_primary_pool=256, page_size=32, offload_chunksize=8192,
)
emb_configs = [
    InferenceEmbeddingConfig(
        feature_names=["item_feat"], table_name="item",
        vocab_size=10000, dim=HDIM, use_dynamicemb=True,
    ),
]
task_config = RankingConfig(
    embedding_configs=emb_configs, prediction_head_arch=[128, 8], num_tasks=1,
)

# ------------------------------------------------------------
# Test 1: HSTU forward WITHOUT KV cache
# ------------------------------------------------------------
print("Test 1: No KV cache...")
with torch.inference_mode():
    model = InferenceDenseModule(hstu_config, None, task_config, use_cudagraph=False)
    model.bfloat16().eval()
    hidden = torch.randn(B * L, HDIM, dtype=torch.bfloat16, device="cuda")
    jd = JaggedData(
        values=hidden,
        seqlen=torch.tensor([L], device="cuda", dtype=torch.int32),
        seqlen_offsets=torch.tensor([0, L], device="cuda", dtype=torch.int32),
        max_seqlen=2048, max_num_candidates=100,
        num_candidates=torch.tensor([0], device="cuda", dtype=torch.int32),
        num_candidates_offsets=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        contextual_max_seqlen=0, has_interleaved_action=True,
    )
    out = model._hstu_block.predict(B, B * L, hidden, jd, None)
    print("  OK")

# ------------------------------------------------------------
# Test 2: HSTU forward WITH KV cache (minimal page alloc)
# ------------------------------------------------------------
print("Test 2: With KV cache...")
with torch.inference_mode():
    model2 = InferenceDenseModule(hstu_config, kv_cache_config, task_config, use_cudagraph=False)
    model2.bfloat16().eval()

    hidden2 = torch.randn(B * L, HDIM, dtype=torch.bfloat16, device="cuda")
    jd2 = JaggedData(
        values=hidden2,
        seqlen=torch.tensor([L], device="cuda", dtype=torch.int32),
        seqlen_offsets=torch.tensor([0, L], device="cuda", dtype=torch.int32),
        max_seqlen=2048, max_num_candidates=100,
        num_candidates=torch.tensor([0], device="cuda", dtype=torch.int32),
        num_candidates_offsets=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        contextual_max_seqlen=0, has_interleaved_action=True,
    )

    page_size = 32
    cache_table = model2.async_kvcache.cache_table_list
    kv_metadata = KVCacheMetadata(
        kv_indices=torch.arange(B, device="cuda", dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        kv_last_page_len=torch.tensor([page_size], device="cuda", dtype=torch.int32),
        batch_indices=torch.arange(B * L, device="cuda", dtype=torch.int32),
        position=torch.arange(L, device="cuda", dtype=torch.int32),
        new_history_nnz=L,
        new_history_nnz_cuda=torch.tensor([L], device="cuda", dtype=torch.int32),
        total_history_lengths=torch.tensor([0], device="cuda", dtype=torch.int32),
        total_history_offsets=torch.tensor([0, 0], device="cuda", dtype=torch.int32),
        kv_cache_table=cache_table,
        kv_onload_handle=paged_kvcache_ops.KVOnloadHandle(),
        kv_offload_handle=paged_kvcache_ops.KVOffloadHandle(),
    )

    out2 = model2._hstu_block.predict(B, B * L, hidden2, jd2, kv_metadata)
    print("  OK")

print("All tests passed!")
