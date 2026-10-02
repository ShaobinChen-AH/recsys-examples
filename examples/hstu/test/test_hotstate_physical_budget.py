import math
import gc
import sys
from pathlib import Path

import pytest
import torch


HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))
EXAMPLES_ROOT = HSTU_ROOT.parent
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from modules.hotstate.physical_budget import (  # noqa: E402
    DEFAULT_SAFETY_MARGIN_BYTES,
    embedding_row_bytes,
    kv_copy_buffer_bytes,
    kv_onload_page_overhead,
    kv_page_bytes,
    max_blocks_in_primary_pool,
    plan_physical_budget,
    safety_margin_bytes,
)


def test_embedding_row_bytes():
    assert embedding_row_bytes(512, element_bytes=2) == 1024
    assert embedding_row_bytes(128, element_bytes=4, optimizer_state_bytes=4) == 1024


def test_exact_kv_page_bytes():
    assert kv_page_bytes(3, 4, 128, 32, element_bytes=2) == 3 * 2 * 32 * 4 * 128 * 2


def test_onload_and_copy_buffer_accounting():
    page = kv_page_bytes(3, 4, 128, 32)
    assert kv_onload_page_overhead(3, 4, 128, 32, 5) == 5 * page
    chunk = 8192 * 2 * 4 * 128 * 2
    assert kv_copy_buffer_bytes(3, 4, 128, 8192, 1, 8) == 9 * chunk


def test_safety_margin_uses_floor_and_ten_percent():
    assert safety_margin_bytes(1 * 1024**3) == math.ceil(0.10 * 1024**3)
    assert safety_margin_bytes(1) == DEFAULT_SAFETY_MARGIN_BYTES


def test_page_count_rounds_down_after_fixed_overhead():
    assert max_blocks_in_primary_pool(100, 32, 9, 1) == 2


def test_budget_split_and_plan_rounding():
    plan = plan_physical_budget(
        state_budget_bytes=512 * 1024**2,
        embedding_hbm_ratio=0.50,
        num_layers=3,
        num_kv_heads=4,
        head_dim=128,
        num_tokens_per_page=32,
        max_batch_size=1,
        max_sequence_length=4296,
        num_tokens_per_chunk=8192,
    )
    assert plan.embedding_hbm_bytes == (512 * 1024**2 - plan.safety_margin_bytes) // 2
    assert plan.blocks_in_primary_pool >= 1
    assert plan.planned_state_bytes <= plan.state_budget_bytes
    assert plan.kv_onload_pages == math.ceil(4296 / 32)


def test_insufficient_budget_rejected():
    with pytest.raises(ValueError, match="cannot fit"):
        plan_physical_budget(
            state_budget_bytes=65 * 1024**2,
            embedding_hbm_ratio=0.50,
            num_layers=3,
            num_kv_heads=4,
            head_dim=128,
            num_tokens_per_page=32,
            max_batch_size=1,
            max_sequence_length=4096,
            num_tokens_per_chunk=8192,
        )


def test_budget_smaller_than_safety_margin_rejected():
    with pytest.raises(ValueError, match="safety margin"):
        plan_physical_budget(
            state_budget_bytes=32 * 1024**2,
            embedding_hbm_ratio=0.50,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            num_tokens_per_page=32,
            max_batch_size=1,
            max_sequence_length=128,
            num_tokens_per_chunk=1024,
        )


def test_negative_buffer_counts_rejected():
    with pytest.raises(ValueError, match="chunk counts"):
        plan_physical_budget(
            state_budget_bytes=128 * 1024**2,
            embedding_hbm_ratio=0.50,
            num_layers=1,
            num_kv_heads=1,
            head_dim=64,
            num_tokens_per_page=32,
            max_batch_size=1,
            max_sequence_length=128,
            num_tokens_per_chunk=1024,
            num_onload_buffer_chunks=-1,
        )


def _build_cuda_smoke_model(dynamic_embedding_hbm_bytes: int, blocks: int):
    from configs import InferenceEmbeddingConfig, RankingConfig
    from configs import get_inference_hstu_config, get_kvcache_config
    from model.inference_ranking_gr import get_inference_ranking_gr

    hidden_dim = 64
    max_seq_len = 136
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim,
        num_layers=1,
        num_attention_heads=1,
        head_dim=hidden_dim,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
        scaling_seqlen=max_seq_len,
    )
    kv_config = get_kvcache_config(
        blocks_in_primary_pool=blocks,
        page_size=32,
        offload_chunksize=1024,
    )
    task_config = RankingConfig(
        embedding_configs=[
            InferenceEmbeddingConfig(
                feature_names=["item_feat"],
                table_name="item",
                vocab_size=1_000_000,
                dim=hidden_dim,
                use_dynamicemb=True,
            ),
            InferenceEmbeddingConfig(
                feature_names=["act_feat"],
                table_name="act",
                vocab_size=32,
                dim=hidden_dim,
                use_dynamicemb=False,
            ),
        ],
        prediction_head_arch=[32, 1],
        num_tasks=1,
    )
    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_config,
        task_config=task_config,
        use_cudagraph=False,
        dynamic_embedding_hbm_bytes=dynamic_embedding_hbm_bytes,
    )
    model.bfloat16()
    model.eval()
    return model


def _shutdown_cuda_smoke_model(model):
    if model is None:
        return
    try:
        kvcache = model.dense_module.async_kvcache
        kvcache.executor.shutdown(wait=True, cancel_futures=True)
        kvcache.onload_worker.shutdown(wait=True, cancel_futures=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_physical_hbm_envelope_smoke():
    """Exercise real DynamicEmb HBM and KV allocations for 20 inference batches."""
    try:
        import dynamicemb  # noqa: F401
        import paged_kvcache_ops  # noqa: F401
        from commons.datasets.hstu_batch import FeatureConfig
        from commons.datasets.random_inference_dataset import RandomInferenceDataset
        from configs import InferenceEmbeddingConfig, RankingConfig
        from configs import get_inference_hstu_config, get_kvcache_config
        from model.inference_ranking_gr import get_inference_ranking_gr
        from modules.hotstate.embedding_adapter import EmbeddingAdapter
        from modules.hotstate.kv_adapter import KVAdapter
        from modules.hotstate.physical_budget import plan_physical_budget
    except Exception as exc:
        pytest.skip(f"CUDA inference dependencies are unavailable: {exc}")

    state_budget = 128 * 1024**2
    plan = plan_physical_budget(
        state_budget_bytes=state_budget,
        embedding_hbm_ratio=0.50,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_tokens_per_page=32,
        max_batch_size=1,
        max_sequence_length=136,
        num_tokens_per_chunk=1024,
    )
    assert plan.embedding_hbm_bytes > 0
    assert plan.blocks_in_primary_pool > 0

    feature_configs = [
        FeatureConfig(
            feature_names=["item_feat", "act_feat"],
            max_item_ids=[999_999, 31],
            max_sequence_length=136,
            is_jagged=False,
        )
    ]
    torch.manual_seed(17)
    dataset = RandomInferenceDataset(
        feature_configs=feature_configs,
        item_feature_name="item_feat",
        contextual_feature_names=[],
        action_feature_name="act_feat",
        max_num_users=20,
        max_batch_size=1,
        max_history_length=64,
        max_num_candidates=8,
        max_incremental_seqlen=16,
        max_num_cached_batches=20,
        full_mode=True,
    )
    requests = list(iter(dataset))

    hbm_model = None
    host_model = None
    try:
        torch.cuda.manual_seed_all(23)
        torch.manual_seed(23)
        hbm_model = _build_cuda_smoke_model(plan.embedding_hbm_bytes, plan.blocks_in_primary_pool)
        hbm_model.dense_module.enable_hotstate(
            total_hbm_bytes=state_budget,
            configured_state_budget_bytes=state_budget,
        )
        hbm_model.dense_module.set_hotstate_embedding_module(hbm_model.sparse_module)
        physical_snapshot = hbm_model.dense_module.hotstate.validate_physical_budget()
        embedding_adapter = EmbeddingAdapter(hbm_model.sparse_module)
        kv_adapter = KVAdapter(hbm_model.dense_module.async_kvcache)
        embedding_bytes = embedding_adapter.physical_hbm_bytes()
        kv_bytes = kv_adapter.physical_hbm_bytes()
        assert embedding_bytes > 0
        assert kv_bytes > 0
        assert embedding_bytes + kv_bytes <= state_budget
        assert physical_snapshot["managed_state_physical_hbm_bytes"] <= state_budget

        # Reset the seed so the host-only control model has the same learned
        # parameters and DynamicEmb initializer sequence.
        torch.cuda.manual_seed_all(23)
        torch.manual_seed(23)
        host_model = _build_cuda_smoke_model(0, plan.blocks_in_primary_pool)

        for index in range(20):
            batch, user_ids, history_lengths = requests[index % len(requests)]
            hbm_model.dense_module.hotstate.before_batch(
                batch, user_ids, history_lengths, batch_idx=index
            )
            with torch.inference_mode():
                hbm_output = hbm_model.forward_with_kvcache(
                    batch, user_ids, history_lengths
                )
                host_output = host_model.forward_with_kvcache(
                    batch, user_ids, history_lengths
                )
            assert torch.allclose(hbm_output, host_output, rtol=2e-2, atol=2e-2)
            measured_embedding = embedding_adapter.physical_hbm_bytes()
            measured_kv = kv_adapter.physical_hbm_bytes()
            assert measured_embedding + measured_kv <= state_budget
    finally:
        _shutdown_cuda_smoke_model(host_model)
        _shutdown_cuda_smoke_model(hbm_model)
