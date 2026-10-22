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
from modules.hotstate.hotstate_controller import HotStateController  # noqa: E402


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


def test_controller_reports_residency_separately_from_fixed_pool_capacity():
    class EmbeddingAdapterStub:
        def physical_hbm_bytes(self):
            return 120

        def logical_hbm_resident_bytes(self):
            return 115

        def hbm_value_capacity_bytes(self):
            return 100

    class KVAdapterStub:
        def physical_hbm_bytes(self):
            return 150

        def actual_resident_kv_bytes(self):
            return 25

        def physical_kv_cache_bytes(self):
            return 200

        def logical_kv_budget_bytes(self):
            return 30

    controller = object.__new__(HotStateController)
    controller.emb_adapter = EmbeddingAdapterStub()
    controller.kv_adapter = KVAdapterStub()
    controller.configured_state_budget_bytes = 400
    controller._embedding_limit_violations = 0
    controller._kv_limit_violations = 0
    controller._total_residency_checks = 0
    controller._residency_limit_history = []
    controller._residency_history_size = 20

    occupancy = controller.get_current_occupancy()
    assert occupancy["embedding_physical_hbm_bytes"] == 120
    assert occupancy["embedding_resident_bytes"] == 115
    assert occupancy["embedding_pool_capacity_bytes"] == 100
    assert occupancy["embedding_utilization_ratio"] == pytest.approx(1.15)
    assert occupancy["kv_physical_hbm_bytes"] == 150
    assert occupancy["kv_resident_bytes"] == 25
    assert occupancy["kv_pool_capacity_bytes"] == 200
    assert occupancy["kv_residency_limit_bytes"] == 30

    drift = controller.detect_residency_limit_drift()
    assert drift["embedding_actual_bytes"] == 115
    assert drift["embedding_limit_bytes"] == 100
    assert drift["kv_actual_bytes"] == 25
    assert drift["kv_limit_bytes"] == 30
    assert drift["has_embedding_violation"] is True
    assert drift["recommendation"] == "reduce_embedding_residency_in_fixed_pool"


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


def _use_smoke_eval_initializer(model, initialized_rows):
    """Give independently constructed DynamicEmb tables identical nonzero misses."""
    tables = model.sparse_module._dynamic_embedding_collection._embedding_tables

    def initialize(values, indices, keys):
        missing_indices = indices.to(dtype=torch.long)
        missing_keys = keys.index_select(0, missing_indices).to(dtype=torch.int64)
        row_values = (
            missing_keys.remainder(251).to(dtype=values.dtype) / 251.0 + 0.125
        )
        initialized_values = row_values.unsqueeze(1).expand(-1, values.size(1))
        values.index_copy_(0, missing_indices, initialized_values)
        for key, row in zip(
            missing_keys.detach().cpu().tolist(), initialized_values.detach().cpu()
        ):
            key = int(key)
            row = row.clone()
            previous = initialized_rows.get(key)
            if previous is not None:
                assert torch.equal(previous, row), (
                    f"key {key} was initialized with different values"
                )
            initialized_rows[key] = row

    tables._eval_initializers[0] = initialize


def _read_dynamic_embedding_rows(model, item_ids):
    from dynamicemb.types import CopyMode

    tables = model.sparse_module._dynamic_embedding_collection._embedding_tables
    storage = tables._storage
    state = getattr(storage, "_hbm", None)
    if state is None:
        state = storage._state
    keys = torch.as_tensor(item_ids, dtype=torch.int64, device=state.device)
    table_ids = torch.zeros_like(keys)
    result = storage.find(keys, table_ids, copy_mode=CopyMode.EMBEDDING)
    assert bool(result[5].all()), "requested DynamicEmb rows are missing after lookup"
    return result[-1]


def _capture_sparse_outputs(captured):
    def capture(_module, _inputs, outputs):
        captured.clear()
        captured.update(
            {
                name: (
                    embedding.values().detach().clone(),
                    embedding.lengths().detach().clone(),
                )
                for name, embedding in outputs.items()
            }
        )

    return capture


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
    hbm_sparse_hook = None
    host_sparse_hook = None
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
        # parameters.
        torch.cuda.manual_seed_all(23)
        torch.manual_seed(23)
        host_model = _build_cuda_smoke_model(0, plan.blocks_in_primary_pool)
        hbm_initialized_rows = {}
        host_initialized_rows = {}
        _use_smoke_eval_initializer(hbm_model, hbm_initialized_rows)
        _use_smoke_eval_initializer(host_model, host_initialized_rows)
        hbm_model.dense_module.async_kvcache.capture_last_kv_metadata = True
        host_model.dense_module.async_kvcache.capture_last_kv_metadata = True
        hbm_sparse_outputs = {}
        host_sparse_outputs = {}
        hbm_sparse_hook = hbm_model.sparse_module.register_forward_hook(
            _capture_sparse_outputs(hbm_sparse_outputs)
        )
        host_sparse_hook = host_model.sparse_module.register_forward_hook(
            _capture_sparse_outputs(host_sparse_outputs)
        )

        host_model.dense_module.load_state_dict(
            hbm_model.dense_module.state_dict(),
            strict=True,
        )
        with torch.no_grad():
            hbm_layers = hbm_model.dense_module._hstu_block._attention_layers
            host_layers = host_model.dense_module._hstu_block._attention_layers

            for hbm_layer, host_layer in zip(hbm_layers, host_layers):
                host_layer._linear_uvqk_weight.copy_(
                    hbm_layer._linear_uvqk_weight
                )
                host_layer._linear_proj_weight.copy_(
                    hbm_layer._linear_proj_weight
                )

        torch.cuda.synchronize()
        host_model.sparse_module._static_embedding_collection.load_state_dict(
            hbm_model.sparse_module._static_embedding_collection.state_dict(),
            strict=True,
        )
        with torch.no_grad():
            hbm_layers = hbm_model.dense_module._hstu_block._attention_layers
            host_layers = host_model.dense_module._hstu_block._attention_layers

            for hbm_layer, host_layer in zip(hbm_layers, host_layers):
                host_layer._linear_uvqk_weight.copy_(
                    hbm_layer._linear_uvqk_weight
                )
                host_layer._linear_proj_weight.copy_(
                    hbm_layer._linear_proj_weight
                )

        torch.cuda.synchronize()

        def max_diff(a, b):
            return float((a.float() - b.float()).abs().max().item())

        for index in range(20):
            batch, user_ids, history_lengths = requests[index % len(requests)]
            before = hbm_model.dense_module.hotstate.before_batch(
                batch, user_ids, history_lengths, batch_idx=index
            )
            if index == 0:
                requested_keys = set(
                    embedding_adapter.batch_item_sequence(batch)
                )
                admitted_keys = {
                    int(key.rsplit(":", 1)[-1])
                    for key in before["physical_admitted_keys"]
                    if str(key).startswith("emb:item:")
                }
                assert admitted_keys, (
                    "first batch produced no physical DynamicEmb admission: "
                    f"{before['transfer_execution']}"
                )
                assert admitted_keys <= requested_keys
                residency = embedding_adapter._observed_residency(
                    sorted(admitted_keys)
                )
                assert residency is not None
                assert all(location == 1 for location in residency)
                with torch.inference_mode():
                    hbm_sparse = hbm_model.sparse_module(batch.features)
                    host_sparse = host_model.sparse_module(batch.features)
                    hbm_dense_state = hbm_model.dense_module.state_dict()
                    host_dense_state = host_model.dense_module.state_dict()
                    assert hbm_dense_state.keys() == host_dense_state.keys()
                    for state_name, hbm_state in hbm_dense_state.items():
                        host_state = host_dense_state[state_name]
                        if isinstance(hbm_state, torch.Tensor):
                            assert torch.equal(hbm_state, host_state), (
                                f"batch=0 dense state differs at {state_name}"
                            )

                    hbm_preprocessed = (
                        hbm_model.dense_module._hstu_block._preprocessor(
                            embeddings=hbm_sparse, batch=batch
                        )
                    )
                    host_preprocessed = (
                        host_model.dense_module._hstu_block._preprocessor(
                            embeddings=host_sparse, batch=batch
                        )
                    )
                    assert torch.equal(
                        hbm_preprocessed.values, host_preprocessed.values
                    ), (
                        "batch=0 preprocessed HSTU inputs differ: "
                        f"max_diff={max_diff(hbm_preprocessed.values, host_preprocessed.values)}"
                    )
                    for metadata_name in (
                        "seqlen",
                        "seqlen_offsets",
                        "num_candidates",
                        "num_candidates_offsets",
                    ):
                        hbm_metadata = getattr(hbm_preprocessed, metadata_name)
                        host_metadata = getattr(host_preprocessed, metadata_name)
                        if hbm_metadata is not None:
                            assert torch.equal(hbm_metadata, host_metadata), (
                                f"batch=0 preprocessed {metadata_name} differs"
                            )
                    hbm_uncached_output = hbm_model.dense_module.forward_nokvcache(
                        batch, hbm_sparse
                    )
                    host_uncached_output = host_model.dense_module.forward_nokvcache(
                        batch, host_sparse
                    )
                assert hbm_sparse_outputs.keys() == host_sparse_outputs.keys()
                for feature_name in hbm_sparse_outputs:
                    hbm_values, hbm_lengths = hbm_sparse_outputs[feature_name]
                    host_values, host_lengths = host_sparse_outputs[feature_name]
                    assert torch.equal(hbm_lengths, host_lengths), (
                        f"batch=0 sparse lengths differ for {feature_name}"
                    )
                    assert torch.equal(hbm_values, host_values), (
                        f"batch=0 sparse outputs differ for {feature_name}: "
                        f"max_diff={max_diff(hbm_values, host_values)}"
                    )
                uncached_difference = max_diff(
                    hbm_uncached_output, host_uncached_output
                )
                assert torch.allclose(
                    hbm_uncached_output,
                    host_uncached_output,
                    rtol=2e-2,
                    atol=2e-2,
                ), (
                    "batch=0 uncached model outputs differ: "
                    f"max_diff={uncached_difference}"
                )
            if index == 1:
                with torch.inference_mode():
                    hbm_sparse = hbm_model.sparse_module(batch.features)
                    host_sparse = host_model.sparse_module(batch.features)
                    hbm_uncached_output = hbm_model.dense_module.forward_nokvcache(
                        batch, hbm_sparse
                    )
                    host_uncached_output = host_model.dense_module.forward_nokvcache(
                        batch, host_sparse
                    )
                assert hbm_sparse_outputs.keys() == host_sparse_outputs.keys()
                for feature_name in hbm_sparse_outputs:
                    hbm_values, hbm_lengths = hbm_sparse_outputs[feature_name]
                    host_values, host_lengths = host_sparse_outputs[feature_name]
                    assert torch.equal(hbm_lengths, host_lengths), (
                        f"batch=1 sparse lengths differ for {feature_name}"
                    )
                    assert torch.equal(hbm_values, host_values), (
                        f"batch=1 sparse outputs differ for {feature_name}: "
                        f"max_diff={max_diff(hbm_values, host_values)}"
                    )
                uncached_difference = max_diff(
                    hbm_uncached_output, host_uncached_output
                )
                assert torch.allclose(
                    hbm_uncached_output,
                    host_uncached_output,
                    rtol=2e-2,
                    atol=2e-2,
                ), (
                    "batch=1 uncached model outputs differ: "
                    f"max_diff={uncached_difference}"
                )
                batch1_uncached_outputs = (
                    hbm_uncached_output.detach().clone(),
                    host_uncached_output.detach().clone(),
                )
            with torch.inference_mode():
                hbm_output = hbm_model.forward_with_kvcache(
                    batch, user_ids, history_lengths
                )
                host_output = host_model.forward_with_kvcache(
                    batch, user_ids, history_lengths
                )
            item_ids = sorted(
                set(embedding_adapter.batch_item_sequence(batch))
            )
            for key in item_ids:
                assert key in hbm_initialized_rows, (
                    f"batch={index} HBM path did not initialize key {key}"
                )
                assert key in host_initialized_rows, (
                    f"batch={index} host path did not initialize key {key}"
                )
                assert torch.equal(
                    hbm_initialized_rows[key], host_initialized_rows[key]
                ), f"batch={index} initializer values differ for key {key}"
            if index == 0:
                admitted_rows = _read_dynamic_embedding_rows(
                    hbm_model, sorted(admitted_keys)
                )
                for key, row in zip(sorted(admitted_keys), admitted_rows):
                    assert torch.equal(row.cpu(), hbm_initialized_rows[key]), (
                        f"physical admission stored a different value for key {key}"
                    )
            difference = max_diff(hbm_output, host_output)
            assert torch.count_nonzero(hbm_output).item() > 0
            assert torch.count_nonzero(host_output).item() > 0
            if index == 1:
                hbm_uncached_output, host_uncached_output = batch1_uncached_outputs
                assert torch.allclose(
                    hbm_output, hbm_uncached_output, rtol=2e-2, atol=2e-2
                ), (
                    "batch=1 HBM cached and uncached outputs differ: "
                    f"max_diff={max_diff(hbm_output, hbm_uncached_output)}"
                )
                assert torch.allclose(
                    host_output, host_uncached_output, rtol=2e-2, atol=2e-2
                ), (
                    "batch=1 host cached and uncached outputs differ: "
                    f"max_diff={max_diff(host_output, host_uncached_output)}"
                )
                hbm_cache = hbm_model.dense_module.async_kvcache
                host_cache = host_model.dense_module.async_kvcache
                assert hbm_cache.last_origin_cached_lengths == host_cache.last_origin_cached_lengths, (
                    "batch=1 cached prefix lengths differ: "
                    f"hbm={hbm_cache.last_origin_cached_lengths}, "
                    f"host={host_cache.last_origin_cached_lengths}"
                )
                assert hbm_cache.last_origin_cached_lengths == [0], (
                    "batch=1 was expected to start with a cold cache: "
                    f"{hbm_cache.last_origin_cached_lengths}"
                )
                assert hbm_cache.last_new_tokens == host_cache.last_new_tokens, (
                    "batch=1 cache prefill token counts differ: "
                    f"hbm={hbm_cache.last_new_tokens}, "
                    f"host={host_cache.last_new_tokens}"
                )
                assert hbm_cache.last_new_tokens == int(history_lengths[0].item()), (
                    "batch=1 prefill did not cover the complete uncached history: "
                    f"new_tokens={hbm_cache.last_new_tokens}, "
                    f"history_length={int(history_lengths[0].item())}"
                )
                uid = int(user_ids[0].item())
                hbm_interval = hbm_cache.gpu_kvcache_mgr.get_user_cache_interval(uid)
                host_interval = host_cache.gpu_kvcache_mgr.get_user_cache_interval(uid)
                assert hbm_interval[0:2] == host_interval[0:2], (
                    "batch=1 logical cache intervals differ: "
                    f"hbm={hbm_interval}, host={host_interval}"
                )
                assert hbm_interval[0:2] == [0, int(history_lengths[0].item())], (
                    "batch=1 cache interval is not the requested history: "
                    f"interval={hbm_interval}, history={int(history_lengths[0].item())}"
                )
                assert hbm_interval[2] == host_interval[2] == 1, (
                    "batch=1 should allocate one page per model: "
                    f"hbm={hbm_interval}, host={host_interval}"
                )
                torch.cuda.synchronize()
                hbm_page = int(hbm_cache.static_page_ids_gpu_buffer[0].item())
                host_page = int(host_cache.static_page_ids_gpu_buffer[0].item())
                hbm_metadata = hbm_cache.last_kv_metadata
                host_metadata = host_cache.last_kv_metadata
                assert hbm_metadata.keys() == host_metadata.keys()
                assert hbm_metadata["kv_indices"].tolist() == [hbm_page]
                assert host_metadata["kv_indices"].tolist() == [host_page]
                hbm_kv = hbm_cache.cache_table[:, hbm_page, :, : int(history_lengths[0].item())]
                host_kv = host_cache.cache_table[:, host_page, :, : int(history_lengths[0].item())]
                assert torch.equal(hbm_kv, host_kv), (
                    "batch=1 K/V values written to the cold-cache page differ: "
                    f"max_diff={max_diff(hbm_kv, host_kv)}"
                )
                for metadata_name in hbm_metadata:
                    if metadata_name == "kv_indices":
                        continue
                    assert torch.equal(
                        hbm_metadata[metadata_name], host_metadata[metadata_name]
                    ), (
                        f"batch=1 consumed KV metadata differs at {metadata_name}: "
                        f"hbm={hbm_metadata[metadata_name].tolist()}, "
                        f"host={host_metadata[metadata_name].tolist()}"
                    )
            if index == 0:
                assert hbm_sparse_outputs.keys() == host_sparse_outputs.keys()
                for feature_name in hbm_sparse_outputs:
                    hbm_values, hbm_lengths = hbm_sparse_outputs[feature_name]
                    host_values, host_lengths = host_sparse_outputs[feature_name]
                    assert torch.equal(hbm_lengths, host_lengths), (
                        f"batch=0 sparse lengths differ for {feature_name}"
                    )
                    assert torch.equal(hbm_values, host_values), (
                        f"batch=0 sparse outputs differ for {feature_name}: "
                        f"max_diff={max_diff(hbm_values, host_values)}"
                    )
                hbm_cache = hbm_model.dense_module.async_kvcache
                host_cache = host_model.dense_module.async_kvcache
                assert (
                    hbm_cache.last_origin_cached_lengths
                    == host_cache.last_origin_cached_lengths
                ), (
                    "batch=0 cached prefix lengths differ: "
                    f"hbm={hbm_cache.last_origin_cached_lengths}, "
                    f"host={host_cache.last_origin_cached_lengths}"
                )
                assert hbm_cache.last_new_tokens == host_cache.last_new_tokens, (
                    "batch=0 cache prefill token counts differ: "
                    f"hbm={hbm_cache.last_new_tokens}, "
                    f"host={host_cache.last_new_tokens}"
                )
            assert torch.allclose(
                hbm_output, host_output, rtol=2e-2, atol=2e-2
            ), f"batch={index}, max_diff={difference}"
            hbm_model.dense_module.hotstate.after_batch(batch, latency_ms=0.0)
            measured_embedding = embedding_adapter.physical_hbm_bytes()
            measured_kv = kv_adapter.physical_hbm_bytes()
            assert measured_embedding + measured_kv <= state_budget
    finally:
        if hbm_sparse_hook is not None:
            hbm_sparse_hook.remove()
        if host_sparse_hook is not None:
            host_sparse_hook.remove()
        _shutdown_cuda_smoke_model(host_model)
        _shutdown_cuda_smoke_model(hbm_model)
