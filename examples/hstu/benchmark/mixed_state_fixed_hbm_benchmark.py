import argparse
import copy
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import commons.checkpoint as checkpoint
import commons.utils.initialize as init
import gin
import torch
from commons.datasets import get_data_loader as get_infer_data_loader
from commons.datasets.inference_dataset import InferenceDataset
from commons.distributed.batch_shuffler_factory import BatchShufflerFactory
from commons.distributed.sharding import make_optimizer_and_shard
from commons.hstu_data_preprocessor import get_common_preprocessors
from commons.pipeline import TrainPipelineFactory
from configs import (
    InferenceEmbeddingConfig,
    KernelBackend,
    PositionEncodingConfig,
    RankingConfig,
    copy_kvcache_metadata,
    get_inference_hstu_config,
    get_kvcache_config,
)
from megatron.core import parallel_state, tensor_parallel
from model import get_ranking_model
from model.inference_ranking_gr import get_inference_ranking_gr
from modules.inference_dense_module import copy_jagged_metadata
from training.trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader as get_train_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (
    BenchmarkDatasetArgs,
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    RankingArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)

DEFAULT_KV_BLOCKS = 10240
DEFAULT_KV_PAGE_SIZE = 32
DEFAULT_KV_OFFLOAD_CHUNKSIZE = 1024


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[index])


def record_memory_snapshot(label: str) -> Dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "label": label,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, ensure_ascii=True) + "\n")


def kv_debug_enabled() -> bool:
    return os.getenv("MIXED_BENCH_DEBUG_KV") == "1"


def kv_debug(message: str) -> None:
    if kv_debug_enabled():
        print(message, flush=True)


def maybe_load_ckpts_local(
    ckpt_load_dir: str,
    model,
    dense_optimizer=None,
) -> None:
    if not ckpt_load_dir:
        return
    if not os.path.exists(ckpt_load_dir):
        raise FileNotFoundError(f"ckpt_load_dir {ckpt_load_dir} does not exist")

    print(f"Loading checkpoints from {ckpt_load_dir}")
    checkpoint.load(ckpt_load_dir, model, dense_optimizer=dense_optimizer)
    print("Checkpoints loaded!!")


def parse_split_specs(spec: str) -> List[Tuple[int, int]]:
    splits: List[Tuple[int, int]] = []
    for raw in spec.split(","):
        cleaned = raw.strip()
        if not cleaned:
            continue
        parts = cleaned.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid split '{cleaned}', expected x:y")
        lhs, rhs = int(parts[0]), int(parts[1])
        if lhs <= 0 or rhs <= 0:
            raise ValueError(f"Invalid split '{cleaned}', values must be positive")
        splits.append((lhs, rhs))
    if not splits:
        raise ValueError("No valid splits were provided")
    return splits


def parse_policy_filter(spec: str) -> List[str]:
    policy_names = []
    for raw in spec.split(","):
        cleaned = raw.strip()
        if cleaned:
            policy_names.append(cleaned)
    return policy_names


def create_ranking_task_config(
    dataset_args: DatasetArgs | BenchmarkDatasetArgs,
    network_args: NetworkArgs,
    embedding_args: List[EmbeddingArgs],
) -> RankingConfig:
    ranking_args = RankingArgs()
    return RankingConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


def parse_training_spec(train_gin: str) -> Dict[str, Any]:
    gin.clear_config()
    gin.parse_config_file(train_gin)

    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args(
        trainer_args.pipeline_type == "prefetch"
    )
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()
    task_config = create_ranking_task_config(dataset_args, network_args, embedding_args)

    if trainer_args.pipeline_type != "prefetch":
        raise ValueError(
            f"Mixed benchmark expects training pipeline_type='prefetch', got {trainer_args.pipeline_type}"
        )
    if dataset_args.max_history_seqlen != 1024:
        raise ValueError(
            f"Mixed benchmark expects training max_history_seqlen=1024, got {dataset_args.max_history_seqlen}"
        )
    if trainer_args.train_batch_size != 1:
        raise ValueError(
            f"Mixed benchmark expects train_batch_size=1, got {trainer_args.train_batch_size}"
        )
    if network_args.kernel_backend.lower() != "pytorch":
        raise ValueError(
            f"Mixed benchmark expects training kernel_backend='pytorch', got {network_args.kernel_backend}"
        )
    if tp_args.tensor_model_parallel_size != 1:
        raise ValueError(
            "Mixed benchmark expects tensor_model_parallel_size=1 for single-GPU execution"
        )

    return {
        "trainer_args": trainer_args,
        "dataset_args": dataset_args,
        "embedding_args": embedding_args,
        "network_args": network_args,
        "optimizer_args": optimizer_args,
        "tp_args": tp_args,
        "task_config": task_config,
    }


def get_inference_dataset_and_embedding_configs_local(
    dataset_args: DatasetArgs,
    embedding_dim: int,
    disable_contextual_features: bool = False,
):
    if dataset_args.dataset_name != "kuairand-1k":
        raise ValueError(
            f"Mixed benchmark only supports dataset_name='kuairand-1k', got {dataset_args.dataset_name}"
        )

    embedding_configs = [
        InferenceEmbeddingConfig(
            feature_names=["user_id"],
            table_name="user_id",
            vocab_size=1000,
            dim=embedding_dim,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=["user_active_degree"],
            table_name="user_active_degree",
            vocab_size=8,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["follow_user_num_range"],
            table_name="follow_user_num_range",
            vocab_size=9,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["fans_user_num_range"],
            table_name="fans_user_num_range",
            vocab_size=9,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["friend_user_num_range"],
            table_name="friend_user_num_range",
            vocab_size=8,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["register_days_range"],
            table_name="register_days_range",
            vocab_size=8,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["video_id"],
            table_name="video_id",
            vocab_size=1_000_000,
            dim=embedding_dim,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=["action_weights"],
            table_name="action_weights",
            vocab_size=233,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
    ]
    if disable_contextual_features:
        embedding_configs = embedding_configs[-2:]
    return embedding_configs


def parse_inference_spec(infer_gin: str) -> Dict[str, Any]:
    gin.clear_config()
    gin.parse_config_file(infer_gin)

    dataset_args = DatasetArgs()
    network_args = NetworkArgs()
    ranking_args = RankingArgs()
    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = len(dataproc._contextual_feature_names)
    total_max_seqlen = (
        dataset_args.max_num_candidates + dataset_args.max_history_seqlen
    ) * 2 + num_contextual_features

    if dataset_args.max_history_seqlen not in (1024, 2048):
        raise ValueError(
            "Mixed benchmark expects inference max_history_seqlen in {1024, 2048}, "
            f"got {dataset_args.max_history_seqlen}"
        )
    if network_args.kernel_backend.lower() != "pytorch":
        raise ValueError(
            f"Mixed benchmark expects inference kernel_backend='pytorch', got {network_args.kernel_backend}"
        )

    embedding_configs = get_inference_dataset_and_embedding_configs_local(
        dataset_args, network_args.hidden_size
    )
    return {
        "dataset_args": dataset_args,
        "network_args": network_args,
        "ranking_args": ranking_args,
        "embedding_configs": embedding_configs,
        "num_contextual_features": num_contextual_features,
        "total_max_seqlen": total_max_seqlen,
        "dataproc": dataproc,
    }


def build_training_loaders(train_spec: Dict[str, Any]):
    trainer_args: TrainerArgs = train_spec["trainer_args"]
    dataset_args = train_spec["dataset_args"]
    task_config: RankingConfig = train_spec["task_config"]
    return get_train_data_loader(
        "ranking",
        dataset_args,
        trainer_args,
        task_config.num_tasks,
    )


def clone_request(uids, dates, seq_endptrs):
    cloned_uids = uids.clone() if torch.is_tensor(uids) else copy.deepcopy(uids)
    cloned_dates = dates.clone() if torch.is_tensor(dates) else copy.deepcopy(dates)
    cloned_seq_endptrs = (
        seq_endptrs.clone() if torch.is_tensor(seq_endptrs) else copy.deepcopy(seq_endptrs)
    )
    return cloned_uids, cloned_dates, cloned_seq_endptrs


def to_python_scalar(value):
    if torch.is_tensor(value):
        return value.item()
    return value


def prepare_inference_requests(
    infer_spec: Dict[str, Any],
    max_batch_size: int,
    total_batches_needed: int,
):
    dataset_args: DatasetArgs = infer_spec["dataset_args"]
    dataproc = infer_spec["dataproc"]
    total_max_seqlen = infer_spec["total_max_seqlen"]

    dataset = InferenceDataset(
        seq_logs_file=dataproc._inference_sequence_file,
        batch_logs_file=dataproc._inference_batch_file,
        batch_size=max_batch_size,
        max_seqlen=total_max_seqlen,
        item_feature_name=dataproc._item_feature_name,
        contextual_feature_names=dataproc._contextual_feature_names,
        action_feature_name=dataproc._action_feature_name,
        max_num_candidates=dataset_args.max_num_candidates,
        item_vocab_size=10_000_000,
        userid_name="user_id",
        date_name="date",
        sequence_endptr_name="interval_indptr",
        timestamp_names=["date", "interval_end_ts"],
    )

    dataloader = get_infer_data_loader(dataset=dataset)
    dataloader_iter = iter(dataloader)
    cur_date = None
    requests = []

    while len(requests) < total_batches_needed:
        try:
            uids, dates, seq_endptrs = next(dataloader_iter)
        except StopIteration as exc:
            raise RuntimeError(
                f"Only collected {len(requests)} inference requests from the first date window, "
                f"but {total_batches_needed} are required"
            ) from exc

        first_date = to_python_scalar(dates[0])
        if cur_date is None:
            cur_date = first_date
        elif first_date != cur_date:
            break

        requests.append(clone_request(uids, dates, seq_endptrs))

    if len(requests) < total_batches_needed:
        raise RuntimeError(
            f"Collected {len(requests)} inference requests from the first date window, "
            f"but {total_batches_needed} are required"
        )

    return dataset, requests


def find_dynamic_embedding_arg(embedding_args: List[EmbeddingArgs], table_name: str):
    for arg in embedding_args:
        if getattr(arg, "table_name", None) == table_name:
            return arg
    raise KeyError(f"Dynamic embedding table '{table_name}' was not found")


def calculate_default_embedding_budget(train_spec: Dict[str, Any]) -> int:
    embedding_args = copy.deepcopy(train_spec["embedding_args"])
    network_args: NetworkArgs = train_spec["network_args"]
    optimizer_args: OptimizerArgs = train_spec["optimizer_args"]
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    return int(dynamic_options_dict["video_id"].global_hbm_for_values)


def get_bytes_per_kv_primary_page_all_layers(infer_spec: Dict[str, Any]) -> int:
    network_args: NetworkArgs = infer_spec["network_args"]
    if network_args.dtype_str == "bfloat16" or network_args.dtype_str == "float16":
        bytes_per_elem = 2
    else:
        bytes_per_elem = 4
    return (
        network_args.num_layers
        * 2
        * DEFAULT_KV_PAGE_SIZE
        * network_args.num_attention_heads
        * network_args.kv_channels
        * bytes_per_elem
    )


def build_budget_plan(
    train_spec: Dict[str, Any],
    infer_spec: Dict[str, Any],
    state_budget_scale: float,
    split_specs: List[Tuple[int, int]],
) -> Dict[str, Any]:
    b_emb_default = calculate_default_embedding_budget(train_spec)
    kv_page_bytes = get_bytes_per_kv_primary_page_all_layers(infer_spec)
    b_kv_default = DEFAULT_KV_BLOCKS * kv_page_bytes
    b_state_default = b_emb_default + b_kv_default
    b_state_target = math.floor(state_budget_scale * b_state_default)

    if b_state_target <= 0:
        raise ValueError(
            f"Derived non-positive state budget target {b_state_target}, check --state-budget-scale"
        )

    default_embedding_ratio = b_emb_default / b_state_default
    default_kv_ratio = b_kv_default / b_state_default

    policies = []

    def make_policy(policy_name: str, b_emb: int, b_kv_requested: int) -> Dict[str, Any]:
        blocks = max(1, b_kv_requested // kv_page_bytes)
        actual_b_kv = blocks * kv_page_bytes
        return {
            "policy": policy_name,
            "embedding_budget_bytes": int(max(1, b_emb)),
            "kv_budget_bytes": int(actual_b_kv),
            "requested_kv_budget_bytes": int(max(1, b_kv_requested)),
            "state_budget_bytes": int(max(1, b_emb) + actual_b_kv),
            "blocks_in_primary_pool": int(blocks),
        }

    b_emb_local = math.floor(default_embedding_ratio * b_state_target)
    b_kv_local = b_state_target - b_emb_local
    policies.append(make_policy("default_local_scaled", b_emb_local, b_kv_local))

    for lhs, rhs in split_specs:
        split_name = f"static_{lhs}_{rhs}"
        b_emb = math.floor(lhs / (lhs + rhs) * b_state_target)
        b_kv = b_state_target - b_emb
        policies.append(make_policy(split_name, b_emb, b_kv))

    return {
        "embedding_budget_default_bytes": int(b_emb_default),
        "kv_budget_default_bytes": int(b_kv_default),
        "state_budget_default_bytes": int(b_state_default),
        "state_budget_target_bytes": int(b_state_target),
        "default_embedding_ratio": float(default_embedding_ratio),
        "default_kv_ratio": float(default_kv_ratio),
        "kv_primary_page_bytes_all_layers": int(kv_page_bytes),
        "policies": policies,
    }


def get_kernel_backend(network_args: NetworkArgs) -> KernelBackend:
    kernel_backend = {
        "cutlass": KernelBackend.CUTLASS,
        "triton": KernelBackend.TRITON,
        "pytorch": KernelBackend.PYTORCH,
    }.get(network_args.kernel_backend.lower())
    if kernel_backend is None:
        raise ValueError(
            f"Inference kernel backend {network_args.kernel_backend} is not supported"
        )
    return kernel_backend


def build_inference_model(
    infer_spec: Dict[str, Any],
    checkpoint_dir: str,
    blocks_in_primary_pool: int,
    max_batch_size: int,
):
    network_args: NetworkArgs = infer_spec["network_args"]
    ranking_args: RankingArgs = infer_spec["ranking_args"]
    emb_configs = infer_spec["embedding_configs"]
    num_contextual_features = infer_spec["num_contextual_features"]
    total_max_seqlen = infer_spec["total_max_seqlen"]

    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    elif network_args.dtype_str == "float16":
        inference_dtype = torch.float16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )
    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        max_batch_size=max_batch_size,
        max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
        scaling_seqlen=total_max_seqlen,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
        kernel_backend=get_kernel_backend(network_args),
    )
    kv_cache_config = get_kvcache_config(
        blocks_in_primary_pool=blocks_in_primary_pool,
        page_size=DEFAULT_KV_PAGE_SIZE,
        offload_chunksize=DEFAULT_KV_OFFLOAD_CHUNKSIZE,
    )
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )
    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }
    model = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=False,
        cudagraph_configs=hstu_cudagraph_configs,
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()

    if checkpoint_dir:
        model.load_checkpoint(checkpoint_dir)
    model.eval()
    return model


def build_training_pipeline(
    train_spec: Dict[str, Any],
    embedding_budget_bytes: int,
    checkpoint_dir: str,
):
    trainer_args: TrainerArgs = train_spec["trainer_args"]
    dataset_args = train_spec["dataset_args"]
    network_args: NetworkArgs = train_spec["network_args"]
    optimizer_args: OptimizerArgs = train_spec["optimizer_args"]
    tp_args: TensorModelParallelArgs = train_spec["tp_args"]

    embedding_args = copy.deepcopy(train_spec["embedding_args"])
    video_embedding_arg = find_dynamic_embedding_arg(embedding_args, "video_id")
    video_embedding_arg.global_hbm_for_values = int(embedding_budget_bytes)

    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_ranking_task_config(dataset_args, network_args, embedding_args)
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )

    if checkpoint_dir and os.path.exists(checkpoint_dir):
        maybe_load_ckpts_local(checkpoint_dir, model, dense_optimizer)
    elif checkpoint_dir:
        print(
            f"[warning] training checkpoint '{checkpoint_dir}' does not exist; continuing without loading"
        )

    if trainer_args.enable_balanced_shuffler:
        batch_shuffler = BatchShufflerFactory.create(
            "hstu",
            num_heads=hstu_config.num_attention_heads,
            head_dim=hstu_config.kv_channels,
            action_interleaved=True,
        )
    else:
        batch_shuffler = BatchShufflerFactory.create("identity")

    pipeline_name = {
        "prefetch": "jagged_prefetch_sparse_dist",
        "native": "jagged_sparse_dist",
        "none": "jagged_none",
    }[trainer_args.pipeline_type]
    pipeline = TrainPipelineFactory.create(
        pipeline_name,
        model=model_train,
        optimizer=dense_optimizer,
        device=torch.device("cuda", torch.cuda.current_device()),
        batch_shuffler=batch_shuffler,
    )
    pipeline._model.train()
    return {
        "pipeline": pipeline,
        "model": model,
        "sharded_model": model_train,
        "dense_optimizer": dense_optimizer,
    }


def infinite_train_batches(loader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def forward_with_kv_metrics(
    model,
    batch,
    user_ids,
    total_history_lengths,
    debug_label: str | None = None,
):
    dense_module = model.dense_module
    async_kvcache = dense_module.async_kvcache
    debug_prefix = f"[kvdebug:{debug_label}] " if debug_label else "[kvdebug] "

    kv_debug(
        debug_prefix
        + "pre_async "
        + f"batch_size={batch.batch_size} "
        + f"max_total_history_len={int(torch.max(total_history_lengths).item())}"
    )

    prepare_kvcache_result = async_kvcache.prepare_kvcache_async(
        batch.batch_size,
        user_ids.tolist(),
        total_history_lengths.tolist(),
        async_kvcache.static_page_ids_gpu_buffer,
        async_kvcache.static_offload_page_ids_gpu_buffer,
        async_kvcache.static_metadata_gpu_buffer,
        async_kvcache.static_onload_handle,
    )
    origin_cached_lengths = torch.tensor(
        prepare_kvcache_result[0], dtype=torch.int32
    )
    kv_debug(
        debug_prefix
        + "after_async "
        + f"max_origin_cached_len={int(torch.max(origin_cached_lengths).item())} "
        + f"new_tokens={int(prepare_kvcache_result[1])}"
    )
    stripped_batch = async_kvcache.strip_cached_tokens(batch, origin_cached_lengths)

    embeddings = model.sparse_module(stripped_batch.features)
    (
        _,
        num_history_tokens,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,
        kvcache_metadata_fut,
        onload_fut,
    ) = prepare_kvcache_result

    jagged_data = dense_module._hstu_block._preprocessor(
        embeddings=embeddings,
        batch=stripped_batch,
        seq_start_position=origin_cached_lengths.cuda(),
    )
    jagged_data.scaling_seqlen = dense_module._scaling_seqlen

    kvcache_metadata = async_kvcache.prepare_kvcache_wait(
        onload_fut,
        kvcache_metadata_fut,
        stripped_batch.batch_size,
        num_history_tokens,
        async_kvcache.static_page_ids_gpu_buffer,
        async_kvcache.static_offload_page_ids_gpu_buffer,
        offload_uids_buffer,
        metadata_host_buffer,
        metadata_gpu_buffer,
        async_kvcache.static_onload_handle,
    )
    num_offload_pages = len(kvcache_metadata.offload_page_ids)
    kv_debug(
        debug_prefix
        + "after_wait "
        + f"num_offload_pages={int(num_offload_pages)} "
        + f"kv_max_seqlen={int(kvcache_metadata.max_seqlen)}"
    )
    async_kvcache.offload_kvcache(kvcache_metadata)
    kv_debug(debug_prefix + "after_offload")
    kvcache_metadata.total_history_offsets += jagged_data.num_candidates_offsets
    kvcache_metadata.total_history_lengths += jagged_data.num_candidates
    kvcache_metadata.max_seqlen += jagged_data.max_num_candidates

    num_tokens = stripped_batch.features.values().shape[0]
    if dense_module.use_cudagraph:
        dense_module._hidden_states[:num_tokens, ...].copy_(
            jagged_data.values, non_blocking=True
        )
        copy_jagged_metadata(dense_module._jagged_metadata, jagged_data)
        copy_kvcache_metadata(dense_module._kvcache_metadata, kvcache_metadata)
        hstu_output = dense_module._hstu_block.predict(
            stripped_batch.batch_size,
            num_tokens,
            dense_module._hidden_states,
            dense_module._jagged_metadata,
            kvcache_metadata,
        )
        jagged_data.values = hstu_output
    else:
        hstu_output = dense_module._hstu_block.predict(
            stripped_batch.batch_size,
            num_tokens,
            jagged_data.values,
            jagged_data,
            kvcache_metadata,
        )
        jagged_data.values = hstu_output
    kv_debug(debug_prefix + "after_predict")

    jagged_data = dense_module._hstu_block._postprocessor(jagged_data)
    logits = dense_module._mlp(jagged_data.values)
    return logits, {
        "new_tokens": int(num_history_tokens),
        "origin_cached_lengths": [int(x) for x in origin_cached_lengths.tolist()],
        "num_offload_pages": int(num_offload_pages),
    }


def run_training_step(pipeline, train_iter) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    pipeline.progress(train_iter)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def build_inference_batch(dataset, request):
    uids, dates, seq_endptrs = request
    batch = dataset.get_input_batch(
        uids,
        dates,
        seq_endptrs,
        torch.zeros_like(seq_endptrs),
        with_contextual_features=True,
        with_ranking_labels=False,
    )
    return batch, uids, seq_endptrs


def run_inference_step(
    model,
    dataset,
    request,
    num_contextual_features: int,
    debug_label: str | None = None,
):
    batch, uids, seq_endptrs = build_inference_batch(dataset, request)
    if batch is None:
        raise RuntimeError("Inference dataset returned an empty batch")
    total_history_lengths = seq_endptrs * 2 + num_contextual_features
    torch.cuda.synchronize()
    start = time.perf_counter()
    _, metrics = forward_with_kv_metrics(
        model,
        batch,
        uids,
        total_history_lengths,
        debug_label=debug_label,
    )
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000.0
    kv_debug(
        (f"[kvdebug:{debug_label}] " if debug_label else "[kvdebug] ")
        + f"after_step latency_ms={latency_ms:.2f}"
    )
    return latency_ms, metrics


def teardown_runtime(runtime: Dict[str, Any]) -> None:
    if not runtime:
        return

    maybe_model = runtime.get("model")
    try:
        dense_module = getattr(maybe_model, "dense_module", None)
        async_kvcache = getattr(dense_module, "async_kvcache", None)
        if async_kvcache is not None:
            async_kvcache.executor.shutdown(wait=False, cancel_futures=True)
            async_kvcache.onload_worker.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    for key in list(runtime.keys()):
        runtime[key] = None

    gc.collect()
    torch.cuda.empty_cache()


def summarize_metrics(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(statistics.fmean(values)),
        "p95_ms": float(percentile(values, 0.95)),
    }


def print_record_summary(record: Dict[str, Any]) -> None:
    policy = record["policy"]
    status = record["status"]
    train_mean = record.get("train_mean_ms", 0.0)
    infer_mean = record.get("infer_mean_ms", 0.0)
    blocks = record.get("blocks_in_primary_pool", 0)
    print(
        f"[{policy}] status={status} train_mean_ms={train_mean:.2f} "
        f"infer_mean_ms={infer_mean:.2f} blocks={blocks}"
    )


def run_policy_benchmark(
    policy: Dict[str, Any],
    train_spec: Dict[str, Any],
    infer_spec: Dict[str, Any],
    train_loader,
    inference_dataset,
    inference_requests,
    checkpoint_train: str,
    checkpoint_infer: str,
    warmup_train_steps: int,
    warmup_infer_batches: int,
    infer_batches_per_cycle: int,
    measure_cycles: int,
    out_jsonl: Path,
    record_type: str = "policy",
) -> Dict[str, Any]:
    train_runtime: Dict[str, Any] = {}
    infer_runtime: Dict[str, Any] = {}
    record: Dict[str, Any] = {
        "record_type": record_type,
        "policy": policy["policy"],
        "status": "error",
        "embedding_budget_bytes": policy["embedding_budget_bytes"],
        "kv_budget_bytes": policy["kv_budget_bytes"],
        "requested_kv_budget_bytes": policy["requested_kv_budget_bytes"],
        "state_budget_bytes": policy["state_budget_bytes"],
        "blocks_in_primary_pool": policy["blocks_in_primary_pool"],
        "train_max_history_seqlen": int(
            train_spec["dataset_args"].max_history_seqlen
        ),
        "infer_max_history_seqlen": int(
            infer_spec["dataset_args"].max_history_seqlen
        ),
        "infer_total_max_seqlen": int(infer_spec["total_max_seqlen"]),
        "warmup_last_started_idx": -1,
        "warmup_last_completed_idx": -1,
        "warmup_last_started_max_seq_endptr": -1,
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        record["memory_before_init"] = record_memory_snapshot("before_init")
        train_runtime = build_training_pipeline(
            train_spec,
            policy["embedding_budget_bytes"],
            checkpoint_train,
        )
        infer_runtime["model"] = build_inference_model(
            infer_spec,
            checkpoint_infer,
            policy["blocks_in_primary_pool"],
            max_batch_size=1,
        )
        record["memory_after_init"] = record_memory_snapshot("after_init")

        train_iter = infinite_train_batches(train_loader)
        total_infer_needed = warmup_infer_batches + measure_cycles * infer_batches_per_cycle
        if total_infer_needed > len(inference_requests):
            raise RuntimeError(
                f"Need {total_infer_needed} inference requests, but only {len(inference_requests)} are available"
            )

        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup_train_steps):
            run_training_step(train_runtime["pipeline"], train_iter)
        warmup_kv_metrics = []
        for idx in range(warmup_infer_batches):
            seq_endptrs = inference_requests[idx][2]
            max_seq_endptr = (
                int(torch.max(seq_endptrs).item())
                if torch.is_tensor(seq_endptrs)
                else int(max(seq_endptrs))
            )
            sum_seq_endptr = (
                int(torch.sum(seq_endptrs).item())
                if torch.is_tensor(seq_endptrs)
                else int(sum(seq_endptrs))
            )
            record["warmup_last_started_idx"] = int(idx)
            record["warmup_last_started_max_seq_endptr"] = int(max_seq_endptr)
            kv_debug(
                f"[warmup:{policy['policy']}] start idx={idx} "
                + f"max_seq_endptr={max_seq_endptr} sum_seq_endptr={sum_seq_endptr}"
            )
            _, metrics = run_inference_step(
                infer_runtime["model"],
                inference_dataset,
                inference_requests[idx],
                infer_spec["num_contextual_features"],
                debug_label=f"warmup[{idx}]",
            )
            warmup_kv_metrics.append(metrics)
            record["warmup_last_completed_idx"] = int(idx)
            completed_max_origin = (
                max(metrics["origin_cached_lengths"])
                if metrics["origin_cached_lengths"]
                else 0
            )
            kv_debug(
                f"[warmup:{policy['policy']}] completed idx={idx} "
                + f"new_tokens={metrics['new_tokens']} "
                + f"max_origin_cached_len={completed_max_origin} "
                + f"num_offload_pages={metrics['num_offload_pages']}"
            )
        record["memory_after_warmup"] = record_memory_snapshot("after_warmup")
        record["warmup_kv_metrics"] = {
            "total_new_tokens": int(
                sum(metric["new_tokens"] for metric in warmup_kv_metrics)
            ),
            "total_offload_pages": int(
                sum(metric["num_offload_pages"] for metric in warmup_kv_metrics)
            ),
            "max_origin_cached_length": int(
                max(
                    (
                        max(metric["origin_cached_lengths"])
                        if metric["origin_cached_lengths"]
                        else 0
                    )
                    for metric in warmup_kv_metrics
                )
            )
            if warmup_kv_metrics
            else 0,
        }

        torch.cuda.reset_peak_memory_stats()
        train_latencies_ms = []
        infer_latencies_ms = []
        infer_metric_agg = {
            "new_tokens_total": 0,
            "num_offload_pages_total": 0,
            "max_origin_cached_length": 0,
        }

        infer_index = warmup_infer_batches
        for _ in range(measure_cycles):
            train_latencies_ms.append(
                run_training_step(train_runtime["pipeline"], train_iter)
            )
            for _ in range(infer_batches_per_cycle):
                infer_latency_ms, metrics = run_inference_step(
                    infer_runtime["model"],
                    inference_dataset,
                    inference_requests[infer_index],
                    infer_spec["num_contextual_features"],
                    debug_label=f"measure[{infer_index}]",
                )
                infer_index += 1
                infer_latencies_ms.append(infer_latency_ms)
                infer_metric_agg["new_tokens_total"] += metrics["new_tokens"]
                infer_metric_agg["num_offload_pages_total"] += metrics["num_offload_pages"]
                if metrics["origin_cached_lengths"]:
                    infer_metric_agg["max_origin_cached_length"] = max(
                        infer_metric_agg["max_origin_cached_length"],
                        max(metrics["origin_cached_lengths"]),
                    )

        record["memory_end"] = record_memory_snapshot("end")
        train_summary = summarize_metrics(train_latencies_ms)
        infer_summary = summarize_metrics(infer_latencies_ms)
        record.update(
            {
                "status": "ok",
                "train_mean_ms": train_summary["mean_ms"],
                "train_p95_ms": train_summary["p95_ms"],
                "infer_mean_ms": infer_summary["mean_ms"],
                "infer_p95_ms": infer_summary["p95_ms"],
                "combined_cycle_mean_ms": train_summary["mean_ms"]
                + infer_batches_per_cycle * infer_summary["mean_ms"],
                "measured_cycles": int(measure_cycles),
                "measured_infer_batches": int(len(infer_latencies_ms)),
                "kv_metrics": infer_metric_agg,
            }
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            record["status"] = "oom"
        else:
            record["status"] = "runtime_error"
        record["error_message"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        record["status"] = "error"
        record["error_message"] = str(exc)
    finally:
        print_record_summary(record)
        append_jsonl(out_jsonl, record)
        teardown_runtime(train_runtime)
        teardown_runtime(infer_runtime)
    return record


def run_calibrate_mode(
    args,
    train_spec: Dict[str, Any],
    infer_spec: Dict[str, Any],
    budget_plan: Dict[str, Any],
) -> None:
    train_loader, _ = build_training_loaders(train_spec)
    inference_dataset, inference_requests = prepare_inference_requests(
        infer_spec,
        max_batch_size=1,
        total_batches_needed=1,
    )

    calibrate_record = {
        "record_type": "calibrate",
        "embedding_budget_default_bytes": budget_plan[
            "embedding_budget_default_bytes"
        ],
        "kv_budget_default_bytes": budget_plan["kv_budget_default_bytes"],
        "state_budget_default_bytes": budget_plan["state_budget_default_bytes"],
        "state_budget_target_bytes": budget_plan["state_budget_target_bytes"],
        "default_embedding_ratio": budget_plan["default_embedding_ratio"],
        "default_kv_ratio": budget_plan["default_kv_ratio"],
        "kv_primary_page_bytes_all_layers": budget_plan[
            "kv_primary_page_bytes_all_layers"
        ],
        "sample_inference_requests_available": len(inference_requests),
        "train_max_history_seqlen": int(
            train_spec["dataset_args"].max_history_seqlen
        ),
        "infer_max_history_seqlen": int(
            infer_spec["dataset_args"].max_history_seqlen
        ),
        "infer_total_max_seqlen": int(infer_spec["total_max_seqlen"]),
    }

    print("Calibration summary:")
    print(f"  B_emb_default: {budget_plan['embedding_budget_default_bytes']}")
    print(f"  B_kv_default: {budget_plan['kv_budget_default_bytes']}")
    print(f"  B_state_default: {budget_plan['state_budget_default_bytes']}")
    print(f"  default embedding ratio: {budget_plan['default_embedding_ratio']:.4f}")
    print(f"  default kv ratio: {budget_plan['default_kv_ratio']:.4f}")
    print(f"  B_state_target: {budget_plan['state_budget_target_bytes']}")

    temp_out = Path(args.out_jsonl)
    temp_out.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(temp_out, calibrate_record)

    runtime = {}
    try:
        runtime["training"] = build_training_pipeline(
            train_spec,
            budget_plan["embedding_budget_default_bytes"],
            args.train_checkpoint_dir,
        )
        teardown_runtime(runtime["training"])
        runtime["training"] = {}

        runtime["inference"] = {
            "model": build_inference_model(
                infer_spec,
                args.infer_checkpoint_dir,
                DEFAULT_KV_BLOCKS,
                max_batch_size=1,
            )
        }
        print("Calibration construction succeeded for both training and inference.")
    finally:
        teardown_runtime(runtime.get("training", {}))
        teardown_runtime(runtime.get("inference", {}))


def run_mixed_mode(
    args,
    train_spec: Dict[str, Any],
    infer_spec: Dict[str, Any],
    budget_plan: Dict[str, Any],
) -> None:
    train_loader, _ = build_training_loaders(train_spec)
    total_infer_batches_needed = args.warmup_infer_batches + (
        args.measure_cycles * args.infer_batches_per_cycle
    )
    inference_dataset, inference_requests = prepare_inference_requests(
        infer_spec,
        max_batch_size=1,
        total_batches_needed=total_infer_batches_needed,
    )
    out_jsonl = Path(args.out_jsonl)

    policies = budget_plan["policies"]
    selected_policy_names = parse_policy_filter(args.policy_filter)
    if selected_policy_names:
        policies = [
            policy
            for policy in policies
            if policy["policy"] in selected_policy_names
        ]
        if not policies:
            available_policies = ", ".join(
                policy["policy"] for policy in budget_plan["policies"]
            )
            requested_policies = ", ".join(selected_policy_names)
            raise ValueError(
                f"No policies matched --policy-filter={requested_policies}. "
                f"Available policies: {available_policies}"
            )

    smoke_record = None
    default_policy = next(
        (policy for policy in policies if policy["policy"] == "default_local_scaled"),
        None,
    )
    if default_policy is not None:
        smoke_record = run_policy_benchmark(
            default_policy,
            train_spec,
            infer_spec,
            train_loader,
            inference_dataset,
            inference_requests,
            args.train_checkpoint_dir,
            args.infer_checkpoint_dir,
            args.warmup_train_steps,
            args.warmup_infer_batches,
            args.infer_batches_per_cycle,
            10,
            out_jsonl,
            record_type="smoke",
        )
        if smoke_record["status"] != "ok":
            raise RuntimeError(
                f"default_local_scaled smoke failed with status={smoke_record['status']}: "
                f"{smoke_record.get('error_message', 'unknown error')}"
            )

    policy_records = []
    for policy in policies:
        record = run_policy_benchmark(
            policy,
            train_spec,
            infer_spec,
            train_loader,
            inference_dataset,
            inference_requests,
            args.train_checkpoint_dir,
            args.infer_checkpoint_dir,
            args.warmup_train_steps,
            args.warmup_infer_batches,
            args.infer_batches_per_cycle,
            args.measure_cycles,
            out_jsonl,
        )
        policy_records.append(record)

    static_records = [
        record
        for record in policy_records
        if record["status"] == "ok" and record["policy"].startswith("static_")
    ]
    if not static_records:
        raise RuntimeError("No successful static split records were produced")

    oracle_records = []
    if len(policies) > 1:
        best_static = min(
            static_records, key=lambda record: record["combined_cycle_mean_ms"]
        )
        oracle_policy = next(
            policy for policy in policies if policy["policy"] == best_static["policy"]
        )
        oracle_records.append(
            run_policy_benchmark(
                {
                    **oracle_policy,
                    "policy": f"oracle_replay({best_static['policy']})",
                },
                train_spec,
                infer_spec,
                train_loader,
                inference_dataset,
                inference_requests,
                args.train_checkpoint_dir,
                args.infer_checkpoint_dir,
                args.warmup_train_steps,
                args.warmup_infer_batches,
                args.infer_batches_per_cycle,
                args.measure_cycles,
                out_jsonl,
                record_type="oracle_replay",
            )
        )

    print("Final mixed benchmark summary:")
    summary_records = policy_records + oracle_records
    for record in summary_records:
        print_record_summary(record)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mixed-state fixed-HBM benchmark for kuairand_1k",
        allow_abbrev=False,
    )
    parser.add_argument("--mode", choices=["calibrate", "mixed"], required=True)
    parser.add_argument(
        "--train-gin",
        default="./training/configs/kuairand_1k_ranking_prefetch_pytorch_ckpt_1024_b1.gin",
    )
    parser.add_argument(
        "--infer-gin",
        default="./inference/configs/kuairand_1k_inference_ranking_1024.gin",
    )
    parser.add_argument(
        "--train-checkpoint-dir",
        default="./checkpoints/kuairand_1k_prefetch_pytorch_1024_b1/iter550",
    )
    parser.add_argument(
        "--infer-checkpoint-dir",
        default="./checkpoints/kuairand_1k_native_pytorch_1024_b1/iter550",
    )
    parser.add_argument("--state-budget-scale", type=float, default=0.85)
    parser.add_argument("--splits", default="30:70,50:50,70:30")
    parser.add_argument("--warmup-train-steps", type=int, default=20)
    parser.add_argument("--warmup-infer-batches", type=int, default=128)
    parser.add_argument("--infer-batches-per-cycle", type=int, default=8)
    parser.add_argument("--measure-cycles", type=int, default=100)
    parser.add_argument("--policy-filter", default="")
    parser.add_argument(
        "--out-jsonl",
        default="./logs/mixed_state_fixed_hbm_1024_b1.jsonl",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    split_specs = parse_split_specs(args.splits)
    train_spec = parse_training_spec(args.train_gin)
    infer_spec = parse_inference_spec(args.infer_gin)
    budget_plan = build_budget_plan(
        train_spec, infer_spec, args.state_budget_scale, split_specs
    )

    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=train_spec["tp_args"].tensor_model_parallel_size
    )
    init.set_random_seed(train_spec["trainer_args"].seed)

    try:
        if args.mode == "calibrate":
            run_calibrate_mode(args, train_spec, infer_spec, budget_plan)
        else:
            run_mixed_mode(args, train_spec, infer_spec, budget_plan)
    finally:
        init.destroy_global_state()


if __name__ == "__main__":
    main()

