import argparse
import json
import math
import statistics
import time
import sys

import gin
import torch
from pathlib import Path

# Reuse existing inference functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference_gr_ranking import get_inference_dataset_and_embedding_configs, get_inference_hstu_model

def percentile(values, q):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[index])

def run_inference_with_latency_tracking(
    checkpoint_dir: str,
    max_bs: int = 1,
    disable_kvcache: bool = False,
    hotstate_controller=None,
):
    """Patch simulate function to track per-batch latencies."""
    from commons.datasets import get_data_loader
    from commons.datasets.inference_dataset import InferenceDataset
    from commons.hstu_data_preprocessor import get_common_preprocessors
    from configs import InferenceEmbeddingConfig, PositionEncodingConfig, RankingConfig, get_inference_hstu_config, get_kvcache_config

    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs()
    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = len(dataproc._contextual_feature_names)

    max_batch_size = max_bs
    total_max_seqlen = dataset_args.max_history_seqlen * 2 + num_contextual_features

    with torch.inference_mode():
        model, hotstate_controller = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
            not disable_kvcache,
            return_controller=True,
        )

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

        dataloader = get_data_loader(dataset=dataset)
        dataloader_iter = iter(dataloader)

        result = run_inference_with_latency_tracking(
            checkpoint_dir=checkpoint_dir,
            max_bs=max_bs,
            disable_kvcache=disable_kvcache,
            hotstate_controller=hotstate_controller,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--max_bs", type=int, default=1)
    parser.add_argument("--disable_kvcache", action="store_true")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    gin.parse_config_file(args.gin_config_file)

    result = run_inference_with_latency_tracking(
        checkpoint_dir=args.checkpoint_dir,
        max_bs=args.max_bs,
        disable_kvcache=args.disable_kvcache,
    )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
