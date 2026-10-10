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
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
            not disable_kvcache,
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

        batch_latencies = []
        num_batches = 0
        start_time = time.time()
        cur_date = None

        while True:
            try:
                uids, dates, seq_endptrs = next(dataloader_iter)
                if dates[0] != cur_date:
                    if cur_date is not None:
                        break
                    cur_date = dates[0]

                batch = dataset.get_input_batch(
                    uids,
                    dates,
                    seq_endptrs,
                    torch.zeros_like(seq_endptrs),
                    with_contextual_features=True,
                    with_ranking_labels=False,
                )
                total_history_lengths = seq_endptrs * 2 + num_contextual_features

                if batch is not None:
                    torch.cuda.synchronize()
                    batch_start = time.perf_counter()

                    if not disable_kvcache:
                        _ = model.forward_with_kvcache(batch, uids, total_history_lengths)
                    else:
                        _ = model.forward_nokvcache(batch)

                    torch.cuda.synchronize()
                    batch_end = time.perf_counter()
                    batch_latencies.append((batch_end - batch_start) * 1000)  # ms

                num_batches += 1
            except StopIteration:
                break

        end_time = time.time()
        total_time = end_time - start_time

        return {
            "num_batches": num_batches,
            "total_time_sec": total_time,
            "mean_latency_ms": statistics.mean(batch_latencies) if batch_latencies else 0.0,
            "stdev_latency_ms": statistics.stdev(batch_latencies) if len(batch_latencies) > 1 else 0.0,
            "p95_latency_ms": percentile(batch_latencies, 0.95),
            "p99_latency_ms": percentile(batch_latencies, 0.99),
            "min_latency_ms": min(batch_latencies) if batch_latencies else 0.0,
            "max_latency_ms": max(batch_latencies) if batch_latencies else 0.0,
        }

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
