"""One measured request lifecycle for inference and paired static baselines."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter


def batch_fingerprint(batch, user_ids, history_lengths):
    """Identify the input before forward trims it to uncached tokens."""
    digest = hashlib.sha256()
    features = batch.features
    values = [user_ids, history_lengths, features.values(), features.lengths()]
    digest.update(repr(tuple(features.keys())).encode("utf-8"))
    for value in values:
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def measure_inference_request(
    model, batch, user_ids, history_lengths, *, controller=None, batch_idx=0,
    use_kvcache=True,
):
    """Measure one prepared-input request using the same boundaries for all policies.

    End-to-end covers before_batch, forward and its device completion wait,
    and after_batch with its completion wait. Forward time is attribution only
    and remains the input to the forward cost model. Input loading, fingerprinting,
    extra physical snapshots, logging, and serialization are outside this scope.

    Return logits, pre-forward control results, post-forward results, and timings.
    """
    control = {}
    post = {}
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
    with torch.inference_mode():
        started = time.perf_counter()
        try:
            if controller is not None:
                control = controller.before_batch(
                    batch, user_ids, history_lengths, batch_idx
                )
            if use_cuda and controller is not None:
                # Finish any device work submitted by the control phase before
                # starting forward-only attribution. The wait remains in the
                # end-to-end request interval.
                torch.cuda.synchronize()
            prepared = time.perf_counter()
            if use_kvcache:
                logits = model.forward_with_kvcache(batch, user_ids, history_lengths)
            else:
                logits = model.forward_nokvcache(batch)
            if use_cuda:
                torch.cuda.synchronize()
            forwarded = time.perf_counter()
            forward_ms = (forwarded - prepared) * 1000.0
            if controller is not None:
                post = controller.after_batch(batch, forward_ms)
                if use_cuda:
                    torch.cuda.synchronize()
            completed = time.perf_counter()
        except Exception as exc:
            if controller is not None:
                controller.abort_batch(str(exc))
            raise

    end_to_end_ms = (completed - started) * 1000.0
    timings = {
        "latency_schema_version": 2,
        "latency_ms": end_to_end_ms,
        "forward_latency_ms": forward_ms,
        "end_to_end_latency_ms": end_to_end_ms,
        "latency_metric": "end_to_end_latency_ms",
        "latency_semantics": "inference_only",
        "end_to_end_scope": (
            "hotstate_control_forward_completion"
            if controller is not None else "forward_completion"
        ),
        "attribution_only_latency_metric": "forward_latency_ms",
        "hotstate_control_ms": (prepared - started) * 1000.0,
        "hotstate_post_control_ms": (completed - forwarded) * 1000.0,
    }
    return logits, control, post, timings


def run_inference_request(model, batch, user_ids, history_lengths, *, controller=None,
                          batch_idx=0, state_budget_bytes=None, static_split="static_default"):
    """Measure a request and emit physical state evidence outside the timed interval."""
    fingerprint = batch_fingerprint(batch, user_ids, history_lengths)
    logits, control, post, timings = measure_inference_request(
        model, batch, user_ids, history_lengths,
        controller=controller, batch_idx=batch_idx,
    )

    if controller is not None:
        physical = controller.physical_hbm_snapshot()
        budget = controller.configured_state_budget_bytes
    else:
        embedding = EmbeddingAdapter(model.sparse_module)
        kv = KVAdapter(model.dense_module.async_kvcache)
        emb_bytes, kv_bytes = embedding.physical_hbm_bytes(), kv.physical_hbm_bytes()
        physical = {"embedding_physical_hbm_bytes": emb_bytes,
                    "kv_physical_hbm_bytes": kv_bytes,
                    "managed_state_physical_hbm_bytes": emb_bytes + kv_bytes,
                    "resident_page_bytes": kv.actual_resident_kv_bytes()}
        budget = state_budget_bytes
    record = {
        **physical,
        **timings,
        "record_type": "hotstate" if controller is not None else "static",
        "implementation_scope": ("fixed_shared_hbm_inference" if controller is not None
                                 else "physical_static_shared_hbm_baseline"),
        "benchmark_variant": "physical_envelope",
        "split": "hotstate" if controller is not None else static_split,
        "configured_state_budget_bytes": budget,
        "batch_idx": int(batch_idx),
        "user_id": int(torch.as_tensor(user_ids).reshape(-1)[0].item()),
        "seq_history_len": int(torch.as_tensor(history_lengths).reshape(-1)[0].item() // 2),
        "batch_fingerprint": fingerprint,
        "physical_pool_resize_applied": False,
        "online_pool_resize_observed": False,
        "model_rebuild_count": 0,
        "policy": controller.policy_config.snapshot() if controller is not None else None,
        "policy_budget": control.get("policy_budget"),
        "physical_admitted_keys": control.get("physical_admitted_keys", []),
        "physical_evicted_keys": control.get("physical_evicted_keys", []),
        "state_trace": control.get("state_trace", []),
        "policy_inputs": control.get("policy_inputs", []),
        "physical_action_rows": int(bool(
            control.get("physical_admitted_keys") or control.get("physical_evicted_keys")
        )),
        "transfer_execution": control.get("transfer_execution", {}),
        "post_control": post,
        "waits": dict(getattr(model.dense_module.async_kvcache, "last_wait_timings", {})),
    }
    if budget is not None and physical["managed_state_physical_hbm_bytes"] > budget:
        raise RuntimeError("measured inference state exceeds the configured physical envelope")
    if controller is not None:
        controller.record_inference_trace(record)
    elif os.getenv("HOTSTATE_TRACE_PATH"):
        path = Path(os.environ["HOTSTATE_TRACE_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    return logits, record
