"""One measured request lifecycle for inference and paired static baselines."""

import hashlib
import os
import time

import torch

from modules.hotstate.embedding_adapter import EmbeddingAdapter
from modules.hotstate.kv_adapter import KVAdapter
from modules.hotstate.trace_writer import write_jsonl_record


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
    use_kvcache=True, trace_enabled=True,
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
    previous_trace_enabled = None
    if controller is not None and hasattr(controller, "_trace_enabled"):
        previous_trace_enabled = controller._trace_enabled
        controller._trace_enabled = bool(trace_enabled)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
    try:
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
                    logits = model.forward_with_kvcache(
                        batch, user_ids, history_lengths
                    )
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
    finally:
        if previous_trace_enabled is not None:
            controller._trace_enabled = previous_trace_enabled

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


def _scalar_tree(value):
    """Keep scalar leaves while dropping per-key lists from a trace payload."""
    if isinstance(value, dict):
        return {
            str(key): compacted
            for key, item in value.items()
            if (compacted := _scalar_tree(item)) is not _OMIT
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _OMIT


_OMIT = object()


def _scalar_cost_snapshot(value):
    if not isinstance(value, dict):
        return _scalar_tree(value)
    return _scalar_tree(
        {key: item for key, item in value.items() if key != "policy"}
    )


def _transfer_execution_summary(execution):
    if not isinstance(execution, dict):
        return {}
    count_fields = {
        "submitted_count": "submitted_ids",
        "completed_count": "completed_ids",
        "failed_count": "failed_ids",
        "deferred_count": "deferred_ids",
        "noop_count": "noop_ids",
        "backend_error_count": "backend_errors",
        "admitted_embedding_count": "admitted_embedding_keys",
        "evicted_embedding_count": "evicted_embedding_keys",
        "released_kv_count": "released_kv_keys",
    }
    summary = {}
    for count_name, detail_name in count_fields.items():
        value = execution.get(count_name)
        if value is None:
            detail = execution.get(detail_name, ())
            value = len(detail) if detail is not None else 0
        summary[count_name] = int(value)
    for field in ("submitted_bytes", "completed_bytes", "elapsed_ms"):
        if execution.get(field) is not None:
            summary[field] = execution[field]
    return summary


def run_inference_request(
    model, batch, user_ids, history_lengths, *, controller=None,
    batch_idx=0, state_budget_bytes=None, static_split="static_default",
    trace_enabled=True,
):
    """Measure a request and emit physical state evidence outside the timed interval."""
    logits, control, post, timings = measure_inference_request(
        model, batch, user_ids, history_lengths,
        controller=controller, batch_idx=batch_idx,
        trace_enabled=trace_enabled,
    )
    if not trace_enabled:
        return logits, None

    fingerprint = batch_fingerprint(batch, user_ids, history_lengths)

    if controller is not None:
        embedding_adapter = getattr(controller, "emb_adapter", None)
        if embedding_adapter is None:
            embedding_adapter = EmbeddingAdapter(
                getattr(model, "sparse_module", None)
            )
        physical = controller.physical_hbm_snapshot()
        budget = controller.configured_state_budget_bytes
    else:
        embedding = EmbeddingAdapter(model.sparse_module)
        embedding_adapter = embedding
        kv = KVAdapter(model.dense_module.async_kvcache)
        emb_bytes, kv_bytes = embedding.physical_hbm_bytes(), kv.physical_hbm_bytes()
        physical = {"embedding_physical_hbm_bytes": emb_bytes,
                    "kv_physical_hbm_bytes": kv_bytes,
                    "managed_state_physical_hbm_bytes": emb_bytes + kv_bytes,
                    "resident_page_bytes": kv.actual_resident_kv_bytes()}
        budget = state_budget_bytes
    trace_detail = (
        getattr(controller, "trace_detail", os.getenv("HOTSTATE_TRACE_DETAIL", "scalar"))
        if controller is not None
        else os.getenv("HOTSTATE_TRACE_DETAIL", "scalar")
    )
    if trace_detail == "full":
        record = {
            **physical,
            **timings,
            "record_type": "hotstate" if controller is not None else "static",
            "implementation_scope": (
                "fixed_shared_hbm_inference" if controller is not None
                else "physical_static_shared_hbm_baseline"
            ),
            "benchmark_variant": "physical_envelope",
            "split": "hotstate" if controller is not None else static_split,
            "configured_state_budget_bytes": budget,
            "batch_idx": int(batch_idx),
            "user_id": int(torch.as_tensor(user_ids).reshape(-1)[0].item()),
            "seq_history_len": int(
                torch.as_tensor(history_lengths).reshape(-1)[0].item() // 2
            ),
            "batch_fingerprint": fingerprint,
            "dynamic_embedding_key_spaces": embedding_adapter.key_space_metadata(),
            "dynamic_feature_id_ranges": embedding_adapter.batch_feature_key_ranges(batch),
            "physical_pool_resize_applied": False,
            "online_pool_resize_observed": False,
            "model_rebuild_count": 0,
            "policy": controller.policy_config.snapshot() if controller is not None else None,
            "policy_budget": control.get("policy_budget"),
            "request_metrics": control.get("request_metrics", {}),
            "control_profile_ms": control.get("profile_ms", {}),
            "online_cost_model_before": control.get("online_cost_model", {}),
            "physical_admitted_keys": control.get("physical_admitted_keys", []),
            "physical_evicted_keys": control.get("physical_evicted_keys", []),
            "state_trace": control.get("state_trace", []),
            "policy_inputs": control.get("policy_inputs", []),
            "physical_action_rows": int(bool(
                control.get("physical_admitted_keys")
                or control.get("physical_evicted_keys")
            )),
            "transfer_execution": control.get("transfer_execution", {}),
            "post_control": post,
            "observed_cost_inputs": post.get("observed_cost_inputs"),
            "online_cost_model_after": post.get("online_cost_model", {}),
            "waits": dict(
                getattr(model.dense_module.async_kvcache, "last_wait_timings", {})
            ),
            "trace_detail": "full",
        }
    else:
        key_spaces = [
            {
                key: value
                for key, value in table.items()
                if key != "feature_names"
            }
            for table in embedding_adapter.key_space_metadata()
        ]
        admitted_count = int(
            control.get(
                "physical_admission_count",
                len(control.get("physical_admitted_keys", []) or []),
            )
        )
        evicted_count = int(
            control.get(
                "physical_eviction_count",
                len(control.get("physical_evicted_keys", []) or []),
            )
        )
        physical_scalars = {
            field: physical[field]
            for field in (
                "embedding_physical_hbm_bytes",
                "kv_physical_hbm_bytes",
                "managed_state_physical_hbm_bytes",
                "resident_page_bytes",
            )
            if field in physical
        }
        record = {
            **physical_scalars,
            **timings,
            "record_type": "hotstate" if controller is not None else "static",
            "implementation_scope": (
                "fixed_shared_hbm_inference" if controller is not None
                else "physical_static_shared_hbm_baseline"
            ),
            "benchmark_variant": "physical_envelope",
            "split": "hotstate" if controller is not None else static_split,
            "configured_state_budget_bytes": budget,
            "batch_idx": int(batch_idx),
            "user_id": int(torch.as_tensor(user_ids).reshape(-1)[0].item()),
            "seq_history_len": int(
                torch.as_tensor(history_lengths).reshape(-1)[0].item() // 2
            ),
            "batch_fingerprint": fingerprint,
            "dynamic_embedding_key_spaces": key_spaces,
            "dynamic_feature_id_ranges": embedding_adapter.batch_feature_key_ranges(batch),
            "physical_pool_resize_applied": False,
            "online_pool_resize_observed": False,
            "model_rebuild_count": 0,
            "trace_detail": "scalar",
            "policy": (
                {"config_id": controller.policy_config.snapshot().get("config_id")}
                if controller is not None
                else None
            ),
            "policy_budget": _scalar_tree(control.get("policy_budget")),
            "request_metrics": _scalar_tree(control.get("request_metrics", {})),
            "control_profile_ms": _scalar_tree(control.get("profile_ms", {})),
            "online_cost_model_before": _scalar_cost_snapshot(
                control.get("online_cost_model", {})
            ),
            "physical_admission_count": admitted_count,
            "physical_eviction_count": evicted_count,
            "physical_action_rows": int(bool(admitted_count or evicted_count)),
            "transfer_execution": _transfer_execution_summary(
                control.get("transfer_execution", {})
            ),
            "transfer_scheduler_after": _scalar_tree(
                post.get("transfer_scheduler", {})
            ),
            "directory_pending_transfers_after": post.get(
                "post_directory_pending_transfers"
            ),
            "observed_cost_inputs": _scalar_tree(
                post.get("observed_cost_inputs")
            ),
            "online_cost_model_after": _scalar_cost_snapshot(
                post.get("online_cost_model", {})
            ),
            "waits": _scalar_tree(
                dict(getattr(model.dense_module.async_kvcache, "last_wait_timings", {}))
            ),
        }
        for field in (
            "physical_kv_pool_bytes",
            "logical_kv_page_limit_pages",
            "active_kv_page_limit_bytes",
            "directory_pending_transfers",
        ):
            if field in control:
                record[field] = control[field]
        if int(batch_idx) != 0:
            record.pop("dynamic_embedding_key_spaces", None)
    if budget is not None and physical["managed_state_physical_hbm_bytes"] > budget:
        raise RuntimeError("measured inference state exceeds the configured physical envelope")
    if controller is not None:
        controller.record_inference_trace(record)
    elif os.getenv("HOTSTATE_TRACE_PATH"):
        write_jsonl_record(os.environ["HOTSTATE_TRACE_PATH"], record)
    return logits, record
