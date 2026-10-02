"""HotState observation and budget validation for DynamicEmb training.

Training DynamicEmb tables are mutable and their optimizer state is fused into
the DynamicEmb runtime.  This controller deliberately does not move rows or
resize tables: it measures the allocations that DynamicEmb owns and exposes a
step-aligned record for the training loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch


class TrainingHotStateController:
    """Observe DynamicEmb's physical cache while preserving training semantics.

    The training consistency contract is intentionally explicit: rows and
    optimizer state are mutable and coupled.  DynamicEmb remains the owner of
    admission, eviction, and row movement; HotState only observes those
    operations and validates the configured per-rank state envelope.
    """

    consistency_contract = "optimizer_coupled_mutable"

    def __init__(
        self,
        model: Any,
        configured_state_budget_bytes: Optional[int] = None,
        trace_path: str = "",
        dynamic_emb_modules: Optional[Sequence[Any]] = None,
    ) -> None:
        self.model = model
        self.configured_state_budget_bytes = (
            None
            if configured_state_budget_bytes is None
            else int(configured_state_budget_bytes)
        )
        if self.configured_state_budget_bytes is not None and self.configured_state_budget_bytes <= 0:
            raise ValueError("configured_state_budget_bytes must be positive")

        self.modules: List[Any] = list(
            dynamic_emb_modules
            if dynamic_emb_modules is not None
            else self._discover_dynamic_emb_modules(model)
        )
        if not self.modules:
            raise RuntimeError(
                "HotState training integration found no DynamicEmb modules after sharding; "
                "enable it only for a model with dynamic embeddings"
            )
        for module in self.modules:
            storage_states, cache_states = self._module_states(module)
            if not storage_states and not cache_states:
                raise RuntimeError(
                    "HotState training integration cannot introspect DynamicEmb storage "
                    f"for module {type(module)!r}"
                )

        self._step: Optional[int] = None
        self._last_snapshot: Optional[dict] = None
        self._trace_path = self._resolve_trace_path(trace_path)
        for module in self.modules:
            setter = getattr(module, "set_record_cache_metrics", None)
            if callable(setter):
                setter(True)

    @staticmethod
    def _discover_dynamic_emb_modules(model: Any) -> List[Any]:
        try:
            from dynamicemb.dump_load import get_dynamic_emb_module

            return list(get_dynamic_emb_module(model))
        except ImportError as exc:
            raise RuntimeError(
                "DynamicEmb is required for HotState training integration"
            ) from exc

    @staticmethod
    def _rank() -> int:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
        except Exception:
            pass
        return 0

    @staticmethod
    def _cuda_memory_snapshot() -> dict:
        if not torch.cuda.is_available():
            return {"free_bytes": None, "total_bytes": None}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}

    @classmethod
    def _resolve_trace_path(cls, trace_path: str) -> Optional[Path]:
        if not trace_path:
            return None
        path = str(trace_path).format(rank=cls._rank())
        # Avoid concurrent appenders corrupting a single JSONL file in DDP.
        if "{rank}" not in str(trace_path):
            try:
                distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
                world_size = torch.distributed.get_world_size() if distributed else 1
            except Exception:
                world_size = 1
            if world_size > 1:
                path = f"{path}.rank{cls._rank()}"
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _dtype_bytes(dtype: Any) -> int:
        try:
            return int(torch.tensor([], dtype=dtype).element_size())
        except Exception:
            return 4

    @staticmethod
    def _tensor_bytes(value: Any, seen_ptrs: set) -> int:
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            return 0
        try:
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            if pointer in seen_ptrs:
                return 0
            seen_ptrs.add(pointer)
            return int(storage.nbytes())
        except Exception:
            return int(value.numel() * value.element_size())

    @classmethod
    def _buffer_bytes(cls, buffer: Any, seen_ptrs: set) -> int:
        if buffer is None:
            return 0
        is_device_buffer = getattr(buffer, "is_device_buffer", None)
        if callable(is_device_buffer):
            try:
                if not bool(is_device_buffer()):
                    return 0
            except Exception:
                return 0
        allocated_bytes = getattr(buffer, "allocated_bytes", None)
        if callable(allocated_bytes):
            try:
                return int(allocated_bytes())
            except Exception:
                pass
        tensor_fn = getattr(buffer, "tensor", None)
        if callable(tensor_fn):
            try:
                return cls._tensor_bytes(tensor_fn(), seen_ptrs)
            except Exception:
                return 0
        return cls._tensor_bytes(buffer, seen_ptrs)

    @classmethod
    def _state_metadata_bytes(cls, state: Any, seen_ptrs: set) -> int:
        total = 0
        key_index_map = getattr(state, "key_index_map", None)
        memory_usage = getattr(key_index_map, "memory_usage", None)
        if callable(memory_usage):
            try:
                total += int(memory_usage())
            except Exception:
                pass

        # These are static device-side lookup metadata tensors.  The value
        # buffers are counted separately so host-backed buffers never enter the
        # HBM total merely because they expose a tensor-like interface.
        for name in (
            "table_ptrs_dev",
            "table_emb_dims",
            "table_value_dims",
            "no_eviction_next_index_dev",
        ):
            total += cls._tensor_bytes(getattr(state, name, None), seen_ptrs)
        return total

    @classmethod
    def _state_physical_bytes(cls, state: Any, seen_ptrs: set) -> int:
        if state is None:
            return 0
        total = cls._state_metadata_bytes(state, seen_ptrs)
        for table in getattr(state, "tables", ()):
            total += cls._buffer_bytes(table, seen_ptrs)
        return total

    @classmethod
    def _module_static_device_bytes(cls, module: Any, seen_ptrs: set) -> int:
        """Count DynamicEmb-owned tensors outside its table-state objects."""
        total = 0
        for accessor in ("named_buffers", "named_parameters"):
            enumerate_values = getattr(module, accessor, None)
            if not callable(enumerate_values):
                continue
            try:
                values = enumerate_values(recurse=False)
            except TypeError:
                values = enumerate_values()
            for _, value in values:
                total += cls._tensor_bytes(value, seen_ptrs)
        total += cls._tensor_bytes(
            getattr(module, "_prefetch_outstanding_keys", None), seen_ptrs
        )
        counter = getattr(module, "_admission_counter", None)
        memory_usage = getattr(counter, "memory_usage", None)
        if callable(memory_usage):
            try:
                total += int(memory_usage())
            except Exception:
                pass
        return total

    @staticmethod
    def _storage_states(storage: Any) -> List[Any]:
        if storage is None:
            return []
        states: List[Any] = []
        # HybridStorage owns separate HBM and host states.  Host values are
        # excluded by _buffer_bytes, but the host tier's device-side hash/index
        # metadata remains part of the physical HBM allocation.
        seen_ids = set()
        for name in ("_hbm", "_host", "_state"):
            state = getattr(storage, name, None)
            if state is not None and id(state) not in seen_ids:
                states.append(state)
                seen_ids.add(id(state))
        return states

    @classmethod
    def _module_states(cls, module: Any) -> tuple[List[Any], List[Any]]:
        tables = getattr(module, "tables", None)
        if callable(tables):
            tables = tables()
        storage_states = (
            list(tables)
            if isinstance(tables, (list, tuple))
            else cls._storage_states(tables)
        )
        cache = getattr(module, "cache", None)
        if callable(cache):
            cache = cache()
        cache_state = getattr(cache, "_state", None) if cache is not None else None
        cache_states = [cache_state] if cache_state is not None else []
        return storage_states, cache_states

    @staticmethod
    def _state_table_rows(state: Any) -> List[int]:
        key_index_map = getattr(state, "key_index_map", None)
        num_tables = int(getattr(state, "num_tables", 0) or 0)
        sizes = getattr(state, "estimated_table_sizes", None)
        if isinstance(sizes, torch.Tensor) and sizes.numel() >= num_tables > 0:
            try:
                values = [max(0, int(item)) for item in sizes.detach().cpu().tolist()[:num_tables]]
                if any(values):
                    return values
            except Exception:
                pass

        if key_index_map is None or num_tables <= 0:
            return []
        per_table: List[int] = []
        for table_id in range(num_tables):
            size_fn = getattr(key_index_map, "size", None)
            value = None
            if callable(size_fn):
                try:
                    value = size_fn(table_id)
                except (TypeError, RuntimeError):
                    value = None
            if value is None:
                break
            per_table.append(max(0, int(value)))
        if len(per_table) == num_tables:
            return per_table
        if callable(getattr(key_index_map, "size", None)):
            try:
                return [max(0, int(key_index_map.size()))]
            except Exception:
                pass
        return []

    @classmethod
    def _logical_state_bytes(cls, state: Any) -> tuple[int, int, int]:
        rows = cls._state_table_rows(state)
        dimensions = getattr(state, "table_value_dims_cpu", None) or []
        embedding_dimensions = getattr(state, "table_emb_dims_cpu", None) or []
        item_bytes = cls._dtype_bytes(getattr(state, "emb_dtype", torch.float32))
        if not rows:
            return 0, 0, 0
        if len(rows) == 1 and len(dimensions) > 1:
            # A pybind table may only expose aggregate size().  Use the largest
            # row footprint and label the result as a conservative observation.
            dimensions = [max(dimensions)]
            embedding_dimensions = [max(embedding_dimensions or dimensions)]
        logical = 0
        embedding = 0
        optimizer = 0
        for index, row_count in enumerate(rows):
            value_dim = int(dimensions[min(index, len(dimensions) - 1)]) if dimensions else 0
            embedding_dim = int(embedding_dimensions[min(index, len(embedding_dimensions) - 1)]) if embedding_dimensions else value_dim
            logical += row_count * value_dim * item_bytes
            embedding += row_count * embedding_dim * item_bytes
            optimizer += row_count * max(0, value_dim - embedding_dim) * item_bytes
        return logical, embedding, optimizer

    @staticmethod
    def _cache_metrics(module: Any) -> dict:
        cache = getattr(module, "cache", None)
        if callable(cache):
            cache = cache()
        metrics = getattr(cache, "cache_metrics", None) if cache is not None else None
        if callable(metrics):
            metrics = metrics()
        values = [0] * 10
        if metrics is not None:
            try:
                values = [int(item) for item in metrics.detach().cpu().tolist()]
            except Exception:
                try:
                    values = [int(item) for item in metrics]
                except Exception:
                    pass
        unique = values[0] if len(values) > 0 else 0
        hits = values[1] if len(values) > 1 else 0
        inserted = values[2] if len(values) > 2 else 0
        evicted = values[3] if len(values) > 3 else 0
        return {
            "available": metrics is not None,
            "unique_lookup_count": unique,
            "hit_count": hits,
            "miss_count": max(0, unique - hits),
            "insert_count": inserted,
            "eviction_count": evicted,
        }

    def physical_hbm_bytes(self) -> int:
        seen_ptrs: set = set()
        total = 0
        for module in self.modules:
            storage_states, cache_states = self._module_states(module)
            total += self._module_static_device_bytes(module, seen_ptrs)
            total += sum(self._state_physical_bytes(state, seen_ptrs) for state in storage_states)
            total += sum(self._state_physical_bytes(state, seen_ptrs) for state in cache_states)
        return int(total)

    def logical_storage_bytes(self) -> int:
        logical = 0
        for module in self.modules:
            storage_states, _ = self._module_states(module)
            logical += sum(self._logical_state_bytes(state)[0] for state in storage_states)
        return int(logical)

    def optimizer_state_bytes(self) -> int:
        optimizer = 0
        for module in self.modules:
            storage_states, _ = self._module_states(module)
            optimizer += sum(self._logical_state_bytes(state)[2] for state in storage_states)
        return int(optimizer)

    def _module_snapshot(self, module: Any) -> dict:
        storage_states, cache_states = self._module_states(module)
        seen_ptrs: set = set()
        static_physical = self._module_static_device_bytes(module, seen_ptrs)
        physical = static_physical + sum(
            self._state_physical_bytes(state, seen_ptrs)
            for state in storage_states + cache_states
        )
        logical = sum(self._logical_state_bytes(state)[0] for state in storage_states)
        embedding, optimizer = 0, 0
        for state in storage_states:
            _, state_embedding, state_optimizer = self._logical_state_bytes(state)
            embedding += state_embedding
            optimizer += state_optimizer
        return {
            "table_names": list(getattr(module, "table_names", ()) or ()),
            "physical_hbm_bytes": int(physical),
            "static_device_metadata_bytes": int(static_physical),
            "logical_storage_bytes": int(logical),
            "logical_embedding_row_bytes": int(embedding),
            "optimizer_state_bytes": int(optimizer),
            "cache": self._cache_metrics(module),
        }

    def _snapshot(self, step: Optional[int] = None) -> dict:
        physical = self.physical_hbm_bytes()
        logical = self.logical_storage_bytes()
        optimizer = self.optimizer_state_bytes()
        modules = [self._module_snapshot(module) for module in self.modules]
        cache = {
            key: sum(item["cache"][key] for item in modules)
            for key in ("unique_lookup_count", "hit_count", "miss_count", "insert_count", "eviction_count")
        }
        snapshot = {
            "mode": "training",
            "consistency_contract": self.consistency_contract,
            "step": None if step is None else int(step),
            "mutable_state_version": None if step is None else int(step),
            "physical_hbm_scope": "dynamicemb_values_and_device_metadata",
            "rank": self._rank(),
            "device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu",
            "cuda_memory": self._cuda_memory_snapshot(),
            "configured_state_budget_bytes": self.configured_state_budget_bytes,
            "state_budget_scope": "per_rank",
            "physical_hbm_bytes": physical,
            "logical_storage_bytes": logical,
            "optimizer_state_bytes": optimizer,
            "cache_metrics": cache,
            "cache_metrics_available_modules": sum(
                1 for item in modules if item["cache"]["available"]
            ),
            "dynamic_emb_modules": modules,
            "timestamp_ns": time.time_ns(),
        }
        if self.configured_state_budget_bytes is not None:
            snapshot["remaining_state_budget_bytes"] = self.configured_state_budget_bytes - physical
        else:
            snapshot["remaining_state_budget_bytes"] = None
        return snapshot

    def validate_training_budget(self) -> dict:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        snapshot = self._snapshot(self._step)
        budget = self.configured_state_budget_bytes
        if budget is not None and snapshot["physical_hbm_bytes"] > budget:
            raise RuntimeError(
                "DynamicEmb physical HBM exceeds the configured training state envelope: "
                f"physical={snapshot['physical_hbm_bytes']} budget={budget}"
            )
        self._last_snapshot = snapshot
        return snapshot

    def physical_hbm_snapshot(self) -> dict:
        """Return a fresh physical training-state measurement."""
        return self._snapshot(self._step)

    def validate_budget(self) -> dict:
        """Compatibility alias for callers that use the generic controller API."""
        return self.validate_training_budget()

    def snapshot(self) -> dict:
        """Return the latest measured snapshot, refreshing it when necessary."""
        if self._last_snapshot is None:
            return self.validate_training_budget()
        return dict(self._last_snapshot)

    def before_train_step(self, step: int) -> None:
        if self._step is not None:
            raise RuntimeError(f"training step {self._step} is still in flight")
        self._step = int(step)

    def after_train_step(self, step: Optional[int] = None) -> dict:
        completed_step = self._step if step is None else int(step)
        try:
            return self.validate_training_budget() if completed_step == self._step else self._snapshot(completed_step)
        finally:
            self._step = None

    def abort_train_step(self) -> None:
        self._step = None

    def _write_trace(self, snapshot: dict) -> None:
        if self._trace_path is None:
            return
        with self._trace_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(snapshot, sort_keys=True) + "\n")

    def record_train_step(self, step: int) -> dict:
        snapshot = self.after_train_step(step)
        self._write_trace(snapshot)
        return snapshot
