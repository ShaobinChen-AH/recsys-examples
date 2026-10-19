from collections import deque
from typing import Any, List, Optional, Tuple

import torch

from modules.hotstate.state_handle import (
    ConsistencyError,
    Placement,
    Reconstructability,
    StateHandle,
    StateType,
)


class EmbeddingAdapter:
    """Bridge DynamicEmb while keeping policy estimates separate from HBM facts."""

    DEFAULT_ROW_BYTES = 1024
    SLIDING_WINDOW_SIZE = 10000

    def __init__(self, embedding_module):
        self._module = embedding_module
        self._recent_keys: deque = deque(maxlen=self.SLIDING_WINDOW_SIZE)
        self._total_size_bytes = 0
        self._row_size_bytes = self.DEFAULT_ROW_BYTES
        self._introspection_error: Optional[Exception] = None

    def _embedding_tables(self):
        if self._module is None:
            raise RuntimeError("no InferenceEmbedding module is connected")
        collection = getattr(self._module, "_dynamic_embedding_collection", None)
        tables = getattr(collection, "_embedding_tables", None)
        if tables is None:
            tables = getattr(self._module, "_embedding_tables", None)
        if tables is None:
            raise RuntimeError(
                "expected InferenceEmbedding._dynamic_embedding_collection"
                "._embedding_tables"
            )
        return tables

    def _storage(self):
        tables = self._embedding_tables()
        storage = getattr(tables, "_storage", None)
        if storage is None:
            raise RuntimeError("DynamicEmb table collection has no _storage")
        return storage

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor, seen_ptrs: set) -> int:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return 0
        try:
            storage = tensor.untyped_storage()
            ptr = int(storage.data_ptr())
            if ptr in seen_ptrs:
                return 0
            seen_ptrs.add(ptr)
            return int(storage.nbytes())
        except Exception:
            return int(tensor.numel() * tensor.element_size())

    def _walk_device_tensors(self, value: Any, seen_objects: set, seen_ptrs: set) -> int:
        if isinstance(value, torch.Tensor):
            return self._tensor_bytes(value, seen_ptrs)
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return 0
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)

        # DynamicEmb's HostVMMTensor may be CUDA-addressable, but it is not
        # HBM.  ExtendableBuffer exposes the authoritative tier marker.
        is_device_buffer = getattr(value, "is_device_buffer", None)
        if callable(is_device_buffer):
            try:
                if not bool(is_device_buffer()):
                    return 0
            except Exception:
                return 0

        # The scored hash table is a pybind object.  Its memory_usage() covers
        # the main table storage and bucket-size array; add overflow and
        # reference-counter tensors without recursively double-counting views.
        memory_usage = getattr(value, "memory_usage", None)
        if callable(memory_usage):
            try:
                total = int(memory_usage())
                for name in (
                    "_ref_counter",
                    "overflow_table_storage_",
                    "overflow_bucket_sizes",
                    "overflow_output_offsets_",
                ):
                    try:
                        total += self._tensor_bytes(getattr(value, name), seen_ptrs)
                    except (AttributeError, TypeError):
                        continue
                return total
            except Exception:
                pass

        tensor_fn = getattr(value, "tensor", None)
        if callable(tensor_fn):
            try:
                allocated_bytes = getattr(value, "allocated_bytes", None)
                if callable(allocated_bytes):
                    return int(allocated_bytes())
                return self._tensor_bytes(tensor_fn(), seen_ptrs)
            except Exception:
                pass

        if isinstance(value, dict):
            return sum(
                self._walk_device_tensors(item, seen_objects, seen_ptrs)
                for item in value.values()
            )
        if isinstance(value, (list, tuple, set)):
            return sum(
                self._walk_device_tensors(item, seen_objects, seen_ptrs)
                for item in value
            )

        total = 0
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            total += self._walk_device_tensors(attributes, seen_objects, seen_ptrs)

        for name in (
            "bucket_sizes",
            "table_bucket_offsets_",
            "per_table_capacity_",
            "table_offsets",
            "bucket_offsets",
        ):
            try:
                total += self._walk_device_tensors(
                    getattr(value, name), seen_objects, seen_ptrs
                )
            except (AttributeError, TypeError):
                continue
        return total

    @staticmethod
    def _state_values_bytes(state: Any) -> int:
        total = 0
        for table in getattr(state, "tables", []):
            tensor_fn = getattr(table, "tensor", None)
            if callable(tensor_fn):
                tensor = tensor_fn()
                total += int(tensor.numel() * tensor.element_size())
        return total

    def calibrate(self):
        """Read actual DynamicEmb storage dimensions; never fabricate a size."""
        if self._module is None:
            return
        try:
            storage = self._storage()
            states = []
            if hasattr(storage, "_hbm"):
                states.append(storage._hbm)
            if hasattr(storage, "_host"):
                states.append(storage._host)
            if hasattr(storage, "_state"):
                states.append(storage._state)
            if not states:
                raise RuntimeError(f"unsupported DynamicEmb storage {type(storage)!r}")

            self._total_size_bytes = sum(
                self._state_values_bytes(state) for state in states
            )
            self._row_size_bytes = self.row_size_bytes()
            self._introspection_error = None
        except Exception as exc:
            self._total_size_bytes = 0
            self._introspection_error = exc

    def physical_hbm_bytes(self) -> int:
        """Return only device-backed DynamicEmb storage bytes."""
        try:
            storage = self._storage()
            # HybridStorage's physical HBM tier is `_hbm`; `_host` contains
            # host-backed values and is intentionally excluded from this
            # tier's budget. DynamicEmbStorage uses `_state` for its one tier.
            state = getattr(storage, "_hbm", None)
            if state is None:
                state = getattr(storage, "_state", None)
            if state is None:
                raise RuntimeError(f"unsupported DynamicEmb storage {type(storage)!r}")
            return int(self._walk_device_tensors(state, set(), set()))
        except Exception as exc:
            detail = self._introspection_error or exc
            raise RuntimeError(
                "unable to introspect DynamicEmb device-backed HBM storage; "
                "physical-HBM mode refuses to use a fabricated footprint"
            ) from detail

    def logical_storage_bytes(self) -> int:
        """Return value-buffer bytes across both HBM and host DynamicEmb tiers."""
        try:
            storage = self._storage()
            states = []
            if hasattr(storage, "_hbm"):
                states.append(storage._hbm)
            if hasattr(storage, "_host"):
                states.append(storage._host)
            if hasattr(storage, "_state"):
                states.append(storage._state)
            if not states:
                raise RuntimeError(f"unsupported DynamicEmb storage {type(storage)!r}")
            return int(sum(self._state_values_bytes(state) for state in states))
        except Exception as exc:
            raise RuntimeError("unable to introspect DynamicEmb logical storage") from exc

    def row_size_bytes(self) -> int:
        try:
            storage = self._storage()
            state = getattr(storage, "_hbm", None)
            if state is None:
                state = getattr(storage, "_state", None)
            dims = [int(x) for x in getattr(state, "table_emb_dims_cpu", [])]
            dtype = getattr(state, "emb_dtype", torch.float32)
            if dims:
                return max(dims) * torch.tensor([], dtype=dtype).element_size()
        except Exception:
            pass
        if self._module is not None:
            configs = getattr(self._module, "dynamic_embedding_configs", [])
            if configs:
                return max(
                    int(config.dim)
                    * torch.tensor([], dtype=torch.float32).element_size()
                    for config in configs
                )
        if self._introspection_error is not None:
            raise RuntimeError("unable to determine DynamicEmb row size") from self._introspection_error
        return int(self._row_size_bytes)

    def hbm_capacity_keys(self) -> int:
        """Return the number of value slots available in the HBM tier."""
        storage = self._storage()
        state = getattr(storage, "_hbm", None)
        if state is None:
            return 0
        capacity = 0
        for table in getattr(state, "tables", []):
            tensor_fn = getattr(table, "tensor", None)
            if callable(tensor_fn):
                capacity += int(tensor_fn().size(0))
        return capacity

    def logical_hbm_resident_bytes(self) -> int:
        """Logical value bytes occupied by rows currently indexed in HBM."""
        storage = self._storage()
        state = getattr(storage, "_hbm", None)
        if state is None:
            state = getattr(storage, "_state", None)
        if state is None:
            return 0
        total = 0
        for table_id in range(int(getattr(state, "num_tables", 0))):
            try:
                row_count = int(state.key_index_map.size(table_id))
                value_dim = int(state.table_value_dims_cpu[table_id])
                element_bytes = torch.tensor(
                    [], dtype=state.emb_dtype
                ).element_size()
                total += row_count * value_dim * element_bytes
            except (AttributeError, IndexError, TypeError, RuntimeError):
                continue
        return int(total)

    def supports_physical_residency_control(self) -> bool:
        collection = getattr(self._module, "_dynamic_embedding_collection", None)
        if (
            getattr(self._module, "_hotstate_admit_strategy", None) is not None
            or getattr(collection, "_hotstate_admit_strategy", None) is not None
        ):
            return False
        storage = self._storage()
        return callable(getattr(storage, "promote_keys", None)) and callable(
            getattr(storage, "evict_keys", None)
        )

    def _residency_tensors(self, item_indices) -> Tuple[torch.Tensor, torch.Tensor]:
        storage = self._storage()
        state = getattr(storage, "_hbm", None)
        device = getattr(state, "device", None)
        if device is None:
            device = torch.device("cuda")
        if hasattr(item_indices, "detach"):
            keys = item_indices.detach().to(device=device, dtype=torch.int64).reshape(-1)
        else:
            keys = torch.as_tensor(item_indices, device=device, dtype=torch.int64).reshape(-1)
        table_ids = torch.zeros_like(keys)
        return keys, table_ids

    def apply_residency_plan(
        self, admitted_item_indices, evicted_item_indices=None
    ) -> Tuple[List[int], List[int]]:
        """Apply a physical HybridStorage residency plan.

        The returned lists contain only rows for which DynamicEmb completed a
        real tier movement.  A host-only DynamicEmb or an admission-smoke
        configuration returns empty lists and remains governed by its existing
        admission strategy.
        """
        if self._module is None or not self.supports_physical_residency_control():
            return [], []

        storage = self._storage()
        admitted_keys: List[int] = []
        evicted_keys: List[int] = []

        if evicted_item_indices is not None and len(evicted_item_indices) > 0:
            keys, table_ids = self._residency_tensors(evicted_item_indices)
            moved = storage.evict_keys(keys, table_ids)
            evicted_keys.extend(int(key) for key in moved.detach().cpu().tolist())

        if admitted_item_indices is not None and len(admitted_item_indices) > 0:
            keys, table_ids = self._residency_tensors(admitted_item_indices)
            # In inference, checkpoint values are authoritative and immutable.
            # ``HybridStorage.admit_keys`` intentionally initializes keys absent
            # from both tiers; using it here would turn a cache miss into a new
            # zero/random row.  HotState may only promote a row already backed
            # by the host tier.
            promote_fn = getattr(storage, "promote_keys", None)
            if not callable(promote_fn):
                raise RuntimeError(
                    "DynamicEmb storage cannot perform authoritative host-row promotion"
                )
            promoted, displaced = promote_fn(keys, table_ids)
            admitted_keys.extend(int(key) for key in promoted.detach().cpu().tolist())
            evicted_keys.extend(int(key) for key in displaced.detach().cpu().tolist())

        return admitted_keys, evicted_keys

    def admit_keys(self, item_indices) -> List[int]:
        """Promote host rows and return the rows actually admitted to HBM."""
        admitted, _evicted = self.apply_residency_plan(item_indices)
        return admitted

    def evict_keys(self, item_indices) -> List[int]:
        """Evict HBM rows to host and return the rows actually moved."""
        _admitted, evicted = self.apply_residency_plan([], item_indices)
        return evicted

    def record_batch_keys(self, item_indices: List[int]):
        for idx in item_indices:
            self._recent_keys.append(idx)

    def _observed_residency(self, keys: List[int]):
        """Return DynamicEmb's actual tier for the requested row keys."""
        if not keys:
            return []
        storage = self._storage()
        residency_fn = getattr(storage, "residency", None)
        if not callable(residency_fn):
            return None
        key_tensor, table_ids = self._residency_tensors(keys)
        locations = residency_fn(key_tensor, table_ids)
        return [int(value) for value in locations.detach().cpu().tolist()]

    def validate_item_keys(self, item_indices) -> None:
        """Ensure every inference lookup has an authoritative row.

        A cache miss is a placement miss, not permission to initialize a new
        model value.  DynamicEmb remains responsible for normal lookup; this
        check prevents the HotState controller from silently changing that
        contract through an admission action.
        """
        keys = [int(value) for value in item_indices]
        if not keys or self._module is None or not self.supports_physical_residency_control():
            return
        observed = self._observed_residency(keys)
        if observed is None:
            raise ConsistencyError(
                "DynamicEmb residency cannot be observed for a correctness check"
            )
        missing = [key for key, location in zip(keys, observed) if int(location) == 0]
        if missing:
            if bool(
                getattr(self._module, "_hotstate_initializer_authoritative", False)
            ):
                # The no-checkpoint inference smoke path has an explicit,
                # deterministic DynamicEmb initializer.  It is a valid source
                # of first materialization; HotState itself still never calls
                # the initializer during a residency admission.
                return
            raise ConsistencyError(
                "inference requested embedding rows absent from both authoritative "
                f"tiers: {missing[:8]}"
            )

    def set_module(self, embedding_module):
        self._module = embedding_module
        self._total_size_bytes = 0
        self.calibrate()

    def hot_key_count(self) -> int:
        return len(set(self._recent_keys))

    def hot_footprint_bytes(self) -> int:
        return self.hot_key_count() * self.row_size_bytes()

    def cold_footprint_bytes(self) -> int:
        return max(0, self._total_size_bytes - self.hot_footprint_bytes())

    def export_handles(self) -> List[StateHandle]:
        # In a HybridStorage configuration, expose row-level observations so
        # the directory can represent real HBM/host placement.  The aggregate
        # fallback is retained for host-only and admission-smoke callers.
        recent_keys = list(dict.fromkeys(int(key) for key in self._recent_keys))
        try:
            observed = self._observed_residency(recent_keys)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            observed = None
        if observed is not None:
            row_bytes = max(1, int(self.row_size_bytes()))
            try:
                hbm_state = getattr(self._storage(), "_hbm", None)
                owner_device = str(getattr(hbm_state, "device", "cuda"))
            except Exception:
                owner_device = None
            handles = []
            for key, location in zip(recent_keys, observed):
                if location == 1:
                    placements = {Placement.HBM}
                    tier = "hbm"
                elif location == 2:
                    placements = {Placement.HOST_DRAM}
                    tier = "host"
                else:
                    placements = set()
                    tier = "absent"
                handles.append(
                    StateHandle(
                        state_type=(
                            StateType.EMBEDDING_HOT_ROWS
                            if location == 1
                            else StateType.EMBEDDING_COLD_ROWS
                        ),
                        logical_key=f"emb:item:{key}",
                        footprint_bytes=row_bytes,
                        placement=placements,
                        reconstructability=Reconstructability.REFETCHABLE,
                        consistency_class="immutable_checkpoint",
                        authoritative_placement=(
                            Placement.HBM if location == 1
                            else Placement.HOST_DRAM if location == 2
                            else None
                        ),
                        owner_device=(owner_device if location == 1 else None),
                        transfer_cost_ms=row_bytes / 25_000_000.0,
                        reconstruction_cost_ms=row_bytes / 25_000_000.0,
                        expected_reuse_window=1,
                        writeback_required=False,
                        dependencies=("model:inference", "table:0"),
                        metadata={
                            "table_id": 0,
                            "tier": tier,
                            "resident": bool(location in (1, 2)),
                            "source": "dynamicemb_hybrid_storage",
                        },
                    )
                )
            return handles

        # These aggregate handles are policy estimates only.  They are never
        # used as the physical HBM byte measurement.
        return [
            StateHandle(
                state_type=StateType.EMBEDDING_HOT_ROWS,
                logical_key="emb:item:hot_rows",
                footprint_bytes=self.hot_footprint_bytes(),
                placement={Placement.HOST_DRAM},
                reconstructability=Reconstructability.REFETCHABLE,
                consistency_class="mutable_writeback",
                writeback_required=True,
                transfer_cost_ms=self.hot_footprint_bytes() / 25_000_000.0,
                reconstruction_cost_ms=self.hot_footprint_bytes() / 25_000_000.0,
            ),
            StateHandle(
                state_type=StateType.EMBEDDING_COLD_ROWS,
                logical_key="emb:item:cold_rows",
                footprint_bytes=self.cold_footprint_bytes(),
                placement={Placement.HOST_DRAM},
                reconstructability=Reconstructability.REFETCHABLE,
                consistency_class="refetchable",
                transfer_cost_ms=self.cold_footprint_bytes() / 25_000_000.0,
                reconstruction_cost_ms=self.cold_footprint_bytes() / 25_000_000.0,
            ),
        ]

    def trigger_flush(self):
        if self._module is not None and hasattr(self._module, "flush"):
            self._module.flush()

    def update_admission_policy(
        self, item_indices, max_admitted_keys=None, enabled: bool = True
    ):
        if self._module is None:
            return
        updater = getattr(self._module, "update_hotstate_admission_policy", None)
        if updater is not None:
            updater(
                item_indices=item_indices,
                max_admitted_keys=max_admitted_keys,
                enabled=enabled,
            )
