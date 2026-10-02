"""Dependency-aware execution for real HotState data movement.

The scheduler keeps copies, mapping releases, and native KV transfers
separate.  A logical KV page-limit change is never reported as I/O.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from modules.hotstate.state_handle import Placement, TransferState


class TransferStatus(Enum):
    QUEUED = auto()
    IN_FLIGHT = auto()
    COMPLETE = auto()
    FAILED = auto()
    DEFERRED = auto()
    CANCELLED = auto()
    NOOP = auto()


class TransferResource(Enum):
    PCIE_H2D = auto()
    PCIE_D2H = auto()
    HBM_RELEASE = auto()


@dataclass
class TransferRequest:
    transfer_id: str
    logical_key: str
    state_class: str
    direction: str
    source: Optional[Placement]
    target: Optional[Placement]
    bytes: int
    priority: float
    deadline_epoch: int
    submitted_epoch: int
    resource: TransferResource
    dependencies: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TransferStatus = TransferStatus.QUEUED
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def record(self) -> dict:
        result = asdict(self)
        result["source"] = self.source.name if self.source is not None else None
        result["target"] = self.target.name if self.target is not None else None
        result["resource"] = self.resource.name
        result["status"] = self.status.name
        result["dependencies"] = list(self.dependencies)
        return result


@dataclass
class TransferExecutionResult:
    admitted_embedding_keys: List[int] = field(default_factory=list)
    evicted_embedding_keys: List[int] = field(default_factory=list)
    released_kv_keys: List[str] = field(default_factory=list)
    submitted_ids: List[str] = field(default_factory=list)
    completed_ids: List[str] = field(default_factory=list)
    failed_ids: List[str] = field(default_factory=list)
    deferred_ids: List[str] = field(default_factory=list)
    noop_ids: List[str] = field(default_factory=list)
    backend_errors: Dict[str, str] = field(default_factory=dict)
    submitted_bytes: int = 0
    completed_bytes: int = 0
    elapsed_ms: float = 0.0

    def snapshot(self) -> dict:
        return asdict(self)


class TransferScheduler:
    """One value-ordered lifecycle for DynamicEmb and native KV movement."""

    def __init__(
        self,
        kv_adapter,
        directory,
        value_engine,
        embedding_adapter=None,
        registry=None,
        *,
        bandwidth_fraction: float = 0.80,
        max_history: int = 512,
    ):
        self.kv = kv_adapter
        self.emb = embedding_adapter
        self.dir = directory
        self.registry = registry
        self.ve = value_engine
        self.bandwidth_fraction = min(1.0, max(0.05, float(bandwidth_fraction)))
        self.max_history = max(32, int(max_history))
        self._next_id = 0
        self._queued: Dict[str, TransferRequest] = {}
        self._in_flight: Dict[str, TransferRequest] = {}
        self._history: List[dict] = []
        self._terminal_status: Dict[str, TransferStatus] = {}
        self._onload_by_generation: Dict[int, List[str]] = {}
        self._users_by_generation: Dict[int, List[int]] = {}
        self._last_native_offload_generation = 0
        self._counters = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "deferred": 0,
            "cancelled": 0,
            "noop": 0,
            "submitted_bytes": 0,
            "completed_bytes": 0,
            "deadline_misses": 0,
        }

    def _new_id(self) -> str:
        transfer_id = f"hotstate-transfer-{self._next_id}"
        self._next_id += 1
        return transfer_id

    @staticmethod
    def _scored(scoring_map: Mapping[str, Any], key: str) -> Any:
        return scoring_map.get(key) if scoring_map is not None else None

    def _priority(
        self,
        scoring_map: Mapping[str, Any],
        key: str,
        direction: str,
        bytes_moved: int,
    ) -> float:
        scored = self._scored(scoring_map, key)
        if scored is None:
            return 0.0
        density = max(
            0.0, float(getattr(scored, "value_density_ms_per_byte", 0.0) or 0.0)
        )
        if direction in ("eviction", "release", "offload"):
            return 1.0 / max(1.0e-12, density + 1.0e-12)
        state_class = "embedding" if key.startswith("emb:") else "kv"
        cost_ms = self.ve.cost_model.estimate_transfer_ms(
            state_class=state_class,
            direction=direction,
            bytes_moved=max(0, int(bytes_moved)),
            queue_depth=self.pending_count(),
        )
        benefit_ms = max(0.0, float(getattr(scored, "net_benefit_ms", 0.0)))
        return benefit_ms / max(1.0e-6, cost_ms)

    def _enqueue(
        self,
        *,
        logical_key: str,
        state_class: str,
        direction: str,
        source: Optional[Placement],
        target: Optional[Placement],
        bytes_moved: int,
        priority: float,
        deadline_epoch: int,
        epoch: int,
        resource: TransferResource,
        dependencies: Iterable[str] = (),
        metadata: Optional[dict] = None,
    ) -> TransferRequest:
        request = TransferRequest(
            transfer_id=self._new_id(),
            logical_key=str(logical_key),
            state_class=str(state_class),
            direction=str(direction),
            source=source,
            target=target,
            bytes=max(0, int(bytes_moved)),
            priority=float(priority),
            deadline_epoch=int(deadline_epoch),
            submitted_epoch=int(epoch),
            resource=resource,
            dependencies=tuple(dependencies),
            metadata=dict(metadata or {}),
        )
        self._queued[request.transfer_id] = request
        return request

    def _ordered(self, requests: Iterable[TransferRequest]) -> List[TransferRequest]:
        return sorted(
            requests,
            key=lambda request: (
                request.deadline_epoch,
                -request.priority,
                request.transfer_id,
            ),
        )

    def _max_bytes_for_epoch(self) -> int:
        model = getattr(self.ve, "cost_model", None)
        if model is None:
            return max(1, int(self.kv._page_bytes()))
        window_ms = max(
            0.1,
            float(model.observed_latency_ewma_ms or model.observed_tail_ewma_ms or 4.0),
        )
        one_mib = 1024 * 1024
        costs = [
            model.estimate_transfer_ms(
                state_class="kv",
                direction=direction,
                bytes_moved=one_mib,
                queue_depth=len(self._in_flight),
            )
            for direction in ("onload", "offload")
        ]
        ms_per_mib = max([1.0e-6] + costs)
        return max(
            int(self.kv._page_bytes()),
            int(window_ms * self.bandwidth_fraction / ms_per_mib * one_mib),
        )

    def execute_residency_plan(
        self,
        *,
        admitted_embedding_keys: Sequence[int],
        evicted_embedding_keys: Sequence[int],
        evicted_kv_keys: Sequence[str],
        scoring_map: Mapping[str, Any],
        protected_uids: Iterable[int],
        epoch: int,
        row_size_bytes: int,
    ) -> TransferExecutionResult:
        """Execute one hot-set plan using the actual storage adapters."""
        started = time.perf_counter()
        result = TransferExecutionResult()
        row_bytes = max(1, int(row_size_bytes))
        protected = {int(uid) for uid in protected_uids}
        evictions: List[TransferRequest] = []
        admissions: List[TransferRequest] = []

        for item_id in dict.fromkeys(int(value) for value in evicted_embedding_keys):
            key = f"emb:item:{item_id}"
            request = self._enqueue(
                logical_key=key,
                state_class="embedding",
                direction="eviction",
                source=Placement.HBM,
                target=Placement.HOST_DRAM,
                bytes_moved=row_bytes,
                priority=self._priority(scoring_map, key, "eviction", row_bytes),
                deadline_epoch=epoch,
                epoch=epoch,
                resource=TransferResource.PCIE_D2H,
                metadata={"item_id": item_id},
            )
            evictions.append(request)

        for key in dict.fromkeys(str(value) for value in evicted_kv_keys):
            try:
                uid = int(key.rsplit(":", 1)[-1])
            except ValueError:
                continue
            if uid in protected:
                continue
            footprint = max(0, self.kv.get_page_count_for_user(uid)) * int(
                self.kv._page_bytes()
            )
            has_host_copy = bool(self.kv.has_host_copy(uid))
            request = self._enqueue(
                logical_key=key,
                state_class="kv",
                direction="release",
                source=Placement.HBM,
                target=Placement.HOST_DRAM if has_host_copy else None,
                # Releasing a primary-pool mapping is a physical capacity
                # action, but it does not itself move bytes.  Keep the freed
                # footprint as metadata so transfer accounting stays honest.
                bytes_moved=0,
                priority=self._priority(scoring_map, key, "release", footprint),
                deadline_epoch=epoch,
                epoch=epoch,
                resource=TransferResource.HBM_RELEASE,
                metadata={
                    "uid": uid,
                    "host_copy": has_host_copy,
                    "released_footprint_bytes": footprint,
                },
            )
            evictions.append(request)

        # Current-request admissions are deadline-critical.  Use the learned
        # per-epoch transfer envelope for background D2H eviction work after
        # reserving the admission bytes; mapping releases consume no link
        # bandwidth.  Deferred policy actions are reconsidered next epoch.
        transfer_budget = self._max_bytes_for_epoch()
        admission_bytes = len(
            {
                int(value)
                for value in admitted_embedding_keys
            }
        ) * row_bytes
        background_budget = max(0, transfer_budget - admission_bytes)
        selected_evictions: List[TransferRequest] = []
        for request in self._ordered(evictions):
            link_bytes = (
                request.bytes
                if request.resource in (TransferResource.PCIE_D2H, TransferResource.PCIE_H2D)
                else 0
            )
            if link_bytes > background_budget:
                self._defer(request, "learned epoch bandwidth budget")
                result.deferred_ids.append(request.transfer_id)
                continue
            selected_evictions.append(request)
            background_budget -= link_bytes

        eviction_ids = tuple(
            request.transfer_id for request in selected_evictions
        )
        for item_id in dict.fromkeys(int(value) for value in admitted_embedding_keys):
            key = f"emb:item:{item_id}"
            entry = self.dir.get(key)
            if entry is not None and Placement.HBM in entry.placements:
                request = self._enqueue(
                    logical_key=key,
                    state_class="embedding",
                    direction="admission",
                    source=Placement.HBM,
                    target=Placement.HBM,
                    bytes_moved=0,
                    priority=0.0,
                    deadline_epoch=epoch,
                    epoch=epoch,
                    resource=TransferResource.PCIE_H2D,
                    metadata={"item_id": item_id},
                )
                self._mark_noop(request, "row already resident in HBM")
                result.noop_ids.append(request.transfer_id)
                continue
            request = self._enqueue(
                logical_key=key,
                state_class="embedding",
                direction="admission",
                source=entry.location if entry is not None else Placement.HOST_DRAM,
                target=Placement.HBM,
                bytes_moved=row_bytes,
                priority=self._priority(scoring_map, key, "admission", row_bytes),
                deadline_epoch=epoch,
                epoch=epoch,
                resource=TransferResource.PCIE_H2D,
                dependencies=eviction_ids,
                metadata={"item_id": item_id},
            )
            admissions.append(request)

        ordered_evictions = self._ordered(selected_evictions)
        ordered_admissions = self._ordered(admissions)
        # Phase 1: release capacity.  Admissions are intentionally not marked
        # in flight until every prerequisite has reached a successful terminal
        # state.  This makes the dependency graph operational rather than
        # trace-only metadata.
        for request in ordered_evictions:
            self._start(request, epoch)
            result.submitted_ids.append(request.transfer_id)
            result.submitted_bytes += request.bytes

        embedding_evictions = [
            request for request in ordered_evictions if request.state_class == "embedding"
        ]
        kv_releases = [
            request for request in ordered_evictions if request.state_class == "kv"
        ]
        moved_evict: List[int] = []
        if self.emb is not None and embedding_evictions:
            try:
                movement_started = time.perf_counter()
                _unused_admitted, moved_evict = self.emb.apply_residency_plan(
                    [],
                    [int(request.metadata["item_id"]) for request in embedding_evictions],
                )
                self._observe_embedding_movement(
                    [],
                    moved_evict,
                    row_bytes,
                    1000.0 * (time.perf_counter() - movement_started),
                )
            except Exception as exc:
                result.backend_errors["embedding_eviction"] = str(exc)
                for request in embedding_evictions:
                    if request.status == TransferStatus.IN_FLIGHT:
                        self._fail(request, str(exc), epoch)
                        result.failed_ids.append(request.transfer_id)

        released: Sequence[str] = []
        if kv_releases:
            try:
                released = self.kv.apply_eviction_plan(
                    [request.logical_key for request in kv_releases],
                    protected_uids=protected,
                )
            except Exception as exc:
                result.backend_errors["kv_release"] = str(exc)
                for request in kv_releases:
                    if request.status == TransferStatus.IN_FLIGHT:
                        self._fail(request, str(exc), epoch)
                        result.failed_ids.append(request.transfer_id)

        moved_evict_set = {int(value) for value in moved_evict}
        released_set = set(released)
        for request in embedding_evictions:
            if request.status != TransferStatus.IN_FLIGHT:
                continue
            item_id = int(request.metadata["item_id"])
            if item_id in moved_evict_set:
                self._complete(request, epoch)
                result.evicted_embedding_keys.append(item_id)
                result.completed_ids.append(request.transfer_id)
                result.completed_bytes += request.bytes
            else:
                self._fail(request, "DynamicEmb did not confirm HBM eviction", epoch)
                result.failed_ids.append(request.transfer_id)

        for request in kv_releases:
            if request.status != TransferStatus.IN_FLIGHT:
                continue
            if request.logical_key in released_set:
                self._complete_release(request, epoch)
                result.released_kv_keys.append(request.logical_key)
                result.completed_ids.append(request.transfer_id)
            else:
                self._fail(request, "KV manager did not release the user", epoch)
                result.failed_ids.append(request.transfer_id)

        # Phase 2: submit only dependency-ready admissions.  A failed release
        # does not trigger a speculative copy that could exceed the physical
        # HBM envelope.  The hot-set planner can retry deferred work next epoch.
        ready_admissions: List[TransferRequest] = []
        for request in ordered_admissions:
            incomplete = [
                dependency
                for dependency in request.dependencies
                if not self._dependency_succeeded(dependency)
            ]
            if incomplete:
                request.metadata["incomplete_dependencies"] = incomplete
                self._defer(request, "prerequisite transfer did not complete")
                result.deferred_ids.append(request.transfer_id)
                continue
            self._start(request, epoch)
            result.submitted_ids.append(request.transfer_id)
            result.submitted_bytes += request.bytes
            ready_admissions.append(request)

        moved_admit: List[int] = []
        displaced: List[int] = []
        if self.emb is not None and ready_admissions:
            try:
                movement_started = time.perf_counter()
                moved_admit, displaced = self.emb.apply_residency_plan(
                    [int(request.metadata["item_id"]) for request in ready_admissions],
                    [],
                )
                self._observe_embedding_movement(
                    moved_admit,
                    displaced,
                    row_bytes,
                    1000.0 * (time.perf_counter() - movement_started),
                )
            except Exception as exc:
                result.backend_errors["embedding_admission"] = str(exc)
                for request in ready_admissions:
                    if request.status == TransferStatus.IN_FLIGHT:
                        self._fail(request, str(exc), epoch)
                        result.failed_ids.append(request.transfer_id)

        moved_admit_set = {int(value) for value in moved_admit}
        for request in ready_admissions:
            if request.status != TransferStatus.IN_FLIGHT:
                continue
            item_id = int(request.metadata["item_id"])
            if item_id in moved_admit_set:
                self._complete(request, epoch)
                result.admitted_embedding_keys.append(item_id)
                result.completed_ids.append(request.transfer_id)
                result.completed_bytes += request.bytes
            else:
                self._fail(request, "DynamicEmb did not confirm HBM admission", epoch)
                result.failed_ids.append(request.transfer_id)

        requested_evictions = {
            int(request.metadata["item_id"]) for request in embedding_evictions
        }
        for item_id in sorted(
            (moved_evict_set | {int(value) for value in displaced})
            - requested_evictions
        ):
            self._record_displacement(item_id, row_bytes, epoch)
            result.evicted_embedding_keys.append(item_id)

        result.elapsed_ms = 1000.0 * (time.perf_counter() - started)
        return result

    def stage_kv_request(
        self,
        user_ids: Sequence[int],
        total_history_lengths: Sequence[int],
        *,
        scoring_map: Mapping[str, Any],
        epoch: int,
    ) -> dict:
        """Start the real request-scoped KV prepare and native H2D onload."""
        requests = []
        for uid in (int(value) for value in user_ids):
            tokens = max(0, int(self.kv.host_token_count(uid)))
            if tokens == 0:
                continue
            key = f"kv:uid:{uid}"
            if self.dir.get(key) is None:
                manifest_fn = getattr(self.kv, "user_manifest", None)
                manifest = manifest_fn(uid) if callable(manifest_fn) else {}
                self.dir.register(
                    key,
                    Placement.HOST_DRAM,
                    authoritative=False,
                    version=int(manifest.get("version", tokens)),
                    freshness_epoch=int(manifest.get("version", tokens)),
                    metadata={"segments": manifest},
                )
            bytes_moved = tokens * int(self.kv.token_bytes())
            request = self._enqueue(
                logical_key=key,
                state_class="kv",
                direction="onload",
                source=Placement.HOST_DRAM,
                target=Placement.HBM,
                bytes_moved=bytes_moved,
                priority=self._priority(scoring_map, key, "onload", bytes_moved),
                deadline_epoch=epoch,
                epoch=epoch,
                resource=TransferResource.PCIE_H2D,
                metadata={"uid": uid, "tokens": tokens, "request_scoped": True},
            )
            self._start(request, epoch)
            requests.append(request)
        try:
            stage = self.kv.stage_request(user_ids, total_history_lengths)
        except Exception as exc:
            for request in requests:
                self._fail(request, str(exc), epoch)
            raise
        generation = int(stage["generation"])
        for request in requests:
            request.metadata["prepare_generation"] = generation
        self._onload_by_generation[generation] = [
            request.transfer_id for request in requests
        ]
        self._users_by_generation[generation] = [int(uid) for uid in user_ids]
        return {
            "prepare_generation": generation,
            "staged": True,
            "onload_transfer_ids": [request.transfer_id for request in requests],
            "onload_bytes": sum(request.bytes for request in requests),
        }

    def complete_staged_kv_request(self, generation: int, epoch: int) -> List[str]:
        completed = []
        users = self._users_by_generation.pop(int(generation), [])
        for transfer_id in self._onload_by_generation.pop(int(generation), []):
            request = self._in_flight.get(transfer_id)
            if request is None:
                continue
            self._complete(request, epoch, keep_source_replica=True)
            self._observe_native_transfer(request)
            completed.append(request.logical_key)
        # The request-scoped native buffers are no longer owned by HotState
        # once their lifecycle is closed, so a later eviction may proceed.
        finish = getattr(self.kv, "finish_staged_request", None)
        if users and callable(finish):
            finish(users, success=True)
        return completed

    def abort_staged_kv_request(
        self, generation: int, epoch: int, error: str = "forward failed"
    ) -> None:
        """Close request-scoped ownership after a failed forward."""
        transfer_ids = self._onload_by_generation.pop(int(generation), [])
        users = self._users_by_generation.pop(int(generation), [])
        for transfer_id in transfer_ids:
            request = self._in_flight.get(transfer_id)
            if request is not None:
                self._fail(request, error, epoch)
        finish = getattr(self.kv, "finish_staged_request", None)
        if users and callable(finish):
            finish(users, success=False)

    def observe_native_kv_activity(self, epoch: int) -> dict:
        """Register native asynchronous D2H work accepted during forward."""
        snapshot = self.kv.native_offload_snapshot()
        generation = int(snapshot.get("generation", 0) or 0)
        submitted_ids = []
        if generation > self._last_native_offload_generation and snapshot.get(
            "accepted", False
        ):
            self._last_native_offload_generation = generation
            for raw_uid, raw_tokens in snapshot.get("tokens_by_user", {}).items():
                uid = int(raw_uid)
                tokens = max(0, int(raw_tokens))
                if tokens == 0:
                    continue
                key = f"kv:uid:{uid}"
                bytes_moved = tokens * int(self.kv.token_bytes())
                request = self._enqueue(
                    logical_key=key,
                    state_class="kv",
                    direction="offload",
                    source=Placement.HBM,
                    target=Placement.HOST_DRAM,
                    bytes_moved=bytes_moved,
                    priority=0.0,
                    deadline_epoch=epoch + 1,
                    epoch=epoch,
                    resource=TransferResource.PCIE_D2H,
                    metadata={
                        "uid": uid,
                        "tokens": tokens,
                        "native_generation": generation,
                        "completion_target": int(
                            snapshot.get("completion_target", generation)
                        ),
                    },
                )
                request.started_at = snapshot.get("submitted_at", time.perf_counter())
                self._start(request, epoch, preserve_start=True)
                submitted_ids.append(request.transfer_id)
        return {
            "native_offload_generation": generation,
            "submitted_ids": submitted_ids,
            "accepted": bool(snapshot.get("accepted", False)),
            "busy": bool(self.kv.is_busy_offloading()),
        }

    def poll_completions(self, epoch: int) -> List[str]:
        completed_keys = []
        completed_generation = int(self.kv.completed_offload_count())
        for request in list(self._in_flight.values()):
            if request.direction != "offload":
                continue
            target = int(request.metadata.get("completion_target", 0))
            if target <= 0 or target > completed_generation:
                continue
            self._complete(request, epoch, keep_source_replica=True)
            self._observe_native_transfer(request)
            completed_keys.append(request.logical_key)
        return completed_keys

    def plan_and_submit(self, current_batch, evicted_keys, scoring_map, epoch):
        """Compatibility wrapper; no synthetic future-demand generation."""
        return self.execute_residency_plan(
            admitted_embedding_keys=[],
            evicted_embedding_keys=[],
            evicted_kv_keys=[
                key for key in evicted_keys if str(key).startswith("kv:uid:")
            ],
            scoring_map=scoring_map,
            protected_uids=(),
            epoch=epoch,
            row_size_bytes=1,
        )

    def cancel_stale(self, logical_keys: Iterable[str]) -> List[str]:
        keys = {str(key) for key in logical_keys}
        cancelled = []
        for request in list(self._queued.values()):
            if request.logical_key not in keys:
                continue
            request.status = TransferStatus.CANCELLED
            self._queued.pop(request.transfer_id, None)
            self._remember_terminal(request)
            self._counters["cancelled"] += 1
            self._append_history(request)
            cancelled.append(request.transfer_id)
        return cancelled

    def _ensure_directory_entry(self, request: TransferRequest) -> None:
        if self.dir.get(request.logical_key) is not None:
            return
        location = request.source or request.target or Placement.HOST_DRAM
        self.dir.register(request.logical_key, location, authoritative=True)

    def _start(
        self, request: TransferRequest, epoch: int, *, preserve_start: bool = False
    ) -> None:
        incomplete = [
            dependency
            for dependency in request.dependencies
            if not self._dependency_succeeded(dependency)
        ]
        if incomplete:
            raise RuntimeError(
                f"transfer {request.transfer_id} has incomplete dependencies: "
                + ", ".join(incomplete)
            )
        self._ensure_directory_entry(request)
        self._queued.pop(request.transfer_id, None)
        request.status = TransferStatus.IN_FLIGHT
        if not preserve_start or request.started_at is None:
            request.started_at = time.perf_counter()
        self._in_flight[request.transfer_id] = request
        self.dir.begin_transfer(
            request.logical_key,
            request.direction,
            request.target or request.source or Placement.HOST_DRAM,
            epoch,
            source=request.source,
            transfer_id=request.transfer_id,
            bytes=request.bytes,
            metadata={
                **request.metadata,
                "priority": request.priority,
                "deadline_epoch": request.deadline_epoch,
                "resource": request.resource.name,
                "dependencies": list(request.dependencies),
            },
        )
        self._counters["submitted"] += 1
        self._counters["submitted_bytes"] += request.bytes

    def _complete(
        self,
        request: TransferRequest,
        epoch: int,
        *,
        keep_source_replica: bool = False,
    ) -> None:
        target = request.target or request.source or Placement.HOST_DRAM
        self.dir.complete_transfer(
            request.logical_key,
            target=target,
            epoch=epoch,
            authoritative=request.state_class == "embedding",
            transfer_id=request.transfer_id,
        )
        request.status = TransferStatus.COMPLETE
        request.completed_at = time.perf_counter()
        self._in_flight.pop(request.transfer_id, None)
        self._remember_terminal(request)
        if keep_source_replica and request.source is not None:
            self.dir.admit_replica(request.logical_key, request.source, epoch=epoch)
            # The transfer added a replica; it did not migrate the only copy.
            self._record_registry(request, epoch, success=True, source=None)
        else:
            self._record_registry(request, epoch, success=True)
        self._record_deadline(request, epoch)
        self._counters["completed"] += 1
        self._counters["completed_bytes"] += request.bytes
        self._refresh_kv_manifest(request)
        self._append_history(request)

    def _complete_release(self, request: TransferRequest, epoch: int) -> None:
        self.dir.complete_release(
            request.logical_key,
            source=Placement.HBM,
            target=request.target,
            epoch=epoch,
            authoritative=request.target == Placement.HOST_DRAM,
            transfer_id=request.transfer_id,
        )
        request.status = TransferStatus.COMPLETE
        request.completed_at = time.perf_counter()
        self._in_flight.pop(request.transfer_id, None)
        self._remember_terminal(request)
        handle = self.registry.get(request.logical_key) if self.registry else None
        if handle is not None:
            handle.remove_replica(Placement.HBM)
            if request.target == Placement.HOST_DRAM:
                handle.set_authoritative(Placement.HOST_DRAM)
        self._record_registry(request, epoch, success=True, source=None)
        self._record_deadline(request, epoch)
        self._counters["completed"] += 1
        self._refresh_kv_manifest(request)
        self._append_history(request)

    def _refresh_kv_manifest(self, request: TransferRequest) -> None:
        if request.state_class != "kv":
            return
        manifest_fn = getattr(self.kv, "user_manifest", None)
        entry = self.dir.get(request.logical_key)
        if not callable(manifest_fn) or entry is None:
            return
        uid = int(request.metadata.get("uid", request.logical_key.rsplit(":", 1)[-1]))
        manifest = manifest_fn(uid)
        entry.metadata["segments"] = manifest
        entry.version = int(manifest.get("version", entry.version))
        entry.freshness_epoch = entry.version
        host_prefix = int(manifest.get("host_prefix_tokens", 0))
        hbm_end = int(manifest.get("hbm_start_token", 0)) + int(
            manifest.get("hbm_tokens", 0)
        )
        if Placement.HOST_DRAM in entry.placements:
            entry.replica_versions[Placement.HOST_DRAM] = host_prefix
        if Placement.HBM in entry.placements:
            entry.replica_versions[Placement.HBM] = hbm_end

    def _fail(self, request: TransferRequest, error: str, epoch: int) -> None:
        if request.status == TransferStatus.IN_FLIGHT:
            self.dir.fail_transfer(
                request.logical_key,
                str(error),
                epoch=epoch,
                transfer_id=request.transfer_id,
            )
        request.status = TransferStatus.FAILED
        request.completed_at = time.perf_counter()
        request.error = str(error)
        self._queued.pop(request.transfer_id, None)
        self._in_flight.pop(request.transfer_id, None)
        self._record_registry(request, epoch, success=False, error=str(error))
        self._record_deadline(request, epoch)
        self._remember_terminal(request)
        self._counters["failed"] += 1
        self._append_history(request)

    def _mark_noop(self, request: TransferRequest, reason: str) -> None:
        request.status = TransferStatus.NOOP
        request.completed_at = time.perf_counter()
        request.metadata["noop_reason"] = reason
        self._queued.pop(request.transfer_id, None)
        self._remember_terminal(request)
        self._counters["noop"] += 1
        self._append_history(request)

    def _defer(self, request: TransferRequest, reason: str) -> None:
        request.status = TransferStatus.DEFERRED
        request.completed_at = time.perf_counter()
        request.metadata["defer_reason"] = reason
        self._queued.pop(request.transfer_id, None)
        self._remember_terminal(request)
        self._counters["deferred"] += 1
        self._append_history(request)

    def _record_registry(
        self,
        request: TransferRequest,
        epoch: int,
        *,
        success: bool,
        error: Optional[str] = None,
        source: Any = ...,
    ) -> None:
        if self.registry is None or self.registry.get(request.logical_key) is None:
            return
        actual_source = request.source if source is ... else source
        self.registry.record_transition(
            request.logical_key,
            action=request.direction,
            source=actual_source,
            target=request.target,
            epoch=epoch,
            success=success,
            transfer_id=request.transfer_id,
            error=error,
            authoritative=request.state_class == "embedding",
        )
        directory_entry = self.dir.get(request.logical_key)
        handle = self.registry.get(request.logical_key)
        if directory_entry is not None and handle is not None and directory_entry.in_flight:
            # StateHandle exposes one active transfer summary; keep it aligned
            # with the directory when another operation for this object remains.
            handle.transfer_state = TransferState.IN_FLIGHT
            handle.transfer_id = directory_entry.transfer_id

    def _record_displacement(self, item_id: int, row_bytes: int, epoch: int) -> None:
        key = f"emb:item:{int(item_id)}"
        request = self._enqueue(
            logical_key=key,
            state_class="embedding",
            direction="eviction",
            source=Placement.HBM,
            target=Placement.HOST_DRAM,
            bytes_moved=row_bytes,
            priority=0.0,
            deadline_epoch=epoch,
            epoch=epoch,
            resource=TransferResource.PCIE_D2H,
            metadata={"item_id": int(item_id), "cause": "hbm_pressure"},
        )
        self._start(request, epoch)
        self._complete(request, epoch)

    def _observe_embedding_movement(
        self,
        admitted: Sequence[int],
        evicted: Sequence[int],
        row_bytes: int,
        elapsed_ms: float,
    ) -> None:
        admitted_bytes = len(admitted) * row_bytes
        evicted_bytes = len(evicted) * row_bytes
        total = admitted_bytes + evicted_bytes
        if total <= 0:
            return
        queue_depth = len(self._in_flight)
        for direction, bytes_moved in (
            ("admission", admitted_bytes),
            ("eviction", evicted_bytes),
        ):
            if bytes_moved <= 0:
                continue
            self.ve.observe_transfer(
                state_class="embedding",
                direction=direction,
                bytes_moved=bytes_moved,
                elapsed_ms=elapsed_ms * bytes_moved / total,
                queue_depth=queue_depth,
            )

    def _observe_native_transfer(self, request: TransferRequest) -> None:
        if request.started_at is None or request.completed_at is None:
            return
        self.ve.observe_transfer(
            state_class="kv",
            direction=request.direction,
            bytes_moved=request.bytes,
            elapsed_ms=max(0.0, 1000.0 * (request.completed_at - request.started_at)),
            queue_depth=len(self._in_flight),
        )

    def _append_history(self, request: TransferRequest) -> None:
        self._history.append(request.record())
        if len(self._history) > self.max_history:
            del self._history[: len(self._history) - self.max_history]

    def _dependency_succeeded(self, transfer_id: str) -> bool:
        return self._terminal_status.get(transfer_id) in (
            TransferStatus.COMPLETE,
            TransferStatus.NOOP,
        )

    def _remember_terminal(self, request: TransferRequest) -> None:
        self._terminal_status[request.transfer_id] = request.status
        referenced = {
            dependency
            for pending in (*self._queued.values(), *self._in_flight.values())
            for dependency in pending.dependencies
        }
        for transfer_id in tuple(self._terminal_status):
            if transfer_id not in referenced:
                self._terminal_status.pop(transfer_id, None)

    def _record_deadline(self, request: TransferRequest, epoch: int) -> None:
        missed = int(epoch) > int(request.deadline_epoch)
        request.metadata["deadline_missed"] = missed
        request.metadata["completion_epoch"] = int(epoch)
        if missed:
            self._counters["deadline_misses"] += 1

    def pending_count(self) -> int:
        return len(self._queued) + len(self._in_flight)

    def pending_bytes(self) -> int:
        return sum(request.bytes for request in self._queued.values()) + sum(
            request.bytes for request in self._in_flight.values()
        )

    def _resource_snapshot(self) -> dict:
        result = {}
        for resource in TransferResource:
            queued = [
                request
                for request in self._queued.values()
                if request.resource == resource
            ]
            in_flight = [
                request
                for request in self._in_flight.values()
                if request.resource == resource
            ]
            result[resource.name] = {
                "queued_count": len(queued),
                "queued_bytes": sum(request.bytes for request in queued),
                "in_flight_count": len(in_flight),
                "in_flight_bytes": sum(request.bytes for request in in_flight),
            }
        return result

    def snapshot(self, history_tail: int = 32) -> dict:
        return {
            "scheduler": "dependency_aware_real_backends",
            "bandwidth_fraction": self.bandwidth_fraction,
            "epoch_bandwidth_budget_bytes": self._max_bytes_for_epoch(),
            "queued_count": len(self._queued),
            "in_flight_count": len(self._in_flight),
            "pending_bytes": self.pending_bytes(),
            "resources": self._resource_snapshot(),
            "counters": dict(self._counters),
            "queued": [request.record() for request in self._queued.values()],
            "in_flight": [request.record() for request in self._in_flight.values()],
            "history_tail": self._history[-max(0, int(history_tail)) :],
        }
