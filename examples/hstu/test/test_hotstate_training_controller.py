"""CPU tests for the training-side HotState observer."""

import json
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

from modules.hotstate.training_controller import TrainingHotStateController  # noqa: E402


class _FakeBuffer:
    def __init__(self, allocated_bytes, device=True):
        self._allocated_bytes = allocated_bytes
        self._device = device

    def is_device_buffer(self):
        return self._device

    def allocated_bytes(self):
        return self._allocated_bytes


class _FakeKeyIndexMap:
    def __init__(self, rows):
        self._rows = rows

    def memory_usage(self):
        return 16

    def size(self, table_id=None):
        if table_id is None:
            return sum(self._rows)
        return self._rows[table_id]


class _FakeState:
    def __init__(self, physical_bytes, rows=(3,), device=True):
        self.tables = [_FakeBuffer(physical_bytes, device=device)]
        self.key_index_map = _FakeKeyIndexMap(list(rows))
        self.num_tables = len(rows)
        self.table_emb_dims_cpu = [2] * len(rows)
        self.table_value_dims_cpu = [3] * len(rows)
        self.emb_dtype = torch.float32
        self.estimated_table_sizes = torch.tensor(rows, dtype=torch.int64)


class _FakeCache:
    def __init__(self):
        self._state = _FakeState(50)
        self.cache_metrics = torch.tensor([10, 7, 4, 2] + [0] * 6, dtype=torch.long)
        self.recorded = False

    def set_record_cache_metrics(self, record):
        self.recorded = record


class _FakeModule:
    def __init__(self):
        self.table_names = ["items"]
        self.tables = type("Storage", (), {"_state": _FakeState(100)})()
        self.cache = _FakeCache()
        self.recorded = False

    def set_record_cache_metrics(self, record):
        self.recorded = record


def test_training_snapshot_reports_mutable_contract_and_cache_metrics(tmp_path):
    module = _FakeModule()
    controller = TrainingHotStateController(
        object(),
        configured_state_budget_bytes=300,
        trace_path=str(tmp_path / "training.jsonl"),
        dynamic_emb_modules=[module],
    )

    snapshot = controller.validate_training_budget()

    assert module.recorded is True
    assert snapshot["mode"] == "training"
    assert snapshot["consistency_contract"] == "optimizer_coupled_mutable"
    assert snapshot["physical_hbm_bytes"] == 182  # value + cache + two hash maps
    assert snapshot["logical_storage_bytes"] == 36
    assert snapshot["optimizer_state_bytes"] == 12
    assert snapshot["cache_metrics"] == {
        "unique_lookup_count": 10,
        "hit_count": 7,
        "miss_count": 3,
        "insert_count": 4,
        "eviction_count": 2,
    }

    controller.before_train_step(4)
    completed = controller.record_train_step(4)
    assert completed["step"] == 4
    line = (tmp_path / "training.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["consistency_contract"] == "optimizer_coupled_mutable"


def test_training_budget_rejection_and_inflight_cleanup():
    controller = TrainingHotStateController(
        object(), configured_state_budget_bytes=100, dynamic_emb_modules=[_FakeModule()]
    )
    controller.before_train_step(1)
    with pytest.raises(RuntimeError, match="exceeds the configured training state envelope"):
        controller.record_train_step(1)

    # A failed validation must not poison the next step.
    controller.configured_state_budget_bytes = 300
    controller.before_train_step(2)
    assert controller.record_train_step(2)["step"] == 2


def test_nested_step_is_rejected_and_abort_resets_state():
    controller = TrainingHotStateController(
        object(), dynamic_emb_modules=[_FakeModule()]
    )
    controller.before_train_step(1)
    with pytest.raises(RuntimeError, match="still in flight"):
        controller.before_train_step(2)
    controller.abort_train_step()
    controller.before_train_step(2)
    controller.abort_train_step()

