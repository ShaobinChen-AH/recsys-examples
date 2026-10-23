"""CPU tests for sparse checkpoint misses in the inference embedding adapter."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))

from modules.hotstate.embedding_adapter import EmbeddingAdapter  # noqa: E402
from modules.hotstate.state_handle import ConsistencyError  # noqa: E402


class _EmbeddingTables:
    def __init__(self):
        self._eval_initializers = [lambda values, indices, keys: values.zero_()]


class _EmbeddingModule:
    def __init__(self):
        self.dynamic_embedding_configs = [
            SimpleNamespace(
                table_name="user_id",
                feature_names=["user_id"],
                vocab_size=1000,
            ),
            SimpleNamespace(
                table_name="video_id",
                feature_names=["video_id"],
                vocab_size=1_000_000,
            ),
        ]
        self._dynamic_embedding_collection = SimpleNamespace(
            _embedding_tables=_EmbeddingTables()
        )


def test_missing_in_range_user_uses_eval_fallback_after_checkpoint_load():
    adapter = EmbeddingAdapter(_EmbeddingModule())
    adapter.supports_physical_residency_control = lambda: True
    adapter._observed_residency = lambda keys: [0 for _key in keys]

    adapter.validate_item_keys([867])


def test_negative_key_is_rejected_without_treating_capacity_as_id_bound():
    adapter = EmbeddingAdapter(_EmbeddingModule())

    with pytest.raises(ConsistencyError, match="non-negative"):
        adapter.validate_key_range([-1])


def test_checkpoint_missing_row_can_be_admitted_with_eval_initializer():
    module = _EmbeddingModule()
    adapter = EmbeddingAdapter(module)
    calls = []

    class Storage:
        def admit_keys(self, keys, table_ids, initializer, *, allow_initialize):
            calls.append((initializer, allow_initialize, keys.tolist()))
            return keys, keys[:0]

    adapter.supports_physical_residency_control = lambda: True
    adapter._storage = lambda: Storage()
    adapter._residency_tensors = lambda keys: (
        torch.tensor(keys, dtype=torch.int64),
        torch.zeros(len(keys), dtype=torch.int64),
    )

    admitted, evicted = adapter.apply_residency_plan([867])

    initializer = module._dynamic_embedding_collection._embedding_tables._eval_initializers[0]
    assert admitted == [867]
    assert evicted == []
    assert calls == [(initializer, True, [867])]


def test_key_space_and_observed_ranges_keep_feature_namespaces_separate():
    adapter = EmbeddingAdapter(_EmbeddingModule())

    class Jagged:
        def __init__(self, values):
            self._values = torch.tensor(values)

        def values(self):
            return self._values

    class Features:
        def __getitem__(self, name):
            return {
                "user_id": Jagged([867]),
                "video_id": Jagged([12, 4_142_676]),
            }[name]

    ranges = adapter.batch_feature_key_ranges(SimpleNamespace(features=Features()))
    key_spaces = adapter.key_space_metadata()

    assert [item["table_name"] for item in key_spaces] == ["user_id", "video_id"]
    assert ranges["user_id"]["max_id"] == 867
    assert ranges["video_id"]["max_id"] == 4_142_676
