# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import List

import pytest
import torch
from dynamicemb.scored_hashtable import (
    ScoreArg,
    ScoreSpec,
    get_scored_table,
)
from dynamicemb_extensions import (
    InsertResult,
    ScorePolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_score_step = 0


def _get_scores(score_policy, keys):
    batch = keys.numel()
    device = keys.device
    global _score_step
    _score_step += 1
    if score_policy == ScorePolicy.ASSIGN:
        return torch.empty(batch, dtype=torch.uint64, device=device).fill_(
            _score_step
        )
    elif score_policy == ScorePolicy.ACCUMULATE:
        return torch.ones(batch, dtype=torch.uint64, device=device)
    else:
        return torch.zeros(batch, dtype=torch.uint64, device=device)


def _make_table(capacity, bucket_capacity=128, key_type=torch.int64,
                score_specs=None, device=None):
    if device is None:
        device = torch.cuda.current_device()
    if score_specs is None:
        score_specs = [ScoreSpec(name="score1", policy=ScorePolicy.CONST)]

    try:
        return get_scored_table(
            capacity=[capacity],
            bucket_capacity=bucket_capacity,
            key_type=key_type,
            score_specs=score_specs,
            device=device,
        )
    except (TypeError, AssertionError):
        return get_scored_table(
            capacity=capacity,
            bucket_capacity=bucket_capacity,
            key_type=key_type,
            score_specs=score_specs,
            device=device,
        )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score_policy", [
    ScorePolicy.ASSIGN,
    ScorePolicy.ACCUMULATE,
    ScorePolicy.GLOBAL_TIMER,
])
def test_single_score_backward_compat(score_policy):
    """A table with exactly one ScoreSpec behaves identically to the old code."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    bucket_capacity = 128
    table = _make_table(
        capacity=13 * bucket_capacity,
        bucket_capacity=bucket_capacity,
        score_specs=[ScoreSpec(name="score1", policy=score_policy)],
        device=device,
    )

    assert table.num_scores_ == 1
    assert table._extra_policies_list == []

    batch_size = 256
    keys = torch.randperm(batch_size, device=device, dtype=torch.int64)
    table_ids = torch.zeros(batch_size, dtype=torch.int64, device=device)

    score_arg = ScoreArg(name="score1", value=_get_scores(score_policy, keys))
    score_copy_0 = score_arg.value.clone()
    insert_results = torch.empty(batch_size, dtype=table.result_type, device=device)

    indices = table.insert(keys, table_ids, score_arg, insert_results)

    assert (
        (insert_results == InsertResult.INSERT.value)
        | (insert_results == InsertResult.ILLEGAL.value)
    ).all()

    # Re-insert — should all be ASSIGN
    score_arg_reinsert = ScoreArg(
        name="score1", value=_get_scores(score_policy, keys)
    )
    score_copy_1 = score_arg_reinsert.value.clone()
    scores_reinsert = torch.empty(batch_size, dtype=torch.int64, device=device)
    insert_results2 = torch.empty(batch_size, dtype=table.result_type, device=device)

    indices_reinsert = table.insert(
        keys, table_ids, score_arg_reinsert, insert_results2,
        score_out=scores_reinsert,
    )

    assert (insert_results2 == InsertResult.ASSIGN.value).all()
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    assert torch.equal(indices[valid], indices_reinsert[valid])

    # Lookup
    _, founds, indices_lookup = table.lookup(
        keys, table_ids,
        ScoreArg(name="score1", value=_get_scores(score_policy, keys),
                 policy=ScorePolicy.CONST),
    )
    assert founds.all()
    assert torch.equal(indices_lookup, indices)

    # Primary score correctness
    if score_policy == ScorePolicy.ASSIGN:
        assert torch.equal(
            scores_reinsert,
            score_arg_reinsert.value.to(torch.int64),
        )
    elif score_policy == ScorePolicy.ACCUMULATE:
        expected = (score_copy_0.to(torch.int64) +
                    score_copy_1.to(torch.int64))
        assert torch.equal(scores_reinsert, expected)
    else:  # GLOBAL_TIMER
        assert (scores_reinsert > 0).all()


@pytest.mark.parametrize("num_extra", [1, 2, 4])
def test_multi_score_creation_metadata(num_extra):
    """Verify num_scores_, scores_list, _extra_policies_list for N scores."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    specs = [ScoreSpec(name="primary", policy=ScorePolicy.ASSIGN)]
    for i in range(num_extra):
        policy = ScorePolicy.ASSIGN if i % 2 == 0 else ScorePolicy.ACCUMULATE
        specs.append(
            ScoreSpec(name=f"extra_{i}", policy=policy, is_reduction=False)
        )

    bucket_capacity = 128
    table = _make_table(
        capacity=bucket_capacity,
        bucket_capacity=bucket_capacity,
        score_specs=specs,
        device=device,
    )

    assert table.num_scores_ == len(specs)
    assert len(table.scores_list) == len(specs)
    for i in range(len(specs)):
        s = table.scores_list[i]
        assert s.dtype == torch.uint64
    assert table._extra_policies_list == [s.policy for s in specs[1:]]


@pytest.mark.parametrize("primary_policy", [
    ScorePolicy.ASSIGN,
    ScorePolicy.ACCUMULATE,
    ScorePolicy.GLOBAL_TIMER,
])
def test_multi_score_insert_lookup(primary_policy):
    """Insert keys with 3 scores, then verify all retrieved correctly."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    bucket_capacity = 128
    table = _make_table(
        capacity=8 * bucket_capacity,
        bucket_capacity=bucket_capacity,
        score_specs=[
            ScoreSpec(name="primary", policy=primary_policy),
            ScoreSpec(name="extra_a", policy=ScorePolicy.ASSIGN,
                       is_reduction=False),
            ScoreSpec(name="extra_b", policy=ScorePolicy.ACCUMULATE,
                       is_reduction=False),
        ],
        device=device,
    )

    assert table.num_scores_ == 3

    batch = 200
    keys = torch.randperm(batch, device=device, dtype=torch.int64)
    tids = torch.zeros(batch, dtype=torch.int64, device=device)

    primary_val = _get_scores(primary_policy, keys)
    extra_a_val = torch.full((batch,), 100, dtype=torch.uint64, device=device)
    extra_b_val = torch.ones((batch,), dtype=torch.uint64, device=device)

    insert_results = torch.empty(batch, dtype=table.result_type, device=device)
    indices = table.insert(
        keys, tids,
        ScoreArg(name="primary", value=primary_val),
        insert_results,
        extra_scores=[
            ScoreArg(name="extra_a", value=extra_a_val),
            ScoreArg(name="extra_b", value=extra_b_val),
        ],
    )

    assert (insert_results == InsertResult.INSERT.value).all()

    # Read extra scores directly from scores_list using indices
    bucket_ids = indices // bucket_capacity
    slot_ids = indices % bucket_capacity

    extra_a_read = table.scores_list[1][bucket_ids, slot_ids]
    extra_b_read = table.scores_list[2][bucket_ids, slot_ids]

    assert torch.equal(extra_a_read.to(torch.uint64), extra_a_val)
    assert torch.equal(extra_b_read.to(torch.uint64), extra_b_val)

    # Re-insert same keys — extra_a (ASSIGN) stays at new value,
    # extra_b (ACCUMULATE) doubles
    extra_a_val2 = torch.full((batch,), 200, dtype=torch.uint64, device=device)
    extra_b_val2 = torch.ones((batch,), dtype=torch.uint64, device=device)

    insert_results2 = torch.empty(batch, dtype=table.result_type, device=device)
    indices2 = table.insert(
        keys, tids,
        ScoreArg(name="primary", value=_get_scores(primary_policy, keys)),
        insert_results2,
        extra_scores=[
            ScoreArg(name="extra_a", value=extra_a_val2),
            ScoreArg(name="extra_b", value=extra_b_val2),
        ],
    )

    assert (insert_results2 == InsertResult.ASSIGN.value).all()
    assert torch.equal(indices, indices2)

    bucket_ids2 = indices2 // bucket_capacity
    slot_ids2 = indices2 % bucket_capacity
    extra_a_read2 = table.scores_list[1][bucket_ids2, slot_ids2]
    extra_b_read2 = table.scores_list[2][bucket_ids2, slot_ids2]

    assert torch.equal(extra_a_read2.to(torch.uint64), extra_a_val2)
    expected_b = (extra_b_val + extra_b_val2).to(torch.uint64)
    assert torch.equal(extra_b_read2.to(torch.uint64), expected_b)

    # Lookup finds all keys
    _, founds, _ = table.lookup(
        keys, tids,
        ScoreArg(name="primary", value=primary_val, policy=ScorePolicy.CONST),
    )
    assert founds.all()


def test_extra_score_const_policy():
    """Extra score with Const policy should always remain 0."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    bucket_capacity = 128
    table = _make_table(
        capacity=8 * bucket_capacity,
        bucket_capacity=bucket_capacity,
        score_specs=[
            ScoreSpec(name="primary", policy=ScorePolicy.ASSIGN),
            ScoreSpec(name="extra_const", policy=ScorePolicy.CONST,
                       is_reduction=False),
        ],
        device=device,
    )

    batch = 128
    keys = torch.randperm(batch, device=device, dtype=torch.int64)
    tids = torch.zeros(batch, dtype=torch.int64, device=device)

    indices = table.insert(
        keys, tids,
        ScoreArg(name="primary", value=_get_scores(ScorePolicy.ASSIGN, keys)),
        extra_scores=[
            ScoreArg(name="extra_const",
                      value=torch.full((batch,), 99, dtype=torch.uint64,
                                       device=device)),
        ],
    )

    bucket_ids = indices // bucket_capacity
    slot_ids = indices % bucket_capacity
    extra_read = table.scores_list[1][bucket_ids, slot_ids]
    assert (extra_read == 0).all(), "Const policy should not update table slot"


def test_multi_score_eviction():
    """Primary (index-0) score drives eviction; extra scores preserved."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    bucket_capacity = 128
    capacity = bucket_capacity

    table = _make_table(
        capacity=capacity,
        bucket_capacity=bucket_capacity,
        score_specs=[
            ScoreSpec(name="primary", policy=ScorePolicy.ASSIGN),
            ScoreSpec(name="extra", policy=ScorePolicy.ASSIGN,
                       is_reduction=False),
        ],
        device=device,
    )

    # Phase 1: fill table to capacity with high primary scores
    n_fill = capacity
    fill_keys = torch.arange(1, n_fill + 1, device=device, dtype=torch.int64)
    fill_tids = torch.zeros(n_fill, dtype=torch.int64, device=device)
    fill_primary = torch.arange(n_fill, 0, -1, device=device, dtype=torch.uint64)
    fill_extra = torch.full((n_fill,), 42, dtype=torch.uint64, device=device)

    table.insert(
        fill_keys, fill_tids,
        ScoreArg(name="primary", value=fill_primary),
        extra_scores=[ScoreArg(name="extra", value=fill_extra)],
    )

    assert table.size(table_id=0) == n_fill

    # Phase 2: insert one more key — should evict lowest-primary-score key
    evict_key = torch.tensor([n_fill + 1], device=device, dtype=torch.int64)
    evict_tid = torch.tensor([0], device=device, dtype=torch.int64)
    evict_primary = torch.tensor([999], device=device, dtype=torch.uint64)
    evict_extra = torch.tensor([77], device=device, dtype=torch.uint64)

    (idx, num_ev, ev_keys, ev_idx, ev_scores,
     ev_tids) = table.insert_and_evict(
        evict_key, evict_tid,
        ScoreArg(name="primary", value=evict_primary),
        extra_scores=[ScoreArg(name="extra", value=evict_extra)],
    )

    assert num_ev == 1
    assert ev_keys[0].item() == n_fill
    assert ev_scores[0].item() == 1

    # Verify new key's extra score is 77
    _, founds_new, idx_new = table.lookup(
        evict_key, evict_tid,
        ScoreArg(name="primary", policy=ScorePolicy.CONST),
    )
    assert founds_new.all()
    bid = idx_new[0].item() // bucket_capacity
    sid = idx_new[0].item() % bucket_capacity
    assert table.scores_list[1][bid, sid].item() == 77


@pytest.mark.parametrize("primary_policy", [
    ScorePolicy.ASSIGN,
    ScorePolicy.ACCUMULATE,
])
def test_multi_score_dump_load(primary_policy):
    """Dump all 3 scores to files, load into new table, verify equality."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    bucket_capacity = 128
    capacity = 2 * bucket_capacity

    specs = [
        ScoreSpec(name="primary", policy=primary_policy),
        ScoreSpec(name="extra_a", policy=ScorePolicy.ASSIGN,
                   is_reduction=False),
        ScoreSpec(name="extra_b", policy=ScorePolicy.ACCUMULATE,
                   is_reduction=False),
    ]

    table = _make_table(
        capacity=capacity, bucket_capacity=bucket_capacity,
        score_specs=specs, device=device,
    )

    n_keys = 200
    keys = torch.randperm(n_keys, device=device, dtype=torch.int64)
    tids = torch.zeros(n_keys, dtype=torch.int64, device=device)

    primary_val = _get_scores(primary_policy, keys)
    extra_a_val = torch.arange(n_keys, device=device, dtype=torch.uint64)
    extra_b_val = torch.ones(n_keys, dtype=torch.uint64, device=device)

    insert_results = torch.empty(n_keys, dtype=table.result_type, device=device)
    indices = table.insert(
        keys, tids,
        ScoreArg(name="primary", value=primary_val),
        insert_results,
        extra_scores=[
            ScoreArg(name="extra_a", value=extra_a_val),
            ScoreArg(name="extra_b", value=extra_b_val),
        ],
    )

    inserted = insert_results == InsertResult.INSERT.value
    inserted_keys = keys[inserted]
    inserted_tids = tids[inserted]
    inserted_indices = indices[inserted]
    n_inserted = inserted.sum().item()

    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = os.path.join(tmpdir, "keys.bin")
        score_files = {
            "primary": os.path.join(tmpdir, "score_primary.bin"),
            "extra_a": os.path.join(tmpdir, "score_extra_a.bin"),
            "extra_b": os.path.join(tmpdir, "score_extra_b.bin"),
        }
        table.dump(key_file, score_files, table_id=0)

        load_table = _make_table(
            capacity=capacity, bucket_capacity=bucket_capacity,
            score_specs=specs, device=device,
        )
        load_table.load(key_file, score_files, table_id=0)

        assert load_table.size(table_id=0) == n_inserted

        _, founds, load_indices = load_table.lookup(
            inserted_keys, inserted_tids,
            ScoreArg(name="primary", policy=ScorePolicy.CONST),
        )
        assert founds.all()

        bid = load_indices // bucket_capacity
        sid = load_indices % bucket_capacity

        orig_extra_a = extra_a_val[inserted]
        orig_extra_b = extra_b_val[inserted]

        load_extra_a = load_table.scores_list[1][bid, sid]
        load_extra_b = load_table.scores_list[2][bid, sid]

        assert torch.equal(load_extra_a.to(torch.uint64), orig_extra_a)
        assert torch.equal(load_extra_b.to(torch.uint64), orig_extra_b)


def test_edge_extra_policies_with_single_score():
    """Passing extra_policies on a single-score table is a harmless no-op."""
    assert torch.cuda.is_available()
    device = torch.cuda.current_device()

    table = _make_table(
        capacity=8 * 128, bucket_capacity=128,
        score_specs=[ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)],
        device=device,
    )

    assert table._extra_policies_list == []

    keys = torch.randperm(128, device=device, dtype=torch.int64)
    tids = torch.zeros(128, dtype=torch.int64, device=device)

    indices = table.insert(
        keys, tids,
        ScoreArg(name="s", value=_get_scores(ScorePolicy.ASSIGN, keys)),
    )
    assert len(indices) == 128

    _, founds, _ = table.lookup(
        keys, tids,
        ScoreArg(name="s", policy=ScorePolicy.CONST),
    )
    assert founds.all()
