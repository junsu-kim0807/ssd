"""Unit tests for pivot ``dynamic_expansion`` slope bucketing and variable row cap."""

from __future__ import annotations

import torch

import pytest

from ssd.engine.pivot_branch_planner import (
    PivotExpansionConfig,
    _dynamic_expansion_branch_counts_from_slope,
    build_pivot_expansion_plan,
)
from ssd.engine.pivot_speculator_sync import PivotRootSpeculatorSync


def test_dynamic_expansion_bucketize_boundary_table() -> None:
    """Plan-required semantics with ``thresholds = [-0.06, -0.05]`` (``bucketize``, ``right=False``)."""
    cfg = PivotExpansionConfig(
        policy="dynamic_expansion",
        criteria="softmax_residual",
        threshold=0.0,
        expansion_pct=1.0,
        topk=5,
        slope_thresholds=(-0.06, -0.05),
    )
    slopes = torch.tensor([-0.0600, -0.0550, -0.0500, -0.0499], dtype=torch.float32)
    expand = torch.ones(4, dtype=torch.bool)
    counts = _dynamic_expansion_branch_counts_from_slope(slopes, expand, cfg)
    assert counts.tolist() == [2, 3, 3, 5]


def test_dynamic_expansion_custom_slope_branch_counts() -> None:
    """Optional ``slope_branch_counts`` maps buckets to 2,3,5,topk instead of 2,3,4,topk."""
    cfg = PivotExpansionConfig(
        policy="dynamic_expansion",
        criteria="softmax_residual",
        threshold=0.0,
        expansion_pct=1.0,
        topk=10,
        slope_thresholds=(-0.06, -0.05, -0.04),
        slope_branch_counts=(2, 3, 5, 10),
    )
    slopes = torch.tensor([-0.0700, -0.0550, -0.0450, -0.0350], dtype=torch.float32)
    expand = torch.ones(4, dtype=torch.bool)
    counts = _dynamic_expansion_branch_counts_from_slope(slopes, expand, cfg)
    assert counts.tolist() == [2, 3, 5, 10]


def test_non_expanded_parents_get_branch_count_one() -> None:
    cfg = PivotExpansionConfig(
        policy="dynamic_expansion",
        criteria="softmax_residual",
        threshold=0.0,
        expansion_pct=1.0,
        topk=5,
        slope_thresholds=(-0.06,),
    )
    slopes = torch.tensor([0.0, -0.07], dtype=torch.float32)
    expand = torch.tensor([True, False], dtype=torch.bool)
    counts = _dynamic_expansion_branch_counts_from_slope(slopes, expand, cfg)
    assert counts.tolist() == [2, 1]


def test_variable_row_cap_uniform_topk_matches_legacy_fixed_extra() -> None:
    """When every expanded parent uses ``branch_counts[i] == topk``, extras match legacy ``topk-1``."""
    bsz = 6
    topk = 5
    kept = [True, True, True, False, True, True]
    scores = [0.5, 0.1, 0.3, 0.0, 0.2, 0.4]
    branch_uniform = [topk] * bsz
    max_rows = bsz + 8

    obj = PivotRootSpeculatorSync.__new__(PivotRootSpeculatorSync)
    obj.max_expand_rows = max_rows
    got = PivotRootSpeculatorSync._apply_variable_row_cap_host(
        obj, kept, scores, branch_uniform
    )

    eps = float(torch.finfo(torch.float32).eps)
    order = [i for i, k in enumerate(kept) if k]
    order.sort(key=lambda i: (scores[i] + i * eps, i))
    budget = max_rows - bsz
    legacy = [False] * bsz
    used = 0
    fixed_extra = topk - 1
    for i in order:
        if fixed_extra == 0:
            legacy[i] = True
            continue
        if used + fixed_extra <= budget:
            legacy[i] = True
            used += fixed_extra

    assert got == legacy


def test_dynamic_expansion_rejects_planner_row_cap() -> None:
    cfg = PivotExpansionConfig(
        policy="dynamic_expansion",
        criteria="softmax_residual",
        threshold=0.0,
        expansion_pct=1.0,
        topk=5,
        slope_thresholds=(-0.06, -0.05),
    )
    logits = torch.zeros(2, 32, dtype=torch.float32)
    with pytest.raises(ValueError, match="host clamp"):
        build_pivot_expansion_plan(
            logits,
            cfg,
            max_expand_rows=100,
            materialize_host=False,
            profile_metadata=False,
        )


def test_variable_row_cap_stable_tie_break_orders_by_index() -> None:
    """Tie on ``scores`` uses ``scores[i] + i * eps`` then index, matching planner ``_cap_low_scores``."""
    obj = PivotRootSpeculatorSync.__new__(PivotRootSpeculatorSync)
    obj.max_expand_rows = 8
    bsz = 4
    kept = [True] * bsz
    scores = [0.0, 0.0, 0.0, 0.0]
    branch_counts = [5, 5, 5, 5]
    out = PivotRootSpeculatorSync._apply_variable_row_cap_host(
        obj, kept, scores, branch_counts
    )
    budget = 8 - 4
    assert budget == 4
    assert out == [True, False, False, False]


def test_variable_row_cap_mixed_branch_counts_budget() -> None:
    obj = PivotRootSpeculatorSync.__new__(PivotRootSpeculatorSync)
    obj.max_expand_rows = 10
    bsz = 4
    kept = [True, True, True, True]
    scores = [0.0, 0.1, 0.2, 0.3]
    branch_counts = [5, 2, 5, 2]
    out = PivotRootSpeculatorSync._apply_variable_row_cap_host(
        obj, kept, scores, branch_counts
    )
    budget = 10 - 4
    assert sum(max(0, branch_counts[i] - 1) for i, k in enumerate(out) if k) <= budget
    assert out == [True, True, False, True]
