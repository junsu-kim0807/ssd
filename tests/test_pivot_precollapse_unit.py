"""Lightweight tests for pivot_precollapse types and policy traits."""

import pytest
import torch

from ssd.engine.pivot_branch_planner import compute_dynamic_expansion_slope
from ssd.engine.pivot_types import PivotPrecollapseDecision
from ssd.engine.spec_policy_traits import uses_pivot_precollapse, uses_pivot_root_expansion


def _rank_from_slope_for_test(slope: float, thresholds: tuple[float, float, float]) -> int:
    """Mirrors ``PivotPrecollapseSpeculatorSync._rank_from_slope`` (keeps tests import-light)."""
    t0, t1, t2 = thresholds
    if slope < t0:
        return 2
    if slope < t1:
        return 3
    if slope < t2:
        return 4
    return 5


def test_uses_pivot_precollapse_trait():
    assert uses_pivot_precollapse("pivot_precollapse")
    assert not uses_pivot_precollapse("pivot")
    assert uses_pivot_root_expansion("pivot_precollapse")


def test_pivot_precollapse_decision_branch_score_len():
    d = PivotPrecollapseDecision(
        winning_branch_idx_per_parent=[0, 1],
        winning_expanded_row_idx_per_parent=[0, 2],
        winning_root_token_per_parent=[10, 20],
        branch_score_per_row=[-1.0, -2.0, -3.0],
        winning_score_per_parent=[-1.0, -3.0],
        branch_count_per_parent=[1, 2],
        before_expansion_batch_size=2,
        after_expansion_batch_size=3,
        committed_len_per_parent=[5, 6],
    )
    assert len(d.branch_score_per_row) == 3
    assert len(d.winning_score_per_parent) == 2
    assert d.committed_len_per_parent == [5, 6]
    assert d.selected_root_rank_per_parent is None
    assert d.slope_score_per_parent is None


def test_pivot_precollapse_decision_slope_metadata():
    d = PivotPrecollapseDecision(
        winning_branch_idx_per_parent=[0],
        winning_expanded_row_idx_per_parent=[0],
        winning_root_token_per_parent=[42],
        branch_score_per_row=[0.0],
        winning_score_per_parent=[0.0],
        branch_count_per_parent=[1],
        before_expansion_batch_size=1,
        after_expansion_batch_size=1,
        selected_root_rank_per_parent=[3],
        slope_score_per_parent=[-0.6],
    )
    assert d.selected_root_rank_per_parent == [3]
    assert d.slope_score_per_parent == [-0.6]


def test_rank_from_slope_boundaries():
    th = (-0.70, -0.58, -0.46)
    assert _rank_from_slope_for_test(-0.71, th) == 2
    assert _rank_from_slope_for_test(-0.70, th) == 3
    assert _rank_from_slope_for_test(-0.59, th) == 3
    assert _rank_from_slope_for_test(-0.58, th) == 4
    assert _rank_from_slope_for_test(-0.47, th) == 4
    assert _rank_from_slope_for_test(-0.46, th) == 5
    assert _rank_from_slope_for_test(0.0, th) == 5


def test_compute_dynamic_expansion_slope_matches_formula():
    # V=5, topk=5: full softmax == softmax over these logits
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0]],
        dtype=torch.float32,
    )
    slope, top_vals, top_ids = compute_dynamic_expansion_slope(logits, topk=5)
    log_z = torch.logsumexp(logits, dim=-1)
    p2 = torch.exp(top_vals[:, 1] - log_z)
    p5 = torch.exp(top_vals[:, 4] - log_z)
    expected = ((p5 - p2) / 3.0).to(torch.float32)
    assert torch.allclose(slope, expected)
    assert top_ids.shape == (1, 5)


def test_compute_dynamic_expansion_slope_rejects_small_topk():
    logits = torch.zeros(1, 5, dtype=torch.float32)
    with pytest.raises(ValueError, match="topk must be >= 3"):
        compute_dynamic_expansion_slope(logits, topk=2)
