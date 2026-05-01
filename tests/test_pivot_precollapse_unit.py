"""Lightweight tests for pivot_precollapse types and policy traits."""

from ssd.engine.pivot_types import PivotPrecollapseDecision
from ssd.engine.spec_policy_traits import uses_pivot_precollapse, uses_pivot_root_expansion


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
