import torch

from ssd.engine.pivot_branch_planner import (
    PivotExpansionConfig,
    apply_capacity_limit,
    build_pivot_expansion_plan,
)


def _logits_from_probs(rows: list[list[float]]) -> torch.Tensor:
    # Any monotonic transform is valid; log(prob) makes softmax recover the row distribution.
    return torch.log(torch.tensor(rows, dtype=torch.float32))


def test_static_top1_pct_half_expands_low_scores():
    logits = _logits_from_probs(
        [
            [0.90, 0.10, 0.00],
            [0.55, 0.45, 0.00],
            [0.80, 0.20, 0.00],
            [0.51, 0.49, 0.00],
        ]
    )
    cfg = PivotExpansionConfig(policy="static", criteria="top1", expansion_pct=0.5, topk=2)
    plan = build_pivot_expansion_plan(logits, cfg)
    # lowest top1 probabilities: rows 3 and 1
    assert plan.expand_mask.tolist() == [False, True, False, True]
    assert plan.branch_counts == [1, 2, 1, 2]


def test_static_residual_pct_quarter_expands_one():
    logits = _logits_from_probs(
        [
            [0.90, 0.09, 0.01],  # residual 0.81
            [0.55, 0.44, 0.01],  # residual 0.11 (smallest)
            [0.80, 0.19, 0.01],  # residual 0.61
            [0.51, 0.39, 0.10],  # residual 0.12
        ]
    )
    cfg = PivotExpansionConfig(policy="static", criteria="residual", expansion_pct=0.25, topk=3)
    plan = build_pivot_expansion_plan(logits, cfg)
    assert plan.expand_mask.tolist() == [False, True, False, False]
    assert plan.expanded_batch_size == 4 + 1 * (3 - 1)


def test_dynamic_top1_threshold():
    logits = _logits_from_probs(
        [
            [0.90, 0.10],
            [0.70, 0.30],
            [0.79, 0.21],
        ]
    )
    cfg = PivotExpansionConfig(policy="dynamic", criteria="top1", threshold=0.8, topk=2)
    plan = build_pivot_expansion_plan(logits, cfg)
    assert plan.expand_mask.tolist() == [False, True, True]


def test_dynamic_residual_threshold():
    logits = _logits_from_probs(
        [
            [0.60, 0.39, 0.01],  # residual 0.21
            [0.52, 0.47, 0.01],  # residual 0.05
            [0.90, 0.09, 0.01],  # residual 0.81
        ]
    )
    cfg = PivotExpansionConfig(policy="dynamic", criteria="residual", threshold=0.2, topk=4)
    plan = build_pivot_expansion_plan(logits, cfg)
    assert plan.expand_mask.tolist() == [False, True, False]


def test_dynamic_softmax_residual_uses_full_vocab_p1_minus_p2():
    logits = _logits_from_probs(
        [
            [0.50, 0.30, 0.20],  # p1 - p2 = 0.2
            [0.90, 0.05, 0.05],  # 0.85
            [0.34, 0.33, 0.33],  # 0.01
        ]
    )
    cfg = PivotExpansionConfig(
        policy="dynamic", criteria="softmax_residual", threshold=0.25, topk=2
    )
    plan = build_pivot_expansion_plan(logits, cfg)
    assert plan.expand_mask.tolist() == [True, False, True]


def test_topk_rows_shape():
    logits = _logits_from_probs(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )
    cfg = PivotExpansionConfig(policy="dynamic", criteria="top1", threshold=0.99, topk=3)
    plan = build_pivot_expansion_plan(logits, cfg)
    assert plan.parent_batch_size == 2
    assert plan.expanded_batch_size == 2 * 3
    assert plan.parent_index_per_branch.shape[0] == 6
    assert plan.root_token_ids.shape[0] == 6


def test_capacity_clamp_reduces_expanded_requests():
    logits = _logits_from_probs(
        [
            [0.6, 0.3, 0.1],
            [0.55, 0.35, 0.1],
            [0.52, 0.38, 0.1],
        ]
    )
    cfg = PivotExpansionConfig(policy="dynamic", criteria="top1", threshold=0.99, topk=3)
    # B=3, topk=3, max rows=5 => max expanded reqs floor((5-3)/(3-1))=1
    plan = build_pivot_expansion_plan(logits, cfg, max_expand_rows=5)
    assert sum(plan.expand_mask.tolist()) == 1
    assert plan.expanded_batch_size == 5


def test_profile_metadata_off_skips_root_softmax_gathers():
    logits = _logits_from_probs([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    cfg = PivotExpansionConfig(policy="dynamic", criteria="top1", threshold=0.99, topk=3)
    plan = build_pivot_expansion_plan(logits, cfg, profile_metadata=False)
    assert int(plan.expanded_batch_size) > 0
    assert (plan.root_token_probs == 0).all()
    plan_m = build_pivot_expansion_plan(logits, cfg, profile_metadata=True)
    assert plan_m.root_token_probs.abs().sum().item() > 0.0


def test_tie_case_deterministic_order():
    scores = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    expand_mask = torch.tensor([True, True, True], dtype=torch.bool)
    # max rows allow exactly one expansion when topk=3, B=3: (5-3)/(3-1)=1
    clamped = apply_capacity_limit(
        expand_mask,
        criteria_scores=scores,
        topk=3,
        max_expand_rows=5,
    )
    # deterministic by parent index, keep first one
    assert clamped.tolist() == [True, False, False]
