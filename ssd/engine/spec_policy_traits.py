"""Spec-policy trait helpers.

Keep policy-string decoding centralized so wiring code doesn't branch
directly on raw strings across the engine.
"""

from __future__ import annotations


def uses_hierarchical_verify(policy: str) -> bool:
    # ``pivot_hierarchical`` is reserved in traits for planned topology wiring,
    # but runtime execution is currently gated with NotImplementedError.
    return policy in {"hierarchical", "pivot_hierarchical"}


def uses_pivot_root_expansion(policy: str) -> bool:
    return policy in {"pivot", "pivot_tree_scratch", "pivot_hierarchical", "pivot_precollapse"}


def uses_pivot_precollapse(policy: str) -> bool:
    return policy == "pivot_precollapse"


def uses_pivot_tree_scratch(policy: str) -> bool:
    return policy == "pivot_tree_scratch"


def uses_intermediate_runner(policy: str) -> bool:
    return uses_hierarchical_verify(policy)


def uses_hv_postprocess(policy: str) -> bool:
    return uses_hierarchical_verify(policy)


def uses_target_varlen_verify(policy: str) -> bool:
    return uses_hierarchical_verify(policy)


def is_pivot_legacy(policy: str) -> bool:
    return policy == "pivot_legacy"


def pivot_max_branches(policy: str, pivot_topk: int, pivot_max_root_branches: int | None = None) -> int:
    if not uses_pivot_root_expansion(policy):
        return 1
    out = max(1, int(pivot_topk))
    if pivot_max_root_branches is not None:
        out = min(out, max(1, int(pivot_max_root_branches)))
    return out
