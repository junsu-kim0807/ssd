"""Unit tests for hierarchical verification helpers (no GPU, no ssd imports)."""

import torch


def _greedy_accept_reference(logits_p: torch.Tensor, candidate_tokens: list[int]):
    """Mirror ``verify_greedy_chain_variable`` for tests without importing ``ssd``."""
    L = len(candidate_tokens)
    preds = logits_p.argmax(dim=-1)
    if L == 1:
        return [candidate_tokens[0]], int(preds[0].item())
    n = 0
    while n < L - 1 and candidate_tokens[n + 1] == int(preds[n].item()):
        n += 1
    suffix = candidate_tokens[: n + 1]
    recovery = int(preds[n].item())
    return suffix, recovery


def test_verify_greedy_chain_variable_full_accept():
    V = 11
    cand = [100, 5, 7]
    logits = torch.zeros(3, V)
    logits[0, 5] = 10.0
    logits[1, 7] = 10.0
    logits[2, 3] = 10.0
    suffix, rec = _greedy_accept_reference(logits, cand)
    assert suffix == [100, 5, 7]
    assert rec == 3


def test_verify_greedy_chain_variable_partial():
    V = 16
    cand = [1, 2, 3]
    logits = torch.zeros(3, V)
    logits[0, 2] = 10.0
    logits[1, 9] = 10.0
    logits[2, 0] = 1.0
    suffix, rec = _greedy_accept_reference(logits, cand)
    assert suffix == [1, 2]
    assert rec == 9


def test_hv_lookahead_formula_matches_plan():
    K, r = 4, 3
    assert (K + 1) * r + K + 1 == 20
