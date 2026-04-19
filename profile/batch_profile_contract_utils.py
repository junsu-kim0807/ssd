"""Small helpers shared by batch_profile (method aliases + verify layout). No torch/transformers."""

from __future__ import annotations


def _normalize_method(method: str) -> str:
    low = method.strip().lower()
    if low == "vanilla":
        return "vanila"
    if low not in {"vanila", "bump", "morphable", "topk_expansion"}:
        raise ValueError(f"Unsupported --method={method}. Use vanila, bump, morphable, or topk_expansion.")
    return low


def _normalize_bonus_method(bonus_method: str) -> str:
    low = bonus_method.strip().lower()
    if low not in {"proactive", "conservative", "adaptive"}:
        raise ValueError(f"Unsupported --bonus-method={bonus_method}. Use proactive, conservative, or adaptive.")
    return low


def _verify_full_seq_no_carry(prompt_ids: list[int], recovery_tok: int, draft_tokens: list[int]) -> list[int]:
    """``prompt + [recovery] + draft`` for parallel inter/target verify (no carry-over prefix)."""
    return list(prompt_ids) + [int(recovery_tok)] + list(draft_tokens)


def _verify_positions_no_carry(prompt_len: int, num_draft_tokens: int) -> list[int]:
    """Logit gather indices for ``_batched_logits_at_positions`` on ``_verify_full_seq_no_carry`` (no carry)."""
    return list(range(prompt_len - 1, prompt_len + int(num_draft_tokens) + 1))
