from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class PivotExpansionConfig:
    policy: Literal["static", "dynamic"] = "dynamic"
    criteria: Literal["top1", "residual"] = "residual"
    # User-facing probability threshold:
    # - top1: p1 threshold
    # - residual: (p1 - p2) threshold
    threshold: float = 0.8
    expansion_pct: float = 0.0
    topk: int = 5
    # Internal hot-path threshold in logit-margin domain.
    logit_threshold: float = field(init=False)

    def __post_init__(self) -> None:
        if self.policy not in {"static", "dynamic"}:
            raise ValueError("policy must be one of {'static', 'dynamic'}")
        if self.criteria not in {"top1", "residual"}:
            raise ValueError("criteria must be one of {'top1', 'residual'}")
        if not (0.0 <= float(self.expansion_pct) <= 1.0):
            raise ValueError("expansion_pct must be in [0, 1]")
        if not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if int(self.topk) < 1:
            raise ValueError("topk must be >= 1")

        if self.criteria == "top1":
            self.logit_threshold = _top1_prob_to_logit_margin_threshold(float(self.threshold))
        else:
            self.logit_threshold = _residual_prob_to_logit_margin_threshold(float(self.threshold))


@dataclass
class PivotExpansionPlan:
    parent_batch_size: int
    expanded_batch_size: int
    expand_mask: torch.Tensor  # [B], bool
    parent_index_per_branch: torch.Tensor  # [B_exp], int64
    branch_index_per_parent: torch.Tensor  # [B_exp], int64
    root_token_ids: torch.Tensor  # [B_exp], int64
    root_token_probs: torch.Tensor  # [B_exp], float
    criteria_scores: torch.Tensor  # [B], float
    top1_probs: torch.Tensor  # [B], float
    residual_scores: torch.Tensor  # [B], float
    branch_counts: list[int]  # len B
    branch_counts_tensor: torch.Tensor  # [B], int64


def _clamp_prob_open_interval(p: float) -> float:
    eps = 1e-6
    return min(max(float(p), eps), 1.0 - eps)


def _top1_prob_to_logit_margin_threshold(p: float) -> float:
    p = _clamp_prob_open_interval(p)
    return math.log(p / (1.0 - p))


def _residual_prob_to_logit_margin_threshold(r: float) -> float:
    r = min(max(float(r), 0.0), 1.0 - 1e-6)
    return math.log((1.0 + r) / (1.0 - r))


def _cap_low_scores(
    candidate_mask: torch.Tensor,
    scores: torch.Tensor,
    max_expand: int,
) -> torch.Tensor:
    if max_expand <= 0:
        return torch.zeros_like(candidate_mask)

    bsz = int(scores.numel())
    if bsz == 0:
        return torch.zeros_like(candidate_mask)

    idx = torch.arange(bsz, device=scores.device, dtype=scores.dtype)
    stable_scores = scores + idx * torch.finfo(scores.dtype).eps
    masked_scores = torch.where(
        candidate_mask,
        stable_scores,
        torch.full_like(stable_scores, float("inf")),
    )

    k = min(max_expand, bsz)
    keep = torch.topk(-masked_scores, k=k, dim=0).indices

    out = torch.zeros_like(candidate_mask)
    valid = candidate_mask[keep]
    out[keep[valid]] = True
    return out


def _select_expand_mask(
    scores: torch.Tensor,
    cfg: PivotExpansionConfig,
) -> torch.Tensor:
    device = scores.device
    bsz = int(scores.numel())
    if bsz == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    if cfg.policy == "dynamic":
        candidates = scores < float(cfg.logit_threshold)
        if float(cfg.expansion_pct) <= 0.0:
            return candidates
        max_expand = int(math.floor(bsz * float(cfg.expansion_pct)))
        max_expand = max(0, min(max_expand, bsz))
        return _cap_low_scores(candidates, scores, max_expand)

    num_expand = int(math.ceil(float(bsz) * float(cfg.expansion_pct)))
    num_expand = max(0, min(num_expand, bsz))
    if num_expand == 0:
        return torch.zeros(bsz, dtype=torch.bool, device=device)

    candidates = torch.ones(bsz, dtype=torch.bool, device=device)
    return _cap_low_scores(candidates, scores, num_expand)


def apply_capacity_limit(
    expand_mask: torch.Tensor,
    *,
    criteria_scores: torch.Tensor,
    topk: int,
    max_expand_rows: int | None,
) -> torch.Tensor:
    if max_expand_rows is None:
        return expand_mask
    if topk <= 1:
        return torch.zeros_like(expand_mask, dtype=torch.bool)

    bsz = int(expand_mask.numel())
    max_rows = int(max_expand_rows)
    if max_rows <= bsz:
        return torch.zeros_like(expand_mask, dtype=torch.bool)

    max_expand_reqs = (max_rows - bsz) // (topk - 1)
    max_expand_reqs = max(0, min(max_expand_reqs, bsz))
    return _cap_low_scores(expand_mask, criteria_scores, max_expand_reqs)


def build_pivot_expansion_plan(
    first_step_logits: torch.Tensor,  # [B, V]
    cfg: PivotExpansionConfig,
    *,
    max_expand_rows: int | None = None,
) -> PivotExpansionPlan:
    if first_step_logits.ndim != 2:
        raise ValueError(
            f"first_step_logits must have shape [B, V], got {tuple(first_step_logits.shape)}"
        )
    bsz, vocab_size = int(first_step_logits.shape[0]), int(first_step_logits.shape[1])
    device = first_step_logits.device
    topk = min(int(cfg.topk), max(1, vocab_size))

    # Need top-2 for logit margin and top-k for root candidates.
    topk_eff = min(max(topk, 2), vocab_size)
    top_vals_all, top_ids_all = torch.topk(first_step_logits.float(), k=topk_eff, dim=-1)

    if vocab_size >= 2:
        logit_margin_scores = top_vals_all[:, 0] - top_vals_all[:, 1]
    else:
        logit_margin_scores = torch.full((bsz,), float("inf"), dtype=torch.float32, device=device)

    criteria_scores = logit_margin_scores
    expand_mask = _select_expand_mask(criteria_scores, cfg)
    expand_mask = apply_capacity_limit(
        expand_mask,
        criteria_scores=criteria_scores,
        topk=topk,
        max_expand_rows=max_expand_rows,
    )

    topk_ids = top_ids_all[:, :topk]
    # Profiling proxies without full-vocab softmax.
    top1_probs = torch.sigmoid(logit_margin_scores)
    residual_scores = torch.tanh(logit_margin_scores / 2.0)
    topk_probs = torch.softmax(top_vals_all[:, :topk], dim=-1)
    branch_counts_t = torch.where(
        expand_mask,
        torch.full((bsz,), topk, dtype=torch.int64, device=device),
        torch.ones((bsz,), dtype=torch.int64, device=device),
    )
    parent_ids = torch.arange(bsz, dtype=torch.int64, device=device)
    parent_index_per_branch = torch.repeat_interleave(parent_ids, branch_counts_t)
    cum = torch.cumsum(branch_counts_t, dim=0)
    starts = cum - branch_counts_t
    branch_index_per_parent = (
        torch.arange(parent_index_per_branch.numel(), dtype=torch.int64, device=device)
        - torch.repeat_interleave(starts, branch_counts_t)
    )
    root_token_ids = topk_ids[parent_index_per_branch, branch_index_per_parent]
    root_token_probs = topk_probs[parent_index_per_branch, branch_index_per_parent]
    expanded_batch_size = int(parent_index_per_branch.numel())
    return PivotExpansionPlan(
        parent_batch_size=bsz,
        expanded_batch_size=expanded_batch_size,
        expand_mask=expand_mask,
        parent_index_per_branch=parent_index_per_branch,
        branch_index_per_parent=branch_index_per_parent,
        root_token_ids=root_token_ids,
        root_token_probs=root_token_probs.to(torch.float32),
        criteria_scores=criteria_scores,
        top1_probs=top1_probs,
        residual_scores=residual_scores,
        branch_counts=branch_counts_t.cpu().tolist(),
        branch_counts_tensor=branch_counts_t,
    )
