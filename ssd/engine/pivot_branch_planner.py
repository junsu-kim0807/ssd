from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class PivotExpansionConfig:
    policy: Literal["static", "dynamic"] = "dynamic"
    criteria: Literal["top1", "residual"] = "residual"
    expansion_pct: float = 0.0
    threshold: float = 0.8
    topk: int = 5


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


def _validate_cfg(cfg: PivotExpansionConfig) -> None:
    if cfg.policy not in {"static", "dynamic"}:
        raise ValueError("policy must be one of {'static', 'dynamic'}")
    if cfg.criteria not in {"top1", "residual"}:
        raise ValueError("criteria must be one of {'top1', 'residual'}")
    if not (0.0 <= float(cfg.expansion_pct) <= 1.0):
        raise ValueError("expansion_pct must be in [0, 1]")
    if not (0.0 <= float(cfg.threshold) <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if int(cfg.topk) < 1:
        raise ValueError("topk must be >= 1")


def _select_expand_mask(
    scores: torch.Tensor,
    cfg: PivotExpansionConfig,
) -> torch.Tensor:
    device = scores.device
    bsz = int(scores.numel())
    if bsz == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    expand_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
    if cfg.policy == "dynamic":
        return scores < float(cfg.threshold)

    num_expand = int(math.ceil(float(bsz) * float(cfg.expansion_pct)))
    num_expand = max(0, min(num_expand, bsz))
    if num_expand == 0:
        return expand_mask

    # Deterministic order: low score first, then lower parent index.
    order = sorted(range(bsz), key=lambda i: (float(scores[i].item()), i))
    keep = order[:num_expand]
    expand_mask[torch.tensor(keep, dtype=torch.int64, device=device)] = True
    return expand_mask


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
    requested = int(expand_mask.sum().item())
    if requested <= max_expand_reqs:
        return expand_mask

    true_indices = torch.nonzero(expand_mask, as_tuple=False).view(-1).tolist()
    # Keep lowest-score expanded requests first (more uncertain).
    keep_order = sorted(true_indices, key=lambda i: (float(criteria_scores[i].item()), i))
    kept = keep_order[:max_expand_reqs]

    out = torch.zeros_like(expand_mask, dtype=torch.bool)
    if kept:
        out[torch.tensor(kept, dtype=torch.int64, device=expand_mask.device)] = True
    return out


def build_pivot_expansion_plan(
    first_step_logits: torch.Tensor,  # [B, V]
    cfg: PivotExpansionConfig,
    *,
    max_expand_rows: int | None = None,
) -> PivotExpansionPlan:
    _validate_cfg(cfg)
    if first_step_logits.ndim != 2:
        raise ValueError(
            f"first_step_logits must have shape [B, V], got {tuple(first_step_logits.shape)}"
        )
    bsz, vocab_size = int(first_step_logits.shape[0]), int(first_step_logits.shape[1])
    device = first_step_logits.device
    topk = min(int(cfg.topk), max(1, vocab_size))

    probs = torch.softmax(first_step_logits.float(), dim=-1)
    top1_probs, _top1_ids = torch.topk(probs, k=1, dim=-1)
    top1_probs = top1_probs[:, 0]

    if vocab_size >= 2:
        top2_probs, _top2_ids = torch.topk(probs, k=2, dim=-1)
        residual_scores = top2_probs[:, 0] - top2_probs[:, 1]
    else:
        residual_scores = top1_probs.clone()

    criteria_scores = top1_probs if cfg.criteria == "top1" else residual_scores
    expand_mask = _select_expand_mask(criteria_scores, cfg)
    expand_mask = apply_capacity_limit(
        expand_mask,
        criteria_scores=criteria_scores,
        topk=topk,
        max_expand_rows=max_expand_rows,
    )

    topk_probs, topk_ids = torch.topk(probs, k=topk, dim=-1)
    parent_index_per_branch: list[int] = []
    branch_index_per_parent: list[int] = []
    root_token_ids: list[int] = []
    root_token_probs: list[float] = []
    branch_counts: list[int] = []

    for bi in range(bsz):
        count = topk if bool(expand_mask[bi].item()) else 1
        branch_counts.append(count)
        for bj in range(count):
            parent_index_per_branch.append(bi)
            branch_index_per_parent.append(bj)
            root_token_ids.append(int(topk_ids[bi, bj].item()))
            root_token_probs.append(float(topk_probs[bi, bj].item()))

    expanded_batch_size = len(parent_index_per_branch)
    return PivotExpansionPlan(
        parent_batch_size=bsz,
        expanded_batch_size=expanded_batch_size,
        expand_mask=expand_mask,
        parent_index_per_branch=torch.tensor(
            parent_index_per_branch, dtype=torch.int64, device=device
        ),
        branch_index_per_parent=torch.tensor(
            branch_index_per_parent, dtype=torch.int64, device=device
        ),
        root_token_ids=torch.tensor(root_token_ids, dtype=torch.int64, device=device),
        root_token_probs=torch.tensor(root_token_probs, dtype=torch.float32, device=device),
        criteria_scores=criteria_scores,
        top1_probs=top1_probs,
        residual_scores=residual_scores,
        branch_counts=branch_counts,
    )
