from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class PivotExpansionConfig:
    policy: Literal["static", "dynamic", "dynamic_expansion"] = "dynamic"
    criteria: Literal["top1", "residual", "softmax_residual"] = "residual"
    # User-facing threshold in [0, 1]:
    # - top1: top-1 probability (binary proxy), converted to logit-margin threshold
    # - residual: (p1-p2) under binary top1-vs-top2 proxy, converted to logit-margin threshold
    # - softmax_residual: full-vocab softmax p_top1 - p_top2; compared directly (no logit conversion)
    threshold: float = 0.8
    expansion_pct: float = 0.0
    topk: int = 5
    # For top1/residual: logit-margin-domain threshold after conversion.
    # For softmax_residual: set equal to ``threshold`` for bookkeeping only; dynamic expansion
    # uses ``_score_threshold`` (user probability-difference), not logit space.
    logit_threshold: float = field(init=False)
    # dynamic_expansion only: strictly increasing thresholds for (p_top5-p_top2)/3 slope buckets.
    slope_thresholds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in {"static", "dynamic", "dynamic_expansion"}:
            raise ValueError(
                "policy must be one of {'static', 'dynamic', 'dynamic_expansion'}"
            )
        if self.criteria not in {"top1", "residual", "softmax_residual"}:
            raise ValueError(
                "criteria must be one of {'top1', 'residual', 'softmax_residual'}"
            )
        if not (0.0 <= float(self.expansion_pct) <= 1.0):
            raise ValueError("expansion_pct must be in [0, 1]")
        if not (0.0 <= float(self.threshold) <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if int(self.topk) < 1:
            raise ValueError("topk must be >= 1")

        if self.policy == "dynamic_expansion":
            if int(self.topk) != 5:
                raise ValueError("dynamic_expansion currently requires pivot_topk == 5")
            st = tuple(float(x) for x in self.slope_thresholds)
            n = len(st)
            if not (1 <= n <= int(self.topk) - 2):
                raise ValueError(
                    "len(slope_thresholds) must be in [1, topk - 2] for dynamic_expansion"
                )
            if n >= 2:
                if any(st[i] >= st[i + 1] for i in range(n - 1)):
                    raise ValueError("slope_thresholds must be strictly increasing")
            if self.criteria != "softmax_residual":
                raise ValueError(
                    "dynamic_expansion requires criteria='softmax_residual' "
                    "(selection score domain must match slope domain)"
                )

        if self.criteria == "top1":
            self.logit_threshold = _top1_prob_to_logit_margin_threshold(float(self.threshold))
        elif self.criteria == "residual":
            self.logit_threshold = _residual_prob_to_logit_margin_threshold(float(self.threshold))
        else:
            # softmax_residual: compare criteria_scores (p_top1 - p_top2) directly to threshold.
            self.logit_threshold = float(self.threshold)


@dataclass
class PivotHostPlan:
    """Host-side mirror of a pivot plan.

    All fields are plain Python lists materialized from GPU tensors via a
    single batched transfer. Reused across capacity clamp, branch construction,
    and bundle output to avoid scattered ``.tolist()`` / ``.item()`` calls.
    """
    parent_index_per_branch: list[int]
    branch_index_per_parent: list[int]
    root_token_ids: list[int]
    branch_counts: list[int]
    expand_mask: list[bool]
    # ``criteria_scores`` is required by capacity clamping (deterministic ordering)
    # so it is always materialized. Other profile fields are optional.
    criteria_scores: list[float] | None = None
    root_token_probs: list[float] | None = None
    top1_probs: list[float] | None = None
    residual_scores: list[float] | None = None
    # dynamic_expansion: per-parent (p_top5 - p_top2) / 3 on full-vocab softmax mass; None otherwise.
    dynamic_expansion_slope_scores: list[float] | None = None


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
    # [B] when policy==dynamic_expansion else None
    dynamic_expansion_slope_scores: torch.Tensor | None = None
    host: PivotHostPlan | None = None


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


def _score_threshold(cfg: PivotExpansionConfig) -> float:
    if cfg.criteria == "softmax_residual":
        return float(cfg.threshold)
    return float(cfg.logit_threshold)


def _select_expand_mask(
    scores: torch.Tensor,
    cfg: PivotExpansionConfig,
) -> torch.Tensor:
    device = scores.device
    bsz = int(scores.numel())
    if bsz == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    if cfg.policy in {"dynamic", "dynamic_expansion"}:
        candidates = scores < _score_threshold(cfg)
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


def _dynamic_expansion_branch_counts_from_slope(
    slope_scores: torch.Tensor,
    expand_mask: torch.Tensor,
    cfg: PivotExpansionConfig,
) -> torch.Tensor:
    """Map slope to per-parent branch count; non-expanded parents get 1."""
    device = slope_scores.device
    bsz = int(slope_scores.numel())
    thresholds = torch.tensor(
        list(cfg.slope_thresholds),
        dtype=torch.float32,
        device=device,
    )
    # Semantics: slope <= T[0] -> 2; T[i-1] < slope <= T[i] -> i+2; slope > T[-1] -> topk.
    bucket = torch.bucketize(slope_scores.float(), thresholds, right=False)
    n = len(cfg.slope_thresholds)
    topk = int(cfg.topk)
    branch_options = list(range(2, 2 + n)) + [topk]
    branch_options_t = torch.tensor(branch_options, dtype=torch.int64, device=device)
    counts = branch_options_t[bucket]
    return torch.where(
        expand_mask,
        counts,
        torch.ones(bsz, dtype=torch.int64, device=device),
    )


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
    materialize_host: bool = True,
    profile_metadata: bool = False,
) -> PivotExpansionPlan:
    """Build expansion indices and optional host mirror.

    When ``profile_metadata`` is False (normal decode / cost-only profiling), skip
    per-branch root probability gather; keep a zero tensor for ``root_token_probs``
    shape compatibility.

    Expansion scores: for ``softmax_residual``, full-vocab ``p_top1 - p_top2`` from
    ``logsumexp``; otherwise binary logit margin ``logit_top1 - logit_top2`` (top-2
    slice). Candidate token ids still come from ``torch.topk`` on logits.
    """
    if first_step_logits.ndim != 2:
        raise ValueError(
            f"first_step_logits must have shape [B, V], got {tuple(first_step_logits.shape)}"
        )
    bsz, vocab_size = int(first_step_logits.shape[0]), int(first_step_logits.shape[1])
    device = first_step_logits.device
    topk = min(int(cfg.topk), max(1, vocab_size))

    if cfg.policy == "dynamic_expansion":
        if vocab_size < 5:
            raise ValueError(
                "dynamic_expansion requires vocab_size >= 5 for top-5 softmax statistics"
            )
        topk_eff = min(max(int(cfg.topk), 5), vocab_size)
    else:
        # Need top-2 for logit margin / full-softmax residual and top-k for root candidates.
        topk_eff = min(max(topk, 2), vocab_size)

    logits_f = first_step_logits.float()
    top_vals_all, top_ids_all = torch.topk(logits_f, k=topk_eff, dim=-1)

    if vocab_size >= 2:
        logit_margin_scores = top_vals_all[:, 0] - top_vals_all[:, 1]
    else:
        logit_margin_scores = torch.full((bsz,), float("inf"), dtype=torch.float32, device=device)

    need_softmax_top_probs = (
        cfg.criteria == "softmax_residual" or cfg.policy == "dynamic_expansion"
    )
    log_z: torch.Tensor | None = None
    p1: torch.Tensor | None = None
    p2: torch.Tensor | None = None
    p5: torch.Tensor | None = None
    if need_softmax_top_probs and vocab_size >= 2:
        log_z = torch.logsumexp(logits_f, dim=-1)
        p1 = torch.exp(top_vals_all[:, 0] - log_z)
        p2 = torch.exp(top_vals_all[:, 1] - log_z)
        if cfg.policy == "dynamic_expansion":
            p5 = torch.exp(top_vals_all[:, 4] - log_z)

    if cfg.criteria == "softmax_residual":
        if vocab_size >= 2:
            assert p1 is not None and p2 is not None
            criteria_scores = p1 - p2
        else:
            criteria_scores = torch.full((bsz,), 1.0, dtype=torch.float32, device=device)
    else:
        criteria_scores = logit_margin_scores

    expand_mask = _select_expand_mask(criteria_scores, cfg)
    if cfg.policy != "dynamic_expansion":
        expand_mask = apply_capacity_limit(
            expand_mask,
            criteria_scores=criteria_scores,
            topk=topk,
            max_expand_rows=max_expand_rows,
        )
    elif max_expand_rows is not None:
        raise ValueError(
            "dynamic_expansion row budget must be enforced in host clamp only; "
            "pass max_expand_rows=None into build_pivot_expansion_plan"
        )

    slope_tensor: torch.Tensor | None = None
    if cfg.policy == "dynamic_expansion":
        assert p2 is not None and p5 is not None
        slope_tensor = ((p5 - p2) / 3.0).to(torch.float32)
        branch_counts_t = _dynamic_expansion_branch_counts_from_slope(
            slope_tensor, expand_mask, cfg
        )
    else:
        branch_counts_t = torch.where(
            expand_mask,
            torch.full((bsz,), topk, dtype=torch.int64, device=device),
            torch.ones((bsz,), dtype=torch.int64, device=device),
        )

    topk_ids = top_ids_all[:, :topk]
    if profile_metadata:
        if cfg.criteria == "softmax_residual":
            if vocab_size >= 2:
                assert p1 is not None and p2 is not None
                top1_probs = p1
                residual_scores = p1 - p2
            else:
                top1_probs = torch.ones((bsz,), dtype=torch.float32, device=device)
                residual_scores = torch.ones((bsz,), dtype=torch.float32, device=device)
        else:
            top1_probs = torch.sigmoid(logit_margin_scores)
            residual_scores = torch.tanh(logit_margin_scores / 2.0)
    else:
        # Same tensor as expansion scores; host/metadata paths do not materialize these when off.
        top1_probs = criteria_scores
        residual_scores = criteria_scores
    parent_ids = torch.arange(bsz, dtype=torch.int64, device=device)
    # NOTE: avoid ``output_size=...`` fast-path here. Some environments hit
    # Repeat.cu device-asserts when provided output_size mismatches internal
    # cumsum bookkeeping. Let ATen derive the output shape from repeats.
    parent_index_per_branch = torch.repeat_interleave(parent_ids, branch_counts_t)
    expanded_batch_size = int(parent_index_per_branch.numel())
    cum = torch.cumsum(branch_counts_t, dim=0)
    starts = cum - branch_counts_t
    starts_per_branch = torch.repeat_interleave(starts, branch_counts_t)
    branch_index_per_parent = (
        torch.arange(expanded_batch_size, dtype=torch.int64, device=device)
        - starts_per_branch
    )
    root_token_ids = topk_ids[parent_index_per_branch, branch_index_per_parent]
    # Profiling / trace only. For softmax_residual + full vocab, use global softmax prob
    # of the chosen root token; else top-k local softmax over the candidate slice.
    if profile_metadata:
        if cfg.criteria == "softmax_residual" and vocab_size >= 2:
            assert log_z is not None
            log_z_exp = log_z[parent_index_per_branch]
            root_logits = top_vals_all[
                parent_index_per_branch, branch_index_per_parent
            ]
            root_token_probs = torch.exp(root_logits - log_z_exp).to(torch.float32)
        else:
            topk_probs = torch.softmax(top_vals_all[:, :topk], dim=-1)
            root_token_probs = topk_probs[parent_index_per_branch, branch_index_per_parent]
    else:
        root_token_probs = torch.zeros(
            expanded_batch_size, dtype=torch.float32, device=device
        )

    host: PivotHostPlan | None = None
    if materialize_host:
        # Batch all int-typed fields into one D2H copy, and all float-typed
        # fields into another. Splitting back is O(B) on host. This collapses
        # what would otherwise be ~6-9 separate ``.tolist()`` syncs into 2
        # (or 3 when profile metadata is on for [B_exp]-shaped probs).
        b_exp = expanded_batch_size

        # ---- int fields: indices [B_exp]*3 + branch_counts [B] + expand_mask [B]
        int_blob = torch.empty(3 * b_exp + 2 * bsz, dtype=torch.int64, device=device)
        int_blob[0:b_exp] = parent_index_per_branch
        int_blob[b_exp:2 * b_exp] = branch_index_per_parent
        int_blob[2 * b_exp:3 * b_exp] = root_token_ids
        int_blob[3 * b_exp:3 * b_exp + bsz] = branch_counts_t
        int_blob[3 * b_exp + bsz:3 * b_exp + 2 * bsz] = expand_mask.to(torch.int64)
        int_host = int_blob.tolist()

        host_parent_index = int_host[0:b_exp]
        host_branch_index = int_host[b_exp:2 * b_exp]
        host_root_ids = int_host[2 * b_exp:3 * b_exp]
        host_branch_counts = int_host[3 * b_exp:3 * b_exp + bsz]
        host_expand_mask = [bool(x) for x in int_host[3 * b_exp + bsz:3 * b_exp + 2 * bsz]]

        # ---- float fields: criteria_scores [B] (always); profile fields [B] when on
        if profile_metadata:
            float_blob = torch.empty(3 * bsz, dtype=torch.float32, device=device)
            float_blob[0:bsz] = criteria_scores.to(torch.float32)
            float_blob[bsz:2 * bsz] = top1_probs.to(torch.float32)
            float_blob[2 * bsz:3 * bsz] = residual_scores.to(torch.float32)
            float_host = float_blob.tolist()
            host_criteria = float_host[0:bsz]
            host_top1 = float_host[bsz:2 * bsz]
            host_residual = float_host[2 * bsz:3 * bsz]
            # ``root_token_probs`` is shaped [B_exp], can't fit into [B]-blob; one extra.
            host_root_probs = root_token_probs.to(torch.float32).tolist()
        else:
            host_criteria = criteria_scores.to(torch.float32).tolist()
            host_top1 = None
            host_residual = None
            host_root_probs = None

        host_slope: list[float] | None = None
        if cfg.policy == "dynamic_expansion" and slope_tensor is not None:
            host_slope = slope_tensor.detach().cpu().tolist()

        host = PivotHostPlan(
            parent_index_per_branch=host_parent_index,
            branch_index_per_parent=host_branch_index,
            root_token_ids=host_root_ids,
            branch_counts=host_branch_counts,
            expand_mask=host_expand_mask,
            criteria_scores=host_criteria,
            root_token_probs=host_root_probs,
            top1_probs=host_top1,
            residual_scores=host_residual,
            dynamic_expansion_slope_scores=host_slope,
        )
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
        branch_counts=(host.branch_counts if host is not None else branch_counts_t.tolist()),
        branch_counts_tensor=branch_counts_t,
        dynamic_expansion_slope_scores=slope_tensor,
        host=host,
    )
