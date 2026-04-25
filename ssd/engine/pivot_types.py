from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssd.engine.sequence import Sequence


@dataclass
class BranchForkState:
    parent_seq_idx: int
    branch_idx: int
    root_token_id: int
    root_confidence: float
    target_shared_prefix_blocks: int
    draft_shared_prefix_blocks: int
    inter_shared_prefix_blocks: int
    draft_private_tail_block_ids: list[int]
    target_private_tail_block_ids: list[int]
    inter_private_tail_block_ids: list[int]
    is_parent_inplace: bool = False


@dataclass
class PivotBranchBundle:
    parent_batch_size: int
    parent_index_per_branch: list[int]
    branch_index_per_parent: list[int]
    branch_counts: list[int]
    root_token_ids: list[int]
    root_token_probs: list[float]
    criteria_scores: list[float] | None = None
    top1_probs: list[float] | None = None
    residual_scores: list[float] | None = None
    expanded_seqs: list["Sequence"] | None = None
    branch_states: list[BranchForkState] | None = None


@dataclass
class PivotCollapseDecision:
    winning_branch_idx_per_parent: list[int]
    accept_len_per_branch: list[int]
    winning_accept_len_per_parent: list[int]
    winning_root_token_per_parent: list[int]
