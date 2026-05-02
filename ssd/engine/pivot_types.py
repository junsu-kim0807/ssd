from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ssd.engine.helpers.pivot_tree_helpers import (
        DraftScratchPackedInputs,
        TargetScratchPackedInputs,
    )
    from ssd.engine.pivot_branch_planner import PivotHostPlan
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
    # When False on precollapse alt branches, commit/release must not touch target KV.
    target_kv_owned: bool = True
    draft_kv_owned: bool = True
    inter_kv_owned: bool = True


@dataclass
class PivotPrecollapseDecision:
    """Draft-score collapse metadata (expanded row indices, B_exp-sized scores)."""

    winning_branch_idx_per_parent: list[int]
    winning_expanded_row_idx_per_parent: list[int]
    winning_root_token_per_parent: list[int]
    branch_score_per_row: list[float]  # len == B_exp before collapse
    winning_score_per_parent: list[float]  # len == B
    branch_count_per_parent: list[int]
    before_expansion_batch_size: int
    after_expansion_batch_size: int
    score_method: str = "logprob_sum"
    # Per-parent ``num_tokens`` before recovery append (debug rollback asserts).
    committed_len_per_parent: list[int] | None = None
    # pivot_precollapse_selection == "slope": root rank in {2,3,4,5} (1-based top-k index).
    selected_root_rank_per_parent: list[int] | None = None
    slope_score_per_parent: list[float] | None = None


@dataclass
class PivotBranchBundle:
    """Bundle delivered from speculator to executor/verifier.

    Holds a reference to ``host_plan`` whose lists already cover all
    parent/branch/root data. Expose the legacy attribute names as properties so
    existing executor/verifier code keeps working without any per-step
    ``.tolist()`` conversion.
    """

    parent_batch_size: int
    host_plan: "PivotHostPlan"
    expanded_seqs: list["Sequence"] | None = None
    branch_states: list[BranchForkState] | None = None
    precollapse_decision: "PivotPrecollapseDecision | None" = None

    @property
    def parent_index_per_branch(self) -> list[int]:
        return self.host_plan.parent_index_per_branch

    @property
    def branch_index_per_parent(self) -> list[int]:
        return self.host_plan.branch_index_per_parent

    @property
    def branch_counts(self) -> list[int]:
        return self.host_plan.branch_counts

    @property
    def root_token_ids(self) -> list[int]:
        return self.host_plan.root_token_ids

    @property
    def root_token_probs(self) -> list[float] | None:
        return self.host_plan.root_token_probs

    @property
    def criteria_scores(self) -> list[float] | None:
        return self.host_plan.criteria_scores

    @property
    def top1_probs(self) -> list[float] | None:
        return self.host_plan.top1_probs

    @property
    def residual_scores(self) -> list[float] | None:
        return self.host_plan.residual_scores


@dataclass
class PivotCollapseDecision:
    winning_branch_idx_per_parent: list[int]
    accept_len_per_branch: list[int]
    winning_accept_len_per_parent: list[int]
    winning_root_token_per_parent: list[int]


@dataclass
class PivotTreeNode:
    node_id: int
    parent_seq_idx: int
    parent_node_id: int | None
    branch_idx: int
    depth: int
    token_id: int
    position: int
    draft_scratch_slot: tuple[int, int] | None = None
    target_scratch_slot: tuple[int, int] | None = None


@dataclass
class ScratchOwner:
    target_block_ids: list[int]
    draft_block_ids: list[int]
    released: bool = False

    def merge(self, other: "ScratchOwner | None") -> "ScratchOwner":
        if other is None:
            return self
        if other is self:
            return self
        self.target_block_ids.extend(other.target_block_ids)
        self.draft_block_ids.extend(other.draft_block_ids)
        return self

    def release_unreleased(self, target_block_manager, draft_block_manager) -> None:
        if self.released:
            return
        target_block_manager.release_scratch_blocks(self.target_block_ids)
        draft_block_manager.release_scratch_blocks(self.draft_block_ids)
        self.released = True


@dataclass
class PivotTreeCommitBundle:
    winner_target_node_ids: list[list[int]]
    winner_draft_node_ids: list[list[int]]
    target_node_slot: dict[int, tuple[int, int]]
    draft_node_slot: dict[int, tuple[int, int]]
    raw_suffix_lens: list[int]
    scratch_owner: ScratchOwner | None = None


@dataclass
class PivotTreeScratchBundle:
    parent_batch_size: int
    expanded_batch_size: int
    host_plan: "PivotHostPlan"
    nodes: list[PivotTreeNode]
    path_node_ids: list[list[int]]
    path_token_ids: list[list[int]]
    path_parent_seq_idx: list[int]
    path_branch_idx: list[int]
    logits_q: "torch.Tensor"
    target_node_to_slot: dict[int, tuple[int, int]]
    draft_node_to_slot: dict[int, tuple[int, int]]
    expanded_seqs: list["Sequence"] | None = None
    branch_states: list[BranchForkState] | None = None
    scratch_owner: ScratchOwner | None = None
    # Phase-1 target scratch packed verify (None in Phase-0 fallback).
    target_scratch_packed: "TargetScratchPackedInputs | None" = None
    # Phase-2 draft scratch packed metadata (None before draft scratch rollout).
    draft_scratch_packed: "DraftScratchPackedInputs | None" = None


@dataclass
class PivotPhase1BranchState:
    parent_seq_idx: int
    branch_idx: int
    root_token_id: int
    draft_shared_prefix_blocks: int
    draft_private_tail_block_ids: list[int]
    is_parent_inplace: bool
    target_scratch_node_ids: list[int]
    target_scratch_slots: list[tuple[int, int]]
