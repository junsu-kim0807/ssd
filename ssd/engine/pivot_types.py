from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
