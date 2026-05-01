from dataclasses import dataclass
import torch
from ssd.engine.sequence import Sequence
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ssd.engine.pivot_types import PivotBranchBundle, PivotTreeCommitBundle, PivotTreeScratchBundle


@dataclass
class SpeculateResult:
    speculations: torch.Tensor
    logits_q: torch.Tensor
    cache_hits: torch.Tensor | None = None
    branch_bundle: "PivotBranchBundle | PivotTreeScratchBundle | None" = None


@dataclass
class VerifyProfileTrace:
    """Batch-shaped verification metadata for profiling (one list entry per request)."""

    verification_models: list[str]
    token_ids_per_position: list[list[int]]
    token_confidence_per_position: list[list[float]]
    recovery_tokens: list[int]
    bonus_tokens: list[int | None]
    # Target / pivot_target rounds: per-row greedy accept lengths (excludes recovery column).
    # Intermediate rounds: leave ``None`` (use ``inter_accept_len`` only) so downstream does not
    # treat placeholder zeros as target misspeculations.
    accept_len: list[int] | None = None
    # Optional hierarchical / pivot intermediate columns (per batch row; None if N/A)
    inter_token_ids_per_position: list[list[int] | None] | None = None
    inter_token_confidence_per_position: list[list[float] | None] | None = None
    inter_accept_len: list[int | None] | None = None
    inter_recovery_token: list[int | None] | None = None
    inter_bonus_token: list[int | None] | None = None
    # Target HV only: consecutive greedy matches along ``candidates`` restricted to
    # indices ``j`` with ``j+1 < len(candidates) - lookahead`` (exclude last K draft tail).
    inter_target_prefix_accept_len: list[int] | None = None
    # Planner / pivot metadata (optional).
    pivot_criteria_score: list[float] | None = None
    pivot_top1_prob: list[float] | None = None
    pivot_residual_score: list[float] | None = None
    pivot_expanded: list[bool] | None = None
    pivot_branch_count: list[int] | None = None
    pivot_selected_branch_idx: list[int] | None = None
    pivot_selected_root_token_id: list[int] | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: torch.Tensor | None = None  # Is this a tensor?
    # hierarchical: intermediate round uses scheduler.postprocess_hv_intermediate_round
    is_hv_intermediate: bool = False
    profile_trace: VerifyProfileTrace | None = None
    postprocess_mode: Literal["speculate", "hv_intermediate", "hv_target"] = "speculate"
    winning_branch_idx_per_parent: list[int] | None = None
    # Pivot expanded-row debug support: absolute expanded row selected per parent.
    winning_branch_row_idx_per_parent: list[int] | None = None
    # Pivot batch expansion summary (works even when profile trace is disabled, e.g. cost mode).
    pivot_before_expansion_batch_size: int | None = None
    pivot_after_expansion_batch_size: int | None = None
    # pivot_precollapse: B after draft-score collapse; target verify batch (always B).
    pivot_after_collapse_batch_size: int | None = None
    pivot_target_verify_batch_size: int | None = None
    scratch_commit_bundle: "PivotTreeCommitBundle | None" = None


class SpeculatorBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass


class VerifierBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        pass
