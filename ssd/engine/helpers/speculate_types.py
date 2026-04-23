from dataclasses import dataclass
import torch
from ssd.engine.sequence import Sequence
from abc import ABC, abstractmethod


@dataclass
class SpeculateResult:
    speculations: torch.Tensor
    logits_q: torch.Tensor
    cache_hits: torch.Tensor | None = None


@dataclass
class VerifyProfileTrace:
    """Batch-shaped verification metadata for profiling (one list entry per request)."""

    verification_models: list[str]
    token_ids_per_position: list[list[int]]
    token_confidence_per_position: list[list[float]]
    accept_len: list[int]
    recovery_tokens: list[int]
    bonus_tokens: list[int | None]
    # Optional hierarchical / pivot intermediate columns (per batch row; None if N/A)
    inter_token_ids_per_position: list[list[int] | None] | None = None
    inter_token_confidence_per_position: list[list[float] | None] | None = None
    inter_accept_len: list[int | None] | None = None
    inter_recovery_token: list[int | None] | None = None
    inter_bonus_token: list[int | None] | None = None
    # Target HV only: consecutive greedy matches along ``candidates`` restricted to
    # indices ``j`` with ``j+1 < len(candidates) - lookahead`` (exclude last K draft tail).
    inter_target_prefix_accept_len: list[int] | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: torch.Tensor | None = None  # Is this a tensor?
    # hierarchical: intermediate round uses scheduler.postprocess_hv_intermediate_round
    is_hv_intermediate: bool = False
    profile_trace: VerifyProfileTrace | None = None


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
