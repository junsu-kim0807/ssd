from __future__ import annotations

import torch

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, VerifierBase
from ssd.engine.sequence import Sequence
from ssd.engine.verifier_hierarchical import VerifierHierarchical


class PivotExecutorHierarchical(VerifierBase):
    """Bridge class for pivot_hierarchical fused wiring.

    v1 keeps hierarchical verification semantics intact and reuses
    VerifierHierarchical primitives while step-level orchestration owns
    branch lifecycle and eventual collapse.
    """

    def __init__(self, inner: VerifierHierarchical):
        super().__init__(inner.lookahead, inner.device)
        self.inner = inner
        self.r = inner.r

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        return self.inner.prefill(seqs, eagle=eagle)

    def verify_intermediate_round(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
        *,
        emit_step_metrics: bool = True,
    ) -> VerifyResult:
        return self.inner.verify_intermediate_round(
            seqs, speculate_result, emit_step_metrics=emit_step_metrics
        )

    def verify_target_round(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
        *,
        emit_step_metrics: bool = True,
    ) -> VerifyResult:
        return self.inner.verify_target_round(
            seqs, speculate_result, emit_step_metrics=emit_step_metrics
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        return self.inner.verify(seqs, speculate_result, eagle=eagle)
