"""Verifier for pivot_precollapse: vanilla B-row target verify + pivot metadata."""

from __future__ import annotations

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.spec_policy_traits import uses_pivot_precollapse
from ssd.engine.verifier import Verifier


class PivotPrecollapseVerifier(Verifier):
    def verify(
        self,
        seqs,
        speculate_result: SpeculateResult,
        eagle: bool = False,
    ) -> VerifyResult:
        result = super().verify(seqs, speculate_result, eagle=eagle)
        bundle = speculate_result.branch_bundle
        decision = getattr(bundle, "precollapse_decision", None) if bundle is not None else None
        if decision is None:
            return result
        bsz = len(seqs)
        result.winning_branch_idx_per_parent = list(decision.winning_branch_idx_per_parent)
        result.winning_branch_row_idx_per_parent = list(range(bsz))
        result.pivot_before_expansion_batch_size = decision.before_expansion_batch_size
        result.pivot_after_expansion_batch_size = decision.after_expansion_batch_size
        result.pivot_after_collapse_batch_size = bsz
        result.pivot_target_verify_batch_size = bsz
        trace = result.profile_trace
        if trace is not None:
            trace.pivot_criteria_score = (
                list(bundle.criteria_scores)
                if bundle.criteria_scores is not None
                else [0.0] * bsz
            )
            trace.pivot_top1_prob = (
                list(bundle.top1_probs)
                if bundle.top1_probs is not None
                else [0.0] * bsz
            )
            trace.pivot_residual_score = (
                list(bundle.residual_scores)
                if bundle.residual_scores is not None
                else [0.0] * bsz
            )
            trace.pivot_expanded = [c > 1 for c in decision.branch_count_per_parent]
            trace.pivot_branch_count = list(decision.branch_count_per_parent)
            trace.pivot_selected_branch_idx = list(decision.winning_branch_idx_per_parent)
            trace.pivot_selected_root_token_id = list(decision.winning_root_token_per_parent)
        return result


def attach_precollapse_rollback_asserts(scheduler, seqs: list, speculate_result: SpeculateResult) -> None:
    """After rollback: committed tape + cached depths (debug; pivot_precollapse only)."""
    cfg = getattr(scheduler, "config", None)
    if cfg is None or not bool(getattr(cfg, "debug_mode", False)):
        return
    pol = getattr(cfg, "spec_policy", "")
    if not uses_pivot_precollapse(pol):
        return
    bundle = getattr(speculate_result, "branch_bundle", None)
    dec = getattr(bundle, "precollapse_decision", None) if bundle is not None else None
    clens = getattr(dec, "committed_len_per_parent", None) if dec is not None else None
    if not clens:
        return
    for seq, clen in zip(seqs, clens):
        assert seq.num_tokens == clen, (seq.num_tokens, clen)
        assert seq.num_cached_tokens == clen, (seq.num_cached_tokens, clen)
        assert seq.num_draft_cached_tokens == clen, (seq.num_draft_cached_tokens, clen)
