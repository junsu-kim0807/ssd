from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from ssd.engine.block_manager import BlockManager
from ssd.engine.helpers.speculate_types import (
    SpeculateResult,
    VerifyProfileTrace,
    VerifyResult,
    VerifierBase,
)
from ssd.engine.pivot_types import BranchForkState, PivotBranchBundle
from ssd.engine.scheduler import Scheduler
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.utils.verify import verify


@dataclass
class _BranchVerifyOutcome:
    suffixes: list[list[int]]
    recovery: list[int]
    accept_len: list[int]


class PivotExecutorFlat(VerifierBase):
    """Target-authoritative flat pivot collapse over expanded root branches."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        scheduler: Scheduler,
        metrics: dict | None = None,
        enable_profile_trace: bool = False,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.enable_profile_trace = enable_profile_trace

    @staticmethod
    def _replace_parent_tail(
        bm: BlockManager,
        parent_block_table: list[int],
        shared_prefix_blocks: int,
        winner_private_tail: list[int],
    ) -> list[int]:
        """Drop parent's pre-allocated tail (it holds branch 0's KV) and graft
        the winner branch's private tail.

        In COW mode this private tail includes the partial COW block (if any)
        plus all divergent tail blocks. ``winner_private_tail`` blocks already
        have ``ref_count == 1`` from fork allocation, so ownership transfer
        simply requires not deallocating them.
        """
        shared_prefix_blocks = max(0, min(shared_prefix_blocks, len(parent_block_table)))
        # Release parent's existing tail (everything past the shared prefix).
        for block_id in parent_block_table[shared_prefix_blocks:]:
            block = bm.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                bm._deallocate_block(block_id)
        return list(parent_block_table[:shared_prefix_blocks]) + list(winner_private_tail)

    def _commit_winner_and_release_forks(
        self,
        parent_seqs: list[Sequence],
        bundle: PivotBranchBundle,
        winners_per_parent: list[int],
    ) -> None:
        """Transfer winner branch private tails into the parent's block tables,
        then release loser forks. ``branch 0`` is parent-in-place — its KV is
        already in the parent's block table, so we never release or transfer it.
        Here, private tail means partial COW block + divergent tail blocks.

        Winner-handling rules (per parent):
          * ``winner == 0``: no transfer; parent already owns the right KV.
          * ``winner != 0``: release parent's existing tail blocks, replace with
            the winner branch's private tail (ownership moves to parent), and
            release ONLY the winner fork's shared prefix refcount bump (the
            private tail must NOT be deallocated because parent now owns it).

        Loser branches (``branch_idx != winner``) get a full ``release_fork``.
        """
        if bundle.branch_states is None or bundle.expanded_seqs is None:
            return
        t_bm: BlockManager = self.scheduler.block_manager
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager

        # Pass 1: graft winner private tails onto the parent block tables.
        # We iterate this pass first so loser releases (which only touch their
        # own private tail) do not accidentally race with the transfer.
        for seq, st in zip(bundle.expanded_seqs, bundle.branch_states):
            winner_b_idx = winners_per_parent[st.parent_seq_idx]
            if st.branch_idx != winner_b_idx:
                continue
            if st.is_parent_inplace:
                # Branch 0 won — parent block tables already hold the correct KV.
                continue
            parent = parent_seqs[st.parent_seq_idx]
            parent.block_table = self._replace_parent_tail(
                t_bm,
                parent.block_table,
                st.target_shared_prefix_blocks,
                st.target_private_tail_block_ids,
            )
            parent.draft_block_table = self._replace_parent_tail(
                d_bm,
                parent.draft_block_table,
                st.draft_shared_prefix_blocks,
                st.draft_private_tail_block_ids,
            )
            if i_bm is not None:
                parent.inter_block_table = self._replace_parent_tail(
                    i_bm,
                    parent.inter_block_table,
                    st.inter_shared_prefix_blocks,
                    st.inter_private_tail_block_ids,
                )

        # Pass 2: release winner shared-prefix refs (private tails were
        # transferred in Pass 1) and fully release every loser fork. Skip
        # parent-in-place branches in both cases — they were never forked.
        for seq, st in zip(bundle.expanded_seqs, bundle.branch_states):
            if st.is_parent_inplace:
                continue
            winner_b_idx = winners_per_parent[st.parent_seq_idx]
            is_winner = st.branch_idx == winner_b_idx
            if is_winner:
                # Private tail belongs to the parent now — release shared prefix only.
                t_bm.release_fork(seq.block_table, [], st.target_shared_prefix_blocks)
                d_bm.release_fork(
                    seq.draft_block_table, [], st.draft_shared_prefix_blocks
                )
                if i_bm is not None:
                    i_bm.release_fork(
                        seq.inter_block_table, [], st.inter_shared_prefix_blocks
                    )
            else:
                t_bm.release_fork(
                    seq.block_table,
                    st.target_private_tail_block_ids,
                    st.target_shared_prefix_blocks,
                )
                d_bm.release_fork(
                    seq.draft_block_table,
                    st.draft_private_tail_block_ids,
                    st.draft_shared_prefix_blocks,
                )
                if i_bm is not None:
                    i_bm.release_fork(
                        seq.inter_block_table,
                        st.inter_private_tail_block_ids,
                        st.inter_shared_prefix_blocks,
                    )

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        token_ids = self.target_model_runner.call("run", seqs, True)
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
        return VerifyResult([], [seq.recovery_token_id for seq in seqs], None)

    def _verify_expanded(
        self,
        expanded_seqs: list[Sequence],
        speculate_result: SpeculateResult,
    ) -> _BranchVerifyOutcome:
        b_exp = len(expanded_seqs)
        result = self.target_model_runner.call("run", expanded_seqs, False, False, True)
        logits_p_flat = result
        for s in expanded_seqs:
            s.num_cached_tokens += self.lookahead + 1
        logits_p = logits_p_flat.view(b_exp, self.lookahead + 1, -1)
        # Match vanilla ``Verifier``: use ``draft_temperature`` for the draft
        # column when present, otherwise fall back to ``temperature``. This
        # preserves speculative-decoding semantics under sampling.
        temps_target = [s.temperature for s in expanded_seqs]
        temps_draft = [
            s.draft_temperature if s.draft_temperature is not None else s.temperature
            for s in expanded_seqs
        ]
        temperatures_target = torch.tensor(
            temps_target, dtype=torch.float32, device=self.device
        )
        temperatures_draft = torch.tensor(
            temps_draft, dtype=torch.float32, device=self.device
        )
        suffixes, recovery = verify(
            logits_p=logits_p,
            logits_q=speculate_result.logits_q,
            speculations=speculate_result.speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=None,
            sampler_x=None,
            async_fan_out=None,
            jit_speculate=False,
        )
        accept_len = [max(0, len(s) - 1) for s in suffixes]
        return _BranchVerifyOutcome(suffixes=suffixes, recovery=recovery, accept_len=accept_len)

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        bundle = speculate_result.branch_bundle
        if bundle is None or bundle.expanded_seqs is None:
            raise ValueError("PivotExecutorFlat requires speculate_result.branch_bundle with expanded_seqs")

        outcome = self._verify_expanded(bundle.expanded_seqs, speculate_result)
        parent_bsz = bundle.parent_batch_size

        winners = [-1] * parent_bsz
        winner_rows = [-1] * parent_bsz
        new_suffixes: list[list[int]] = [[] for _ in range(parent_bsz)]
        recovery_tokens = [0] * parent_bsz

        per_parent_rows: list[list[int]] = [[] for _ in range(parent_bsz)]
        for row_idx, pidx in enumerate(bundle.parent_index_per_branch):
            per_parent_rows[pidx].append(row_idx)

        for pidx in range(parent_bsz):
            rows = per_parent_rows[pidx]
            best_row = rows[0]
            best_accept = outcome.accept_len[best_row]
            for r in rows[1:]:
                # Debug-friendly deterministic rule for temp=0 parity investigations:
                # only switch away from branch 0 when acceptance length strictly improves.
                if outcome.accept_len[r] > best_accept:
                    best_row = r
                    best_accept = outcome.accept_len[r]
            winners[pidx] = bundle.branch_index_per_parent[best_row]
            winner_rows[pidx] = int(best_row)
            new_suffixes[pidx] = outcome.suffixes[best_row]
            recovery_tokens[pidx] = outcome.recovery[best_row]

        if bool(getattr(getattr(self.scheduler, "config", None), "debug_mode", False)):
            spec_rows = speculate_result.speculations.detach().cpu().tolist()
            expanded_accept_len = [int(x) for x in outcome.accept_len]
            collapse_row = {
                "pivot_round_debug": True,
                "stage": "collapse",
                "expanded_request_speculative_token_ids": [
                    [int(tok) for tok in row] for row in spec_rows
                ],
                "expanded_request_target_acceptance_length": expanded_accept_len,
                "selected_request_ids": [int(x) for x in winners],
                "selected_expanded_row_ids": [int(x) for x in winner_rows],
                "after_collapse_accepted_token_ids": [
                    [int(tok) for tok in suffix] for suffix in new_suffixes
                ],
            }
            print(json.dumps(collapse_row, ensure_ascii=False), flush=True)

        profile_trace = None
        if self.enable_profile_trace:
            per_parent_expanded = [c > 1 for c in bundle.branch_counts]
            per_parent_branch_count = list(bundle.branch_counts)
            per_parent_top1_prob = [0.0] * parent_bsz
            per_parent_selected_root = [0] * parent_bsz
            by_parent_rows: list[list[int]] = [[] for _ in range(parent_bsz)]
            for row_idx, pidx in enumerate(bundle.parent_index_per_branch):
                by_parent_rows[pidx].append(row_idx)
            for pidx in range(parent_bsz):
                rows = by_parent_rows[pidx]
                if rows:
                    if bundle.top1_probs is not None:
                        per_parent_top1_prob[pidx] = float(bundle.top1_probs[pidx])
                    else:
                        per_parent_top1_prob[pidx] = float(bundle.root_token_probs[rows[0]])
                    winner_row = rows[winners[pidx]]
                    per_parent_selected_root[pidx] = int(bundle.root_token_ids[winner_row])
            profile_trace = VerifyProfileTrace(
                verification_models=["target"] * parent_bsz,
                token_ids_per_position=[[] for _ in range(parent_bsz)],
                token_confidence_per_position=[[] for _ in range(parent_bsz)],
                recovery_tokens=list(recovery_tokens),
                bonus_tokens=[None for _ in range(parent_bsz)],
                accept_len=[max(0, len(s) - 1) for s in new_suffixes],
                pivot_criteria_score=(
                    list(bundle.criteria_scores)
                    if bundle.criteria_scores is not None
                    else [0.0 for _ in range(parent_bsz)]
                ),
                pivot_top1_prob=per_parent_top1_prob,
                pivot_residual_score=(
                    list(bundle.residual_scores)
                    if bundle.residual_scores is not None
                    else [0.0 for _ in range(parent_bsz)]
                ),
                pivot_expanded=per_parent_expanded,
                pivot_branch_count=per_parent_branch_count,
                pivot_selected_branch_idx=list(winners),
                pivot_selected_root_token_id=per_parent_selected_root,
            )
        self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend([len(s) for s in new_suffixes])
        # Commit winner branch KV onto parent block tables BEFORE releasing
        # losers; without this, the parent's draft/target KV would not match
        # the suffix that ``postprocess_speculate`` is about to commit, and the
        # next decode step would see ``num_*_cached_tokens`` ahead of the real
        # KV frontier.
        self._commit_winner_and_release_forks(seqs, bundle, winners)
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=None,
            is_hv_intermediate=False,
            profile_trace=profile_trace,
            postprocess_mode="speculate",
            winning_branch_idx_per_parent=winners,
            winning_branch_row_idx_per_parent=winner_rows,
            pivot_before_expansion_batch_size=int(parent_bsz),
            pivot_after_expansion_batch_size=int(len(bundle.parent_index_per_branch)),
        )
