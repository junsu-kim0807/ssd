from __future__ import annotations

import torch

from ssd.engine.block_manager import BlockManager
from ssd.engine.helpers.pivot_tree_helpers import (
    build_phase0_packed_inputs,
)
from ssd.engine.helpers.speculate_types import (
    SpeculateResult,
    VerifyProfileTrace,
    VerifyResult,
    VerifierBase,
)
from ssd.engine.pivot_executor_flat import _BranchVerifyOutcome
from ssd.engine.pivot_types import PivotTreeCommitBundle, PivotTreeScratchBundle
from ssd.engine.scheduler import Scheduler
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.utils.verify import verify


class PivotTreeScratchExecutor(VerifierBase):
    """Target-tree pivot executor for phase-1/phase-2 scratch commit paths."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        draft_model_runner: ModelRunner,
        scheduler: Scheduler,
        metrics: dict | None = None,
        enable_profile_trace: bool = False,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.draft_model_runner = draft_model_runner
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        self.enable_profile_trace = enable_profile_trace

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        token_ids = self.target_model_runner.call("run", seqs, True)
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
        return VerifyResult([], [seq.recovery_token_id for seq in seqs], None)

    def _verify_flat_safety_mode(
        self,
        expanded_seqs: list[Sequence],
        speculate_result: SpeculateResult,
    ) -> _BranchVerifyOutcome:
        """Phase-0 safe fallback: flat expanded verify semantics."""
        b_exp = len(expanded_seqs)
        logits_p_flat = self.target_model_runner.call("run", expanded_seqs, False, False, True)
        for s in expanded_seqs:
            s.num_cached_tokens += self.lookahead + 1
        logits_p = logits_p_flat.view(b_exp, self.lookahead + 1, -1)
        temps_target = [s.temperature for s in expanded_seqs]
        temps_draft = [
            s.draft_temperature if s.draft_temperature is not None else s.temperature
            for s in expanded_seqs
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)
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

    def _phase0_compare_logits_flat_vs_tree(
        self,
        expanded_seqs: list[Sequence],
        speculate_result: SpeculateResult,
    ) -> None:
        """Phase-0 checks: compare flat verify logits against packed tree logits."""
        cfg = getattr(self.scheduler, "config", None)
        if not bool(getattr(cfg, "debug_phase0_flat_compare", False)):
            return
        spec_rows = speculate_result.speculations.detach().cpu().tolist()
        packed = build_phase0_packed_inputs(
            expanded_seqs,
            spec_rows,
            block_size=self.scheduler.block_size,
            device=self.device,
            use_draft_table=False,
        )
        q_indptr = packed.cu_seqlens_q
        q_lens = (q_indptr[1:] - q_indptr[:-1]).to(torch.int64)
        kv_lens = packed.context_lens.to(torch.int64)
        expected_mask_len = int((q_lens * kv_lens).sum().item())
        actual_mask_len = (
            int(packed.tree_attn_mask.numel())
            if packed.tree_attn_mask is not None
            else -1
        )
        assert actual_mask_len == expected_mask_len, (
            "phase0 tree_attn_mask length mismatch: "
            f"expected={expected_mask_len}, actual={actual_mask_len}, "
            f"q_lens={q_lens.detach().cpu().tolist()}, "
            f"kv_lens={kv_lens.detach().cpu().tolist()}"
        )
        q_total = int(q_lens.sum().item())
        assert int(packed.input_ids.numel()) == q_total
        assert int(packed.positions.numel()) == q_total
        assert int(packed.slot_mapping.numel()) == q_total
        logits_flat = self.target_model_runner.call("run", expanded_seqs, False, False, True)
        logits_flat = logits_flat.view(len(expanded_seqs), self.lookahead + 1, -1)
        logits_tree = self.target_model_runner.call(
            "run_packed_tree_decode",
            packed.input_ids,
            packed.positions,
            packed.slot_mapping,
            packed.context_lens,
            packed.block_tables,
            packed.cu_seqlens_q,
            packed.max_seqlen_q,
            packed.tree_attn_mask,
        )
        logits_tree = logits_tree.view(len(expanded_seqs), self.lookahead + 1, -1)
        _ = torch.max(torch.abs(logits_tree - logits_flat)).item()

    @staticmethod
    def _replace_parent_tail(
        bm: BlockManager,
        parent_block_table: list[int],
        shared_prefix_blocks: int,
        winner_private_tail: list[int],
    ) -> list[int]:
        shared_prefix_blocks = max(0, min(shared_prefix_blocks, len(parent_block_table)))
        for block_id in parent_block_table[shared_prefix_blocks:]:
            block = bm.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                bm._deallocate_block(block_id)
        return list(parent_block_table[:shared_prefix_blocks]) + list(winner_private_tail)

    def _commit_winner_and_release_forks(
        self,
        parent_seqs: list[Sequence],
        bundle: PivotTreeScratchBundle,
        winners_per_parent: list[int],
    ) -> None:
        """Temporary safety mode: identical fork commit semantics as flat pivot.

        Until production target-tree scratch verify + slot-copy commit is fully
        implemented, keep target/draft/intermediate ownership transfer logic
        exactly aligned with ``PivotExecutorFlat`` so non-zero winner branches
        cannot corrupt parent KV frontiers.
        """
        if bundle.branch_states is None or bundle.expanded_seqs is None:
            return
        t_bm: BlockManager = self.scheduler.block_manager
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager

        # Pass 1: graft winner private tails onto parent tables.
        for seq, st in zip(bundle.expanded_seqs, bundle.branch_states):
            winner_b_idx = winners_per_parent[st.parent_seq_idx]
            if st.branch_idx != winner_b_idx:
                continue
            if st.is_parent_inplace:
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

        # Pass 2: release winner shared-prefix refs; fully release losers.
        for seq, st in zip(bundle.expanded_seqs, bundle.branch_states):
            if st.is_parent_inplace:
                continue
            winner_b_idx = winners_per_parent[st.parent_seq_idx]
            is_winner = st.branch_idx == winner_b_idx
            if is_winner:
                t_bm.release_fork(seq.block_table, [], st.target_shared_prefix_blocks)
                d_bm.release_fork(seq.draft_block_table, [], st.draft_shared_prefix_blocks)
                if i_bm is not None:
                    i_bm.release_fork(seq.inter_block_table, [], st.inter_shared_prefix_blocks)
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

    def _commit_phase1_draft_graft_and_release_forks(
        self,
        parent_seqs: list[Sequence],
        bundle: PivotTreeScratchBundle,
        winners_per_parent: list[int],
    ) -> None:
        """Phase-1: graft draft/inter winner tails only; target KV via scratch slot copy."""
        if bundle.branch_states is None or bundle.expanded_seqs is None:
            return
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager
        t_bm: BlockManager = self.scheduler.block_manager

        for seq, st in zip(bundle.expanded_seqs, bundle.branch_states):
            winner_b_idx = winners_per_parent[st.parent_seq_idx]
            if st.branch_idx != winner_b_idx:
                continue
            if st.is_parent_inplace:
                continue
            parent = parent_seqs[st.parent_seq_idx]
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

        for seq, st in zip(bundle.expanded_seqs, bundle.branch_states):
            if st.is_parent_inplace:
                continue
            winner_b_idx = winners_per_parent[st.parent_seq_idx]
            is_winner = st.branch_idx == winner_b_idx
            if is_winner:
                t_bm.release_fork(seq.block_table, [], st.target_shared_prefix_blocks)
                d_bm.release_fork(seq.draft_block_table, [], st.draft_shared_prefix_blocks)
                if i_bm is not None:
                    i_bm.release_fork(seq.inter_block_table, [], st.inter_shared_prefix_blocks)
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

    @staticmethod
    def _build_scratch_commit_bundle(
        bundle: PivotTreeScratchBundle,
        winner_rows: list[int],
        new_suffixes: list[list[int]],
    ) -> PivotTreeCommitBundle:
        won_tgt: list[list[int]] = []
        won_draft: list[list[int]] = []
        for pidx in range(bundle.parent_batch_size):
            row = int(winner_rows[pidx])
            L = len(new_suffixes[pidx])
            assert 0 <= row < len(bundle.path_node_ids), (
                f"winner row out of range: row={row}, n_rows={len(bundle.path_node_ids)}"
            )
            nodes = [int(x) for x in bundle.path_node_ids[row][:L]]
            assert len(nodes) == L, (
                f"winner_target_node_ids length mismatch: len(nodes)={len(nodes)}, expected={L}"
            )
            for nid in nodes:
                assert nid in bundle.target_node_to_slot, (
                    f"target node id {nid} missing from target_node_to_slot"
                )
            won_tgt.append(nodes)
            if bundle.draft_node_to_slot:
                for nid in nodes:
                    assert nid in bundle.draft_node_to_slot, (
                        f"draft node id {nid} missing from draft_node_to_slot"
                    )
                won_draft.append(list(nodes))
            else:
                won_draft.append([])
        return PivotTreeCommitBundle(
            winner_target_node_ids=won_tgt,
            winner_draft_node_ids=won_draft,
            target_node_slot=bundle.target_node_to_slot,
            draft_node_slot=bundle.draft_node_to_slot,
            raw_suffix_lens=[len(s) for s in new_suffixes],
            scratch_owner=bundle.scratch_owner,
        )

    def _phase1a_debug_logits_compare_flat_vs_scratch(
        self,
        bundle: PivotTreeScratchBundle,
        logits_p_scratch: torch.Tensor,
    ) -> None:
        return

    def _verify_with_target_scratch(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
        bundle: PivotTreeScratchBundle,
    ) -> VerifyResult:
        assert bundle.draft_node_to_slot or bundle.expanded_seqs is not None
        assert bundle.target_scratch_packed is not None
        if not bool(getattr(self.target_model_runner, "enforce_eager", False)):
            raise RuntimeError(
                "Phase-1 pivot_tree_scratch target scratch verify requires "
                "target ModelRunner.enforce_eager=True until CUDA-graph support lands."
            )
        for st in bundle.branch_states or []:
            assert not st.target_private_tail_block_ids, (
                "Phase-1 invariant: no target private tail on expanded rows "
                "(speculator must use fork_target_kv=False / shared-prefix target tables)."
            )
        packed = bundle.target_scratch_packed
        q_indptr = packed.cu_seqlens_q
        q_lens = (q_indptr[1:] - q_indptr[:-1]).to(torch.int64)
        kv_lens = packed.context_lens.to(torch.int64)
        expected_mask_len = int((q_lens * kv_lens).sum().item())
        actual_mask_len = (
            int(packed.tree_attn_mask.numel())
            if packed.tree_attn_mask is not None
            else -1
        )
        assert actual_mask_len == expected_mask_len, (
            f"tree_attn_mask length mismatch: expected {expected_mask_len}, got {actual_mask_len}"
        )
        q_total = int(q_lens.sum().item())
        assert int(packed.input_ids.numel()) == q_total
        assert int(packed.positions.numel()) == q_total
        assert int(packed.slot_mapping.numel()) == q_total
        logits_tree_flat = self.target_model_runner.call(
            "run_packed_tree_decode",
            packed.input_ids,
            packed.positions,
            packed.slot_mapping,
            packed.context_lens,
            packed.block_tables,
            packed.cu_seqlens_q,
            packed.max_seqlen_q,
            packed.tree_attn_mask,
        )
        expected_q = int(packed.input_ids.numel())
        n_rows = len(bundle.path_node_ids)
        row_len = len(bundle.path_node_ids[0]) if n_rows else 0
        assert n_rows * row_len == expected_q, (
            f"Phase1A row geometry mismatch: n_rows={n_rows}, row_len={row_len}, "
            f"expected_q={expected_q}"
        )
        # run_packed_tree_decode can return either flattened [N,V] or row-shaped [B_exp,K+1,V].
        if logits_tree_flat.ndim == 2:
            assert int(logits_tree_flat.shape[0]) == expected_q, (
                f"logits rows mismatch: logits={int(logits_tree_flat.shape[0])}, expected_q={expected_q}"
            )
            logits_p = logits_tree_flat.view(n_rows, row_len, -1)
        elif logits_tree_flat.ndim == 3:
            assert int(logits_tree_flat.shape[0]) == n_rows, (
                f"logits batch mismatch: logits={int(logits_tree_flat.shape[0])}, n_rows={n_rows}"
            )
            assert int(logits_tree_flat.shape[1]) == row_len, (
                f"logits row_len mismatch: logits={int(logits_tree_flat.shape[1])}, row_len={row_len}"
            )
            logits_p = logits_tree_flat
        else:
            raise AssertionError(
                f"unexpected logits shape from run_packed_tree_decode: {list(logits_tree_flat.shape)}"
            )
        assert logits_p.shape[:2] == speculate_result.speculations.shape, (
            f"logits_p {logits_p.shape[:2]} vs speculations {speculate_result.speculations.shape[:2]}"
        )
        self._phase1a_debug_logits_compare_flat_vs_scratch(bundle, logits_p)
        temps_target = [seqs[int(pidx)].temperature for pidx in bundle.path_parent_seq_idx]
        temps_draft = [
            (
                seqs[int(pidx)].draft_temperature
                if seqs[int(pidx)].draft_temperature is not None
                else seqs[int(pidx)].temperature
            )
            for pidx in bundle.path_parent_seq_idx
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)
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
        parent_bsz = bundle.parent_batch_size
        winners = [-1] * parent_bsz
        winner_rows = [-1] * parent_bsz
        new_suffixes: list[list[int]] = [[] for _ in range(parent_bsz)]
        recovery_tokens = [0] * parent_bsz
        per_parent_rows: list[list[int]] = [[] for _ in range(parent_bsz)]
        for row_idx, pidx in enumerate(bundle.path_parent_seq_idx):
            per_parent_rows[pidx].append(row_idx)
        force_b = getattr(getattr(self.scheduler, "config", None), "debug_force_pivot_winner_branch", None)
        for pidx in range(parent_bsz):
            rows = per_parent_rows[pidx]
            best_row = rows[0]
            best_accept = accept_len[best_row]
            if force_b is not None:
                forced = int(force_b)
                for r in rows:
                    if bundle.path_branch_idx[r] == forced:
                        best_row = r
                        best_accept = accept_len[r]
                        break
            else:
                for r in rows[1:]:
                    if accept_len[r] > best_accept:
                        best_row = r
                        best_accept = accept_len[r]
            winners[pidx] = bundle.path_branch_idx[best_row]
            winner_rows[pidx] = int(best_row)
            new_suffixes[pidx] = suffixes[best_row]
            recovery_tokens[pidx] = recovery[best_row]

        if not bundle.draft_node_to_slot:
            self._commit_phase1_draft_graft_and_release_forks(seqs, bundle, winners)
        commit_bundle = self._build_scratch_commit_bundle(bundle, winner_rows, new_suffixes)


        profile_trace = None
        if self.enable_profile_trace:
            per_parent_expanded = [c > 1 for c in bundle.host_plan.branch_counts]
            per_parent_branch_count = list(bundle.host_plan.branch_counts)
            profile_trace = VerifyProfileTrace(
                verification_models=["target"] * parent_bsz,
                token_ids_per_position=[[] for _ in range(parent_bsz)],
                token_confidence_per_position=[[] for _ in range(parent_bsz)],
                accept_len=[max(0, len(s) - 1) for s in new_suffixes],
                recovery_tokens=list(recovery_tokens),
                bonus_tokens=[None for _ in range(parent_bsz)],
                pivot_expanded=per_parent_expanded,
                pivot_branch_count=per_parent_branch_count,
                pivot_selected_branch_idx=list(winners),
            )
        self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend(
            [len(s) for s in new_suffixes]
        )
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
            pivot_after_expansion_batch_size=int(bundle.expanded_batch_size),
            scratch_commit_bundle=commit_bundle,
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        bundle = speculate_result.branch_bundle
        if not isinstance(bundle, PivotTreeScratchBundle):
            raise ValueError("PivotTreeScratchExecutor requires PivotTreeScratchBundle")

        has_target_scratch = bool(bundle.target_node_to_slot)
        _has_draft_scratch = bool(bundle.draft_node_to_slot)
        if has_target_scratch:
            assert bundle.target_scratch_packed is not None
            return self._verify_with_target_scratch(seqs, speculate_result, bundle)
        # Explicit Phase-0 safety-mode invariants.
        assert bundle.expanded_seqs is not None
        assert bundle.branch_states is not None, "Phase-0 fallback requires branch_states"
        assert bundle.scratch_owner is None

        self._phase0_compare_logits_flat_vs_tree(bundle.expanded_seqs, speculate_result)
        outcome = self._verify_flat_safety_mode(bundle.expanded_seqs, speculate_result)
        parent_bsz = bundle.parent_batch_size
        winners = [-1] * parent_bsz
        winner_rows = [-1] * parent_bsz
        new_suffixes: list[list[int]] = [[] for _ in range(parent_bsz)]
        recovery_tokens = [0] * parent_bsz

        per_parent_rows: list[list[int]] = [[] for _ in range(parent_bsz)]
        for row_idx, pidx in enumerate(bundle.path_parent_seq_idx):
            per_parent_rows[pidx].append(row_idx)

        for pidx in range(parent_bsz):
            rows = per_parent_rows[pidx]
            best_row = rows[0]
            best_accept = outcome.accept_len[best_row]
            for r in rows[1:]:
                if outcome.accept_len[r] > best_accept:
                    best_row = r
                    best_accept = outcome.accept_len[r]
            winners[pidx] = bundle.path_branch_idx[best_row]
            winner_rows[pidx] = int(best_row)
            new_suffixes[pidx] = outcome.suffixes[best_row]
            recovery_tokens[pidx] = outcome.recovery[best_row]

        self._commit_winner_and_release_forks(seqs, bundle, winners)

        profile_trace = None
        if self.enable_profile_trace:
            per_parent_expanded = [c > 1 for c in bundle.host_plan.branch_counts]
            per_parent_branch_count = list(bundle.host_plan.branch_counts)
            profile_trace = VerifyProfileTrace(
                verification_models=["target"] * parent_bsz,
                token_ids_per_position=[[] for _ in range(parent_bsz)],
                token_confidence_per_position=[[] for _ in range(parent_bsz)],
                accept_len=[max(0, len(s) - 1) for s in new_suffixes],
                recovery_tokens=list(recovery_tokens),
                bonus_tokens=[None for _ in range(parent_bsz)],
                pivot_expanded=per_parent_expanded,
                pivot_branch_count=per_parent_branch_count,
                pivot_selected_branch_idx=list(winners),
            )
        self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend(
            [len(s) for s in new_suffixes]
        )
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
            pivot_after_expansion_batch_size=int(bundle.expanded_batch_size),
            scratch_commit_bundle=None,
        )
