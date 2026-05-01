"""Sync pivot speculator with draft-score collapse before target verify (B rows)."""

from __future__ import annotations

import os
from collections import defaultdict
from time import perf_counter

import numpy as np
import torch

from ssd.engine.block_manager import BlockManager
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_branch_planner import (
    PivotExpansionPlan,
    PivotHostPlan,
    build_pivot_expansion_plan,
)
from ssd.engine.pivot_executor_flat import PivotExecutorFlat
from ssd.engine.pivot_speculator_sync import PivotRootSpeculatorSync
from ssd.engine.pivot_types import (
    BranchForkState,
    PivotBranchBundle,
    PivotPrecollapseDecision,
)
from ssd.engine.sequence import Sequence


class PivotPrecollapseSpeculatorSync(PivotRootSpeculatorSync):
    """Pivot root expansion with draft-only alt branches; collapse before target verify."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        draft_model_runner,
        target_model_runner,
        intermediate_runner,
        scheduler,
        expansion_cfg,
        max_expand_rows: int | None = None,
        enable_profile_trace: bool = False,
        *,
        score_method: str = "logprob_sum",
        score_temperature_aware: bool = False,
    ):
        if score_method not in {"logprob_sum", "logit_sum"}:
            raise ValueError(
                "pivot_precollapse_score_method must be one of {'logprob_sum', 'logit_sum'}; "
                f"got {score_method!r}"
            )
        if score_temperature_aware:
            raise NotImplementedError(
                "pivot_precollapse_score_temperature_aware=True is not wired yet"
            )
        super().__init__(
            lookahead,
            device,
            draft_model_runner,
            target_model_runner,
            intermediate_runner,
            scheduler,
            expansion_cfg,
            max_expand_rows=max_expand_rows,
            enable_profile_trace=enable_profile_trace,
        )
        self.score_method = score_method
        self.score_temperature_aware = score_temperature_aware

    @staticmethod
    def _replace_parent_spec_tail(parent: Sequence, committed_len: int, suffix: list[int]) -> None:
        del parent.token_ids[committed_len:]
        parent.token_ids.extend(suffix)
        parent.num_tokens = committed_len + len(suffix)
        parent.last_token = parent.token_ids[-1]

    def _apply_precollapse_capacity_limit(
        self,
        seqs: list[Sequence],
        plan: PivotExpansionPlan,
    ) -> PivotExpansionPlan:
        """Draft-only expansion clamp (no target free-block accounting)."""
        bsz = len(seqs)
        if bsz == 0:
            return plan
        host = plan.host
        assert host is not None, "PivotExpansionPlan.host required for capacity clamp"
        branch_counts_src = list(host.branch_counts)

        expand_mask_host = list(host.expand_mask)
        for i, seq in enumerate(seqs):
            if seq.num_tokens + self.lookahead > self.scheduler.max_model_len:
                expand_mask_host[i] = False

        d_bm: BlockManager = self.scheduler.draft_block_manager
        free_d = len(d_bm.free_block_ids)

        scores_host = host.criteria_scores
        assert scores_host is not None
        order = sorted(range(bsz), key=lambda i: (scores_host[i], i))
        kept_host = [False] * bsz

        for idx in order:
            if not expand_mask_host[idx]:
                continue
            seq = seqs[idx]
            required_tokens = seq.num_tokens + self.lookahead
            per_extra_d = self._private_blocks_needed(
                seq.num_draft_cached_tokens,
                required_tokens,
                seq.block_size,
            )
            extras = max(0, int(branch_counts_src[idx]) - 1)
            need_d = per_extra_d * extras
            if need_d <= free_d:
                kept_host[idx] = True
                free_d -= need_d

        clamped_host = self._apply_variable_row_cap_host(
            kept_host, scores_host, branch_counts_src
        )
        if clamped_host == host.expand_mask:
            return plan

        per_parent_roots: list[list[tuple[int, float]]] = [[] for _ in range(bsz)]
        rp_src = host.root_token_probs
        for row_idx, (pidx, bidx) in enumerate(
            zip(host.parent_index_per_branch, host.branch_index_per_parent)
        ):
            rid = host.root_token_ids[row_idx]
            rp = rp_src[row_idx] if rp_src is not None else 0.0
            while len(per_parent_roots[pidx]) <= bidx:
                per_parent_roots[pidx].append((rid, rp))
            per_parent_roots[pidx][bidx] = (rid, rp)

        pib: list[int] = []
        bip: list[int] = []
        rid_out: list[int] = []
        rp_out: list[float] = []
        bcounts: list[int] = []
        for pidx in range(bsz):
            count = len(per_parent_roots[pidx]) if clamped_host[pidx] else 1
            bcounts.append(count)
            for bidx in range(count):
                pib.append(pidx)
                bip.append(bidx)
                rid, rp = per_parent_roots[pidx][bidx]
                rid_out.append(rid)
                rp_out.append(rp)

        new_host = PivotHostPlan(
            parent_index_per_branch=pib,
            branch_index_per_parent=bip,
            root_token_ids=rid_out,
            branch_counts=bcounts,
            expand_mask=clamped_host,
            criteria_scores=list(scores_host),
            root_token_probs=rp_out if rp_src is not None else None,
            top1_probs=list(host.top1_probs) if host.top1_probs is not None else None,
            residual_scores=list(host.residual_scores) if host.residual_scores is not None else None,
            dynamic_expansion_slope_scores=(
                list(host.dynamic_expansion_slope_scores)
                if host.dynamic_expansion_slope_scores is not None
                else None
            ),
        )

        return PivotExpansionPlan(
            parent_batch_size=bsz,
            expanded_batch_size=len(pib),
            expand_mask=torch.tensor(clamped_host, dtype=torch.bool, device=self.device),
            parent_index_per_branch=torch.tensor(pib, dtype=torch.int64, device=self.device),
            branch_index_per_parent=torch.tensor(bip, dtype=torch.int64, device=self.device),
            root_token_ids=torch.tensor(rid_out, dtype=torch.int64, device=self.device),
            root_token_probs=torch.tensor(rp_out, dtype=torch.float32, device=self.device),
            criteria_scores=plan.criteria_scores,
            top1_probs=plan.top1_probs,
            residual_scores=plan.residual_scores,
            branch_counts=bcounts,
            branch_counts_tensor=torch.tensor(bcounts, dtype=torch.int64, device=self.device),
            dynamic_expansion_slope_scores=plan.dynamic_expansion_slope_scores,
            host=new_host,
        )

    def _commit_precollapse_draft_winner_and_release(
        self,
        parent_seqs: list[Sequence],
        expanded_seqs: list[Sequence | None],
        branch_states: list[BranchForkState | None],
        winners_per_parent: list[int],
    ) -> None:
        """Graft winner draft tail only; release loser draft forks (no target/inter)."""
        d_bm: BlockManager = self.scheduler.draft_block_manager

        for seq, st in zip(expanded_seqs, branch_states):
            if seq is None or st is None:
                continue
            if st.branch_idx != winners_per_parent[st.parent_seq_idx]:
                continue
            if st.is_parent_inplace:
                continue
            if not st.draft_kv_owned:
                continue
            parent = parent_seqs[st.parent_seq_idx]
            parent.draft_block_table = PivotExecutorFlat._replace_parent_tail(
                d_bm,
                parent.draft_block_table,
                st.draft_shared_prefix_blocks,
                st.draft_private_tail_block_ids,
            )

        parent_alt_count_d: dict[int, int] = defaultdict(int)
        parent_d_shared: dict[int, int] = {}
        loser_d_tails: list[int] = []

        for seq, st in zip(expanded_seqs, branch_states):
            if seq is None or st is None or st.is_parent_inplace:
                continue
            if not st.draft_kv_owned:
                continue
            pidx = st.parent_seq_idx
            parent_alt_count_d[pidx] += 1
            parent_d_shared[pidx] = st.draft_shared_prefix_blocks
            if st.branch_idx != winners_per_parent[pidx]:
                loser_d_tails.extend(st.draft_private_tail_block_ids)

        for pidx, count in parent_alt_count_d.items():
            d_bm.release_shared_prefix_n(
                parent_seqs[pidx].draft_block_table, parent_d_shared[pidx], count
            )
        if loser_d_tails:
            d_bm._deallocate_n_blocks(loser_d_tails)

    def _branch_scores(
        self,
        speculations: torch.Tensor,
        logits_q: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expanded-row scalar scores [B_exp]."""
        k = self.lookahead
        cand = speculations[:, 1 : k + 1].long()
        if self.score_method == "logit_sum":
            sel = logits_q.gather(2, cand.unsqueeze(-1)).squeeze(-1)
            return sel.sum(dim=1).to(torch.float32)

        # logprob_sum (default)
        v = logits_q.float()
        lse = torch.logsumexp(v, dim=-1)
        sel = v.gather(2, cand.unsqueeze(-1)).squeeze(-1)
        return (sel - lse).sum(dim=1)

    def speculate(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
        fork_target_kv: bool = True,
    ) -> SpeculateResult:
        del fork_target_kv  # precollapse ignores target fork; signature kept for API parity
        assert not recovery_already_appended, (
            "pivot_precollapse does not support recovery_already_appended yet"
        )
        assert self.lookahead >= 1, "pivot_precollapse requires speculate_k >= 1"
        batch_size = len(seqs)
        if batch_size == 0:
            return SpeculateResult(
                speculations=torch.zeros(0, self.lookahead + 1, dtype=torch.int64, device=self.device),
                logits_q=torch.zeros(0, self.lookahead, 0, dtype=torch.float32, device=self.device),
            )

        committed_len_per_parent = [seq.num_tokens for seq in seqs]

        recovery_tokens: list[int] = []
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            recovery_tokens.append(int(seq.recovery_token_id))
            if recovery_already_appended:
                continue
            prov = seq.hv_provisional_token_ids
            skip_append = len(prov) > 0 and prov[-1] == seq.recovery_token_id
            if not skip_append:
                seq.append_token(seq.recovery_token_id)

        cond_parent_after_first: torch.Tensor | None = None
        if self.use_eagle:
            for seq in seqs:
                assert seq.last_target_hidden_state is not None, (
                    "Eagle pivot_precollapse requires seq.last_target_hidden_state "
                    "(from target prefill / post-verify)."
                )
            cond0 = torch.stack(
                [seq.last_target_hidden_state for seq in seqs], dim=0
            ).to(self.device)
            first_token_ids, first_logits_q, cond_parent_after_first = self.draft_model_runner.call(
                "run", seqs, False, True, True, cond0
            )
        else:
            first_token_ids, first_logits_q = self.draft_model_runner.call(
                "run", seqs, False, True, True
            )
        for s in seqs:
            s.num_draft_cached_tokens += 1

        if self._cached_root_tokens_per_parent is None:
            t_plan0 = perf_counter()
            plan = build_pivot_expansion_plan(
                first_logits_q,
                self.expansion_cfg,
                max_expand_rows=None,
                materialize_host=True,
                profile_metadata=self.enable_profile_trace,
            )
            t_plan1 = perf_counter()
            t_cap0 = perf_counter()
            plan = self._apply_precollapse_capacity_limit(seqs, plan)
            t_cap1 = perf_counter()
            self._override_branch0_roots(plan, first_token_ids, first_logits_q)
            t_cap2 = perf_counter()
            assert plan.host is not None
            roots_per_parent: list[list[int]] = [[] for _ in range(batch_size)]
            for pidx, bidx, tok in zip(
                plan.host.parent_index_per_branch,
                plan.host.branch_index_per_parent,
                plan.host.root_token_ids,
            ):
                while len(roots_per_parent[pidx]) <= bidx:
                    roots_per_parent[pidx].append(tok)
                roots_per_parent[pidx][bidx] = tok
            self._cached_root_tokens_per_parent = roots_per_parent
            self._cached_criteria_scores_per_parent = list(plan.host.criteria_scores or [])
            self._cached_slope_scores_per_parent = (
                list(plan.host.dynamic_expansion_slope_scores)
                if plan.host.dynamic_expansion_slope_scores is not None
                else None
            )
            if self.enable_profile_trace and plan.host.top1_probs is not None:
                self._cached_top1_probs_per_parent = list(plan.host.top1_probs)
            else:
                self._cached_top1_probs_per_parent = None
            if self.enable_profile_trace and plan.host.residual_scores is not None:
                self._cached_residual_scores_per_parent = list(plan.host.residual_scores)
            else:
                self._cached_residual_scores_per_parent = None
        else:
            t_plan0 = perf_counter()
            parent_index_per_branch: list[int] = []
            branch_index_per_parent: list[int] = []
            root_token_ids: list[int] = []
            branch_counts: list[int] = []
            for pidx, roots in enumerate(self._cached_root_tokens_per_parent):
                branch_counts.append(len(roots))
                for bidx, tok in enumerate(roots):
                    parent_index_per_branch.append(pidx)
                    branch_index_per_parent.append(bidx)
                    root_token_ids.append(tok)

            b_exp_cached = len(parent_index_per_branch)
            if self.enable_profile_trace and b_exp_cached > 0:
                probs = torch.softmax(first_logits_q.float(), dim=-1)
                pids_t = torch.tensor(parent_index_per_branch, dtype=torch.long, device=self.device)
                rids_t = torch.tensor(root_token_ids, dtype=torch.long, device=self.device)
                root_token_probs_t = probs[pids_t, rids_t].to(torch.float32)
            else:
                root_token_probs_t = torch.zeros(b_exp_cached, dtype=torch.float32, device=self.device)

            expand_mask_host = [len(r) > 1 for r in self._cached_root_tokens_per_parent]
            crit_cached = self._cached_criteria_scores_per_parent or [0.0] * batch_size
            slope_cached = self._cached_slope_scores_per_parent
            tp_cached = self._cached_top1_probs_per_parent
            rs_cached = self._cached_residual_scores_per_parent
            host = PivotHostPlan(
                parent_index_per_branch=list(parent_index_per_branch),
                branch_index_per_parent=list(branch_index_per_parent),
                root_token_ids=list(root_token_ids),
                branch_counts=list(branch_counts),
                expand_mask=expand_mask_host,
                criteria_scores=list(crit_cached),
                root_token_probs=(
                    root_token_probs_t.tolist() if self.enable_profile_trace else None
                ),
                top1_probs=(
                    list(tp_cached)
                    if self.enable_profile_trace and tp_cached is not None
                    else ([0.0] * batch_size if self.enable_profile_trace else None)
                ),
                residual_scores=(
                    list(rs_cached)
                    if self.enable_profile_trace and rs_cached is not None
                    else ([0.0] * batch_size if self.enable_profile_trace else None)
                ),
                dynamic_expansion_slope_scores=(
                    list(slope_cached) if slope_cached is not None else None
                ),
            )
            slope_t = (
                torch.tensor(slope_cached, dtype=torch.float32, device=self.device)
                if slope_cached is not None
                else None
            )
            top1_t = (
                torch.tensor(tp_cached, dtype=torch.float32, device=self.device)
                if self.enable_profile_trace and tp_cached is not None
                else torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            )
            res_t = (
                torch.tensor(rs_cached, dtype=torch.float32, device=self.device)
                if self.enable_profile_trace and rs_cached is not None
                else torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            )
            plan = PivotExpansionPlan(
                parent_batch_size=batch_size,
                expanded_batch_size=b_exp_cached,
                expand_mask=torch.tensor(
                    expand_mask_host,
                    dtype=torch.bool,
                    device=self.device,
                ),
                parent_index_per_branch=torch.tensor(
                    parent_index_per_branch, dtype=torch.int64, device=self.device
                ),
                branch_index_per_parent=torch.tensor(
                    branch_index_per_parent, dtype=torch.int64, device=self.device
                ),
                root_token_ids=torch.tensor(root_token_ids, dtype=torch.int64, device=self.device),
                root_token_probs=root_token_probs_t,
                criteria_scores=torch.tensor(crit_cached, dtype=torch.float32, device=self.device),
                top1_probs=top1_t,
                residual_scores=res_t,
                branch_counts=branch_counts,
                branch_counts_tensor=torch.tensor(branch_counts, dtype=torch.int64, device=self.device),
                dynamic_expansion_slope_scores=slope_t,
                host=host,
            )
            t_plan1 = perf_counter()
            t_cap0 = perf_counter()
            plan = self._apply_precollapse_capacity_limit(seqs, plan)
            t_cap1 = perf_counter()
            self._override_branch0_roots(plan, first_token_ids, first_logits_q)
            t_cap2 = perf_counter()

        self._debug_pivot_expansion(first_logits_q=first_logits_q, plan=plan)

        assert plan.host is not None
        host = plan.host
        parent_idx_list = host.parent_index_per_branch
        branch_idx_list = host.branch_index_per_parent
        root_token_list = host.root_token_ids
        root_prob_list = (
            host.root_token_probs
            if host.root_token_probs is not None
            else [0.0] * len(parent_idx_list)
        )
        b_exp = len(parent_idx_list)
        expanded_seqs: list[Sequence | None] = [None] * b_exp
        branch_states: list[BranchForkState | None] = [None] * b_exp

        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager
        draft_copy_src_all: list[int] = []
        draft_copy_dst_all: list[int] = []
        draft_copy_valid_all: list[int] = []
        num_nonzero_branches = 0
        num_target_cow_copy_blocks = 0
        num_draft_cow_copy_blocks = 0
        num_inter_cow_copy_blocks = 0

        t_branch_construct0 = perf_counter()
        for row_idx, (parent_idx, branch_idx, root_token) in enumerate(
            zip(parent_idx_list, branch_idx_list, root_token_list)
        ):
            if branch_idx == 0:
                continue
            num_nonzero_branches += 1
            parent = seqs[parent_idx]
            draft_cached0 = parent.num_draft_cached_tokens
            branch_seq = parent.clone_spec_for_branch()
            required_tokens = parent.num_tokens + self.lookahead
            branch_seq.block_table = list(parent.block_table)
            d_plan = d_bm.make_cow_fork_block_table(
                parent.draft_block_table,
                cached_tokens=draft_cached0,
                required_total_tokens=required_tokens,
            )
            branch_seq.draft_block_table = d_plan.fork_block_table
            if i_bm is not None:
                branch_seq.inter_block_table = list(parent.inter_block_table)
            else:
                branch_seq.inter_block_table = []

            if d_plan.copy_src_block_ids:
                draft_copy_src_all.extend(d_plan.copy_src_block_ids)
                draft_copy_dst_all.extend(d_plan.copy_dst_block_ids)
                draft_copy_valid_all.extend(d_plan.copy_valid_tokens)
                num_draft_cow_copy_blocks += len(d_plan.copy_src_block_ids)
            branch_seq.append_token(int(root_token))
            expanded_seqs[row_idx] = branch_seq
            branch_states[row_idx] = BranchForkState(
                parent_seq_idx=int(parent_idx),
                branch_idx=int(branch_idx),
                root_token_id=int(root_token),
                root_confidence=root_prob_list[row_idx],
                target_shared_prefix_blocks=0,
                draft_shared_prefix_blocks=d_plan.shared_prefix_blocks,
                inter_shared_prefix_blocks=0,
                draft_private_tail_block_ids=d_plan.private_tail_block_ids,
                target_private_tail_block_ids=[],
                inter_private_tail_block_ids=[],
                is_parent_inplace=False,
                target_kv_owned=False,
                draft_kv_owned=True,
                inter_kv_owned=False,
            )
            assert branch_states[row_idx].draft_shared_prefix_blocks == draft_cached0 // parent.block_size

        t_branch_construct1 = perf_counter()

        self._cuda_sync_for_pivot_cost()
        t_target_copy0 = perf_counter()
        t_target_copy1 = perf_counter()

        t_draft_copy0 = perf_counter()
        if draft_copy_src_all:
            self._copy_partial_cow_block(
                copy_src_block_ids=draft_copy_src_all,
                copy_dst_block_ids=draft_copy_dst_all,
                copy_valid_tokens=draft_copy_valid_all,
                kv_cache=self.draft_model_runner.kv_cache,
            )
        self._cuda_sync_for_pivot_cost()
        t_draft_copy1 = perf_counter()

        t_inter_copy0 = perf_counter()
        t_inter_copy1 = perf_counter()

        t_branch0_setup0 = perf_counter()
        for row_idx, (parent_idx, branch_idx, root_token) in enumerate(
            zip(parent_idx_list, branch_idx_list, root_token_list)
        ):
            if branch_idx != 0:
                continue
            parent = seqs[parent_idx]
            parent.append_token(int(root_token))
            expanded_seqs[row_idx] = parent
            branch_states[row_idx] = BranchForkState(
                parent_seq_idx=int(parent_idx),
                branch_idx=0,
                root_token_id=int(root_token),
                root_confidence=root_prob_list[row_idx],
                target_shared_prefix_blocks=len(parent.block_table),
                draft_shared_prefix_blocks=len(parent.draft_block_table),
                inter_shared_prefix_blocks=(
                    len(parent.inter_block_table) if i_bm is not None else 0
                ),
                draft_private_tail_block_ids=[],
                target_private_tail_block_ids=[],
                inter_private_tail_block_ids=[],
                is_parent_inplace=True,
            )
        t_branch0_setup1 = perf_counter()

        t_initial_pack0 = perf_counter()
        spec_np = np.empty((b_exp, self.lookahead + 1), dtype=np.int64)
        if b_exp > 0:
            spec_np[:, 0] = [recovery_tokens[parent_idx_list[i]] for i in range(b_exp)]
            spec_np[:, 1] = root_token_list

        logits_q_rows = [first_logits_q[plan.parent_index_per_branch]]
        t_initial_pack1 = perf_counter()

        cond_exp: torch.Tensor | None = None
        if self.use_eagle and b_exp > 0:
            assert cond_parent_after_first is not None
            pib = plan.parent_index_per_branch
            if not isinstance(pib, torch.Tensor):
                pib = torch.as_tensor(
                    pib, dtype=torch.long, device=cond_parent_after_first.device
                )
            else:
                pib = pib.to(device=cond_parent_after_first.device, dtype=torch.long)
            cond_exp = cond_parent_after_first[pib]
        conditioning: torch.Tensor | None = cond_exp if self.use_eagle else None

        t_tail_draft0 = perf_counter()
        if self.lookahead >= 2:
            for _k in range(1, self.lookahead):
                if self.use_eagle:
                    token_ids_k, logits_k, conditioning = self.draft_model_runner.call(
                        "run", expanded_seqs, False, True, True, conditioning
                    )
                else:
                    token_ids_k, logits_k = self.draft_model_runner.call(
                        "run", expanded_seqs, False, True, True
                    )
                for s in expanded_seqs:
                    if s is not None:
                        s.num_draft_cached_tokens += 1
                logits_q_rows.append(logits_k)
                for row_idx, (seq, tok) in enumerate(zip(expanded_seqs, token_ids_k)):
                    if seq is None:
                        continue
                    seq.append_token(tok)
                    spec_np[row_idx, _k + 1] = tok
        t_tail_draft1 = perf_counter()

        t_extra_draft0 = perf_counter()
        if b_exp > 0:
            if self.use_eagle:
                self.draft_model_runner.call(
                    "run", expanded_seqs, False, True, True, conditioning
                )
            else:
                self.draft_model_runner.call("run", expanded_seqs, False, True, True)
            for s in expanded_seqs:
                if s is not None:
                    s.num_draft_cached_tokens += 1
        t_extra_draft1 = perf_counter()

        t_final_pack0 = perf_counter()
        logits_q = torch.stack(logits_q_rows, dim=1)
        speculations = torch.from_numpy(spec_np).to(self.device, non_blocking=True)

        cfg = getattr(self.scheduler, "config", None)
        dbg = cfg is not None and bool(getattr(cfg, "debug_mode", False))
        if dbg:
            assert speculations.shape == (b_exp, self.lookahead + 1)
            assert logits_q.shape[0] == b_exp
            assert logits_q.shape[1] == self.lookahead
            cand_dbg = speculations[:, 1 : self.lookahead + 1]
            assert cand_dbg.shape == (b_exp, self.lookahead)
            assert torch.equal(speculations[:, 1], plan.root_token_ids)
            pib_t = plan.parent_index_per_branch.to(first_logits_q.device)
            assert torch.all(logits_q[:, 0, :] == first_logits_q[pib_t])

        no_expansion = plan.expanded_batch_size == plan.parent_batch_size or all(
            c == 1 for c in plan.branch_counts
        )

        if no_expansion:
            winner_rows = [-1] * batch_size
            for row_idx, (pidx, bidx) in enumerate(zip(parent_idx_list, branch_idx_list)):
                if int(bidx) == 0:
                    winner_rows[int(pidx)] = row_idx
            assert all(r >= 0 for r in winner_rows), winner_rows
            winners_branch = [0] * batch_size
            branch_scores_list = [0.0] * b_exp
            winning_score_per_parent = [0.0] * batch_size
        else:
            scores_t = self._branch_scores(speculations, logits_q)
            branch_scores_list = [float(x) for x in scores_t.detach().cpu().tolist()]
            per_parent_rows: list[list[int]] = [[] for _ in range(batch_size)]
            for row_idx, pidx in enumerate(parent_idx_list):
                per_parent_rows[pidx].append(row_idx)
            winner_rows = [0] * batch_size
            winners_branch = [0] * batch_size
            winning_score_per_parent = [0.0] * batch_size
            for pidx in range(batch_size):
                rows = per_parent_rows[pidx]
                best = rows[0]
                best_s = float(scores_t[best].item())
                for r in rows[1:]:
                    s = float(scores_t[r].item())
                    if s > best_s:
                        best = r
                        best_s = s
                winner_rows[pidx] = best
                winners_branch[pidx] = int(branch_idx_list[best])
                winning_score_per_parent[pidx] = best_s

        winner_rows_t = torch.tensor(winner_rows, dtype=torch.long, device=self.device)
        collapsed_speculations = speculations.index_select(0, winner_rows_t)
        collapsed_logits_q = logits_q.index_select(0, winner_rows_t)

        winning_roots: list[int] = []
        winning_expanded_rows: list[int] = []
        for pidx in range(batch_size):
            wr = winner_rows[pidx]
            winning_expanded_rows.append(wr)
            winning_roots.append(int(root_token_list[wr]))

        for pidx, parent in enumerate(seqs):
            clen = committed_len_per_parent[pidx]
            suffix = [int(x) for x in collapsed_speculations[pidx].tolist()]
            win_row = winner_rows[pidx]
            win_seq = expanded_seqs[win_row]
            assert win_seq is not None
            self._replace_parent_spec_tail(parent, clen, suffix)
            parent.num_draft_cached_tokens = win_seq.num_draft_cached_tokens
            parent.num_cached_tokens = clen

        self._commit_precollapse_draft_winner_and_release(
            seqs,
            expanded_seqs,
            branch_states,
            winners_branch,
        )

        decision = PivotPrecollapseDecision(
            winning_branch_idx_per_parent=list(winners_branch),
            winning_expanded_row_idx_per_parent=list(winning_expanded_rows),
            winning_root_token_per_parent=winning_roots,
            branch_score_per_row=branch_scores_list,
            winning_score_per_parent=winning_score_per_parent,
            branch_count_per_parent=list(host.branch_counts),
            before_expansion_batch_size=batch_size,
            after_expansion_batch_size=b_exp,
            score_method=self.score_method,
            committed_len_per_parent=list(committed_len_per_parent),
        )

        if dbg:
            for pidx, parent in enumerate(seqs):
                clen = committed_len_per_parent[pidx]
                assert parent.num_cached_tokens == clen
                assert parent.num_draft_cached_tokens == parent.num_tokens
                tail = list(parent.token_ids[clen:])
                cand = [int(x) for x in collapsed_speculations[pidx].detach().cpu().tolist()]
                assert tail == cand, (tail, cand)

        branch_bundle = PivotBranchBundle(
            parent_batch_size=batch_size,
            host_plan=plan.host,
            expanded_seqs=None,
            branch_states=None,
            precollapse_decision=decision,
        )

        t_final_pack1 = perf_counter()
        branch_construct_s = t_branch_construct1 - t_branch_construct0
        target_cow_copy_s = 0.0
        draft_cow_copy_s = t_draft_copy1 - t_draft_copy0
        inter_cow_copy_s = 0.0
        cow_copy_s = draft_cow_copy_s
        draft_cow_copy_mode = os.environ.get(
            "SSD_PIVOT_DRAFT_COW_COPY_MODE", "bucketed_partial"
        )
        branch0_setup_s = t_branch0_setup1 - t_branch0_setup0
        initial_pack_s = t_initial_pack1 - t_initial_pack0
        tail_draft_s = t_tail_draft1 - t_tail_draft0
        extra_draft_s = t_extra_draft1 - t_extra_draft0
        final_pack_s = t_final_pack1 - t_final_pack0
        self._write_pivot_cost_row(
            {
                "kind": "pivot_draft_microcost",
                "pivot_policy": "pivot_precollapse",
                "step_id": int(self._pivot_cost_step_id),
                "parent_batch_size": int(batch_size),
                "expanded_batch_size": int(b_exp),
                "pivot_before_expansion_batch_size": int(batch_size),
                "pivot_after_expansion_batch_size": int(b_exp),
                "pivot_after_collapse_batch_size": int(batch_size),
                "pivot_target_verify_batch_size": int(batch_size),
                "expansion_ratio": float(b_exp / batch_size) if batch_size > 0 else 0.0,
                "num_nonzero_branches": int(num_nonzero_branches),
                "num_target_cow_copy_blocks": 0,
                "pivot_num_target_cow_copy_blocks": 0,
                "num_draft_cow_copy_blocks": int(num_draft_cow_copy_blocks),
                "num_inter_cow_copy_blocks": int(num_inter_cow_copy_blocks),
                "pivot_plan_build_s": float(t_plan1 - t_plan0),
                "pivot_capacity_clamp_s": float(t_cap1 - t_cap0),
                "pivot_branch0_override_s": float(t_cap2 - t_cap1),
                "pivot_branch_construct_s": float(branch_construct_s),
                "pivot_cow_copy_s": float(cow_copy_s),
                "pivot_target_cow_copy_s": float(target_cow_copy_s),
                "pivot_draft_cow_copy_s": float(draft_cow_copy_s),
                "pivot_inter_cow_copy_s": float(inter_cow_copy_s),
                "pivot_draft_cow_copy_mode": draft_cow_copy_mode,
                "pivot_target_cow_copy_mode": "bucketed_partial",
                "pivot_inter_cow_copy_mode": "bucketed_partial",
                "pivot_branch0_setup_s": float(branch0_setup_s),
                "pivot_initial_pack_s": float(initial_pack_s),
                "pivot_tail_draft_forward_s": float(tail_draft_s),
                "pivot_extra_draft_forward_s": float(extra_draft_s),
                "pivot_final_pack_s": float(final_pack_s),
                "pivot_expand_pack_s": float(initial_pack_s + final_pack_s),
                "pivot_branch_construction_plus_cow_s": float(
                    branch_construct_s + cow_copy_s + branch0_setup_s
                ),
                "cuda_synchronized": os.environ.get("SSD_PROFILE_PIVOT_SYNC", "0") == "1",
            }
        )

        return SpeculateResult(
            speculations=collapsed_speculations,
            logits_q=collapsed_logits_q,
            cache_hits=None,
            branch_bundle=branch_bundle,
        )
