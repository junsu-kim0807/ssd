"""Sync pivot speculator with draft-score collapse before target verify.

Selection modes (set via ``pivot_precollapse_selection``):

- ``score`` / ``slope``: target verify always sees ``B`` rows. Single per-parent
  winner is committed as the parent's draft tail before target verify.
- ``score_expansion``: target verify sees ``B + selected_count`` rows. For each
  expanded parent, branch 0 (the vanilla draft continuation) AND the best alt
  branch (by accumulated draft confidence) survive precollapse and feed
  ``PivotExecutorFlat`` which performs the final target-authoritative collapse.
"""

from __future__ import annotations

import os
from collections import defaultdict
from time import perf_counter

import torch

from ssd.engine.block_manager import BlockManager, CowForkPlan
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_branch_planner import (
    PivotExpansionPlan,
    PivotHostPlan,
    build_pivot_expansion_plan,
    compute_dynamic_expansion_slope,
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
        precollapse_selection: str = "score",
        precollapse_slope_thresholds: tuple[float, float, float] = (-0.70, -0.58, -0.46),
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
        if precollapse_selection not in {"score", "slope", "score_expansion"}:
            raise ValueError(
                "pivot_precollapse_selection must be one of "
                "{'score', 'slope', 'score_expansion'}; "
                f"got {precollapse_selection!r}"
            )
        if len(precollapse_slope_thresholds) != 3:
            raise ValueError(
                "pivot_precollapse_slope_thresholds must have length 3; "
                f"got {len(precollapse_slope_thresholds)}"
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
        self.precollapse_selection = precollapse_selection
        self.precollapse_slope_thresholds = tuple(float(x) for x in precollapse_slope_thresholds)
        self._cached_selected_ranks_per_parent: list[int] | None = None

    def reset_step_state(self) -> None:
        super().reset_step_state()
        self._cached_selected_ranks_per_parent = None

    def _rank_from_slope(self, slope: float) -> int:
        t0, t1, t2 = self.precollapse_slope_thresholds
        if slope < t0:
            return 2
        if slope < t1:
            return 3
        if slope < t2:
            return 4
        return 5

    def _build_slope_direct_plan(
        self,
        first_logits_q: torch.Tensor,
        batch_size: int,
    ) -> tuple[PivotExpansionPlan, list[int], list[float]]:
        """B-row plan: one root per parent from Top-{2,3,4,5} by slope bucket (no expansion)."""
        tk = int(self.expansion_cfg.topk)
        slope_t, _top_vals, top_ids = compute_dynamic_expansion_slope(
            first_logits_q.float(), tk
        )
        device = first_logits_q.device
        top_ids_host = top_ids.detach().cpu().tolist()
        slope_list = slope_t.detach().cpu().tolist()
        selected_ranks: list[int] = []
        root_token_ids: list[int] = []
        for pidx in range(batch_size):
            rank = self._rank_from_slope(float(slope_list[pidx]))
            selected_ranks.append(rank)
            root_token_ids.append(int(top_ids_host[pidx][rank - 1]))

        pib = list(range(batch_size))
        bip = [0] * batch_size
        bcounts = [1] * batch_size
        expand_mask_host = [False] * batch_size

        host = PivotHostPlan(
            parent_index_per_branch=pib,
            branch_index_per_parent=bip,
            root_token_ids=root_token_ids,
            branch_counts=bcounts,
            expand_mask=expand_mask_host,
            criteria_scores=list(slope_list),
            root_token_probs=None,
            top1_probs=None,
            residual_scores=None,
            dynamic_expansion_slope_scores=list(slope_list),
        )

        zf = torch.zeros(batch_size, dtype=torch.float32, device=device)
        plan = PivotExpansionPlan(
            parent_batch_size=batch_size,
            expanded_batch_size=batch_size,
            expand_mask=torch.zeros(batch_size, dtype=torch.bool, device=device),
            parent_index_per_branch=torch.arange(
                batch_size, dtype=torch.int64, device=device
            ),
            branch_index_per_parent=torch.zeros(
                batch_size, dtype=torch.int64, device=device
            ),
            root_token_ids=torch.tensor(root_token_ids, dtype=torch.int64, device=device),
            root_token_probs=torch.zeros(batch_size, dtype=torch.float32, device=device),
            criteria_scores=slope_t.to(device),
            top1_probs=zf,
            residual_scores=zf,
            branch_counts=bcounts,
            branch_counts_tensor=torch.ones(batch_size, dtype=torch.int64, device=device),
            dynamic_expansion_slope_scores=slope_t.to(device),
            host=host,
        )
        return plan, selected_ranks, slope_list

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
        """Expansion clamp.

        - "score" / "slope" modes: draft-only (target / inter never forked).
        - "score_expansion" mode: also reserves target (and intermediate, when
          ``i_bm is not None``) capacity for exactly ONE alt row per kept parent
          because that alt row will survive precollapse and feed target verify.
        """
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
        t_bm: BlockManager = self.scheduler.block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager
        free_d = len(d_bm.free_block_ids)
        free_t = len(t_bm.free_block_ids)
        free_i = len(i_bm.free_block_ids) if i_bm is not None else 10**12

        score_expansion = self.precollapse_selection == "score_expansion"

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

            # score_expansion: one alt row per kept parent will survive precollapse.
            # Size at the committed frontier so the reservation matches the
            # post-scoring target COW fork (clen + lookahead + 1).
            # Note: ``seq.num_tokens + self.lookahead + 1`` would be off by one
            # because recovery has already been appended, making num_tokens =
            # committed_len + 1.
            if score_expansion:
                clen = seq.num_cached_tokens
                target_extra_one_alt = self._private_blocks_needed(
                    cached_tokens=clen,
                    required_tokens=clen + self.lookahead + 1,
                    block_size=seq.block_size,
                )
                inter_extra_one_alt = (
                    self._private_blocks_needed(
                        cached_tokens=seq.num_inter_cached_tokens,
                        required_tokens=seq.num_inter_cached_tokens + self.lookahead + 1,
                        block_size=seq.block_size,
                    )
                    if i_bm is not None
                    else 0
                )
                need_t = target_extra_one_alt
                need_i = inter_extra_one_alt
            else:
                need_t = 0
                need_i = 0

            if need_d <= free_d and need_t <= free_t and need_i <= free_i:
                kept_host[idx] = True
                free_d -= need_d
                free_t -= need_t
                free_i -= need_i

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

    def _release_score_expansion_loser_draft_forks(
        self,
        parent_seqs: list[Sequence],
        expanded_seqs: list[Sequence | None],
        branch_states: list[BranchForkState | None],
        retained_abs_rows: set[int],
    ) -> None:
        """Release draft forks for non-zero branches that were dropped by
        score_expansion precollapse (i.e. not branch 0 and not the best alt).

        Retained alt rows keep their draft fork intact because target verify
        still consumes their draft KV; ``PivotExecutorFlat._commit_winner_and_release_forks``
        will release or graft them after the target-authoritative collapse.
        Branch 0 is parent-in-place and is never touched here.

        Each loser contributes exactly +1 shared-prefix ref bump (from
        ``make_cow_fork_block_table``) and 1 private-tail block list. We
        decrement those refs and dealloc the private tails per parent.
        """
        d_bm: BlockManager = self.scheduler.draft_block_manager

        parent_loser_count_d: dict[int, int] = defaultdict(int)
        parent_d_shared: dict[int, int] = {}
        loser_d_tails: list[int] = []

        for row_idx, (seq, st) in enumerate(zip(expanded_seqs, branch_states)):
            if seq is None or st is None or st.is_parent_inplace:
                continue
            if not st.draft_kv_owned:
                continue
            if row_idx in retained_abs_rows:
                # Retained alt: keep its shared-prefix ref bump and private
                # tail; PivotExecutorFlat releases/grafts after target collapse.
                continue
            pidx = st.parent_seq_idx
            parent_loser_count_d[pidx] += 1
            parent_d_shared[pidx] = st.draft_shared_prefix_blocks
            loser_d_tails.extend(st.draft_private_tail_block_ids)

        for pidx, count in parent_loser_count_d.items():
            d_bm.release_shared_prefix_n(
                parent_seqs[pidx].draft_block_table, parent_d_shared[pidx], count
            )
        if loser_d_tails:
            d_bm._deallocate_n_blocks(loser_d_tails)

    def _allocate_target_cow_for_alt(
        self,
        parent: Sequence,
        clen: int,
    ) -> tuple[CowForkPlan, CowForkPlan | None]:
        """Allocate target (and intermediate, if present) COW forks for a
        retained alt branch using the committed frontier.

        Sizes are computed from ``clen`` (== ``num_cached_tokens`` at speculate
        entry, before recovery append). ``required_total_tokens`` is
        ``clen + lookahead + 1`` because target verify will consume ``K + 1``
        query tokens (recovery + K draft tokens) per row, leaving the target KV
        at exactly that depth. Using ``parent.num_tokens + lookahead`` here
        would over-reserve because the in-place branch-0 setup and the K-1
        tail-draft loop have already inflated ``parent.num_tokens`` to
        ``clen + 1 + K``.
        """
        t_bm: BlockManager = self.scheduler.block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager
        required_total = clen + self.lookahead + 1
        t_plan = t_bm.make_cow_fork_block_table(
            parent.block_table,
            cached_tokens=clen,
            required_total_tokens=required_total,
        )
        i_plan: CowForkPlan | None = None
        if i_bm is not None and self.intermediate_runner is not None:
            i_plan = i_bm.make_cow_fork_block_table(
                parent.inter_block_table,
                cached_tokens=clen,
                required_total_tokens=required_total,
            )
        return t_plan, i_plan

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
            selected_ranks_for_cache: list[int] | None = None
            if self.precollapse_selection == "slope":
                plan, selected_ranks_for_cache, _slope_scores = self._build_slope_direct_plan(
                    first_logits_q, batch_size
                )
                t_plan1 = perf_counter()
                t_cap0 = t_plan1
                t_cap1 = t_plan1
                t_cap2 = t_plan1
            else:
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
            self._cached_selected_ranks_per_parent = (
                list(selected_ranks_for_cache)
                if selected_ranks_for_cache is not None
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
            if self.precollapse_selection != "slope":
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
        speculations_exp = torch.empty(
            (b_exp, self.lookahead + 1), dtype=torch.long, device=self.device
        )
        host_spec_rows: list[list[int]] = []
        if b_exp > 0:
            recovery_t = torch.tensor(
                recovery_tokens, dtype=torch.long, device=self.device
            )
            pib_t = plan.parent_index_per_branch.to(
                device=self.device, dtype=torch.long
            )
            speculations_exp[:, 0] = recovery_t.index_select(0, pib_t)
            speculations_exp[:, 1] = plan.root_token_ids.to(
                device=self.device, dtype=torch.long
            )
            tail_pad = self.lookahead - 1
            host_spec_rows = [
                [recovery_tokens[parent_idx_list[i]], int(root_token_list[i])]
                + [0] * tail_pad
                for i in range(b_exp)
            ]

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
                token_ids_k_t = (
                    token_ids_k
                    if isinstance(token_ids_k, torch.Tensor)
                    else torch.tensor(
                        token_ids_k, dtype=torch.long, device=self.device
                    )
                )
                speculations_exp[:, _k + 1] = token_ids_k_t
                for row_idx, (seq, tok) in enumerate(zip(expanded_seqs, token_ids_k)):
                    if seq is None:
                        continue
                    seq.append_token(tok)
                    host_spec_rows[row_idx][_k + 1] = int(tok)
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

        cfg = getattr(self.scheduler, "config", None)
        dbg = cfg is not None and bool(getattr(cfg, "debug_mode", False))
        if dbg:
            assert speculations_exp.shape == (b_exp, self.lookahead + 1)
            assert len(logits_q_rows) == self.lookahead
            assert torch.equal(
                speculations_exp[:, 1],
                plan.root_token_ids.to(speculations_exp.device, dtype=torch.long),
            )
            pib_t_dbg = plan.parent_index_per_branch.to(first_logits_q.device)
            assert torch.all(logits_q_rows[0] == first_logits_q[pib_t_dbg])

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
            scores_t = torch.zeros(b_exp, dtype=torch.float32, device=self.device)
            if self.score_method == "logit_sum":
                for j, logits_j in enumerate(logits_q_rows):
                    tok_j = speculations_exp[:, j + 1]
                    scores_t = scores_t + logits_j.gather(
                        1, tok_j.unsqueeze(1)
                    ).squeeze(1).float()
            else:  # logprob_sum
                for j, logits_j in enumerate(logits_q_rows):
                    v = logits_j.float()
                    tok_j = speculations_exp[:, j + 1]
                    scores_t = scores_t + (
                        v.gather(1, tok_j.unsqueeze(1)).squeeze(1)
                        - torch.logsumexp(v, dim=-1)
                    )
            branch_scores_list = [float(x) for x in scores_t.detach().cpu().tolist()]
            per_parent_rows: list[list[int]] = [[] for _ in range(batch_size)]
            for row_idx, pidx in enumerate(parent_idx_list):
                per_parent_rows[pidx].append(row_idx)
            winner_rows = [0] * batch_size
            winners_branch = [0] * batch_size
            winning_score_per_parent = [0.0] * batch_size
            for pidx in range(batch_size):
                rows = per_parent_rows[pidx]
                # Prefer non-branch-0 when this parent was expanded: collapse winner is argmax
                # over alt branches only; branch 0 stays in the batch for vanilla / fallback.
                alt_rows = [r for r in rows if int(branch_idx_list[r]) != 0]
                if alt_rows:
                    best = alt_rows[0]
                    best_s = branch_scores_list[best]
                    for r in alt_rows[1:]:
                        s = branch_scores_list[r]
                        if s > best_s:
                            best = r
                            best_s = s
                else:
                    best = rows[0]
                    best_s = branch_scores_list[best]
                winner_rows[pidx] = best
                winners_branch[pidx] = int(branch_idx_list[best])
                winning_score_per_parent[pidx] = best_s

        # ------------------------------------------------------------------
        # Compute per-parent winning roots / expanded-row indices for the
        # speculator's PivotPrecollapseDecision (used by all selection modes;
        # winners_branch may be re-derived by score_expansion downstream).
        # ------------------------------------------------------------------
        winning_roots: list[int] = []
        winning_expanded_rows: list[int] = []
        for pidx in range(batch_size):
            wr = winner_rows[pidx]
            winning_expanded_rows.append(wr)
            winning_roots.append(int(root_token_list[wr]))

        score_expansion = self.precollapse_selection == "score_expansion"

        target_cow_copy_s = 0.0
        if not score_expansion:
            # ----- score / slope: collapse to B rows; graft single winner -----
            winner_rows_t = torch.tensor(
                winner_rows, dtype=torch.long, device=self.device
            )
            out_speculations = speculations_exp.index_select(0, winner_rows_t)
            out_logits_q = torch.stack(
                [logits_j.index_select(0, winner_rows_t) for logits_j in logits_q_rows],
                dim=1,
            )

            for pidx, parent in enumerate(seqs):
                clen = committed_len_per_parent[pidx]
                win_row = winner_rows[pidx]
                suffix = list(host_spec_rows[win_row])
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

            retained_host = plan.host
            retained_seqs_for_bundle: list[Sequence] | None = None
            retained_states_for_bundle: list[BranchForkState] | None = None
            after_collapse_bsz = batch_size
            target_verify_bsz = batch_size
        else:
            # ----- score_expansion: retain branch 0 + best alt per parent -----
            per_parent_rows_all: list[list[int]] = [[] for _ in range(batch_size)]
            for row_idx, pidx in enumerate(parent_idx_list):
                per_parent_rows_all[pidx].append(row_idx)

            retained_abs_rows: list[int] = []
            parent_index_retained: list[int] = []
            branch_index_retained: list[int] = []
            branch_counts_retained: list[int] = []
            root_token_retained: list[int] = []
            root_prob_retained: list[float] = []
            alt_winner_row_per_parent: list[int | None] = [None] * batch_size

            for pidx in range(batch_size):
                rows_p = per_parent_rows_all[pidx]
                # Branch 0 is always added first per-parent. PivotExecutorFlat
                # initializes ``best_row = rows[0]`` and only switches winner
                # on strictly-longer accept_len; putting alt at index 0 would
                # silently invert "branch 0 unless alt strictly better" semantics.
                branch0_row = next(
                    r for r in rows_p if int(branch_idx_list[r]) == 0
                )
                retained_abs_rows.append(branch0_row)
                parent_index_retained.append(pidx)
                branch_index_retained.append(0)
                root_token_retained.append(int(root_token_list[branch0_row]))
                root_prob_retained.append(float(root_prob_list[branch0_row]))

                alt_rows = [r for r in rows_p if int(branch_idx_list[r]) != 0]
                if alt_rows:
                    best = alt_rows[0]
                    best_s = branch_scores_list[best]
                    for r in alt_rows[1:]:
                        s = branch_scores_list[r]
                        if s > best_s:
                            best = r
                            best_s = s
                    retained_abs_rows.append(best)
                    parent_index_retained.append(pidx)
                    # Keep the original sparse alt branch index (e.g. could be 3
                    # under topk=5). PivotExecutorFlat looks winners up via
                    # branch_index_per_parent equality, so sparse is fine.
                    branch_index_retained.append(int(branch_idx_list[best]))
                    root_token_retained.append(int(root_token_list[best]))
                    root_prob_retained.append(float(root_prob_list[best]))
                    branch_counts_retained.append(2)
                    alt_winner_row_per_parent[pidx] = best
                    # Update precollapse decision to reflect the actual retained
                    # alt (not a no-alt fallback to branch 0 as the legacy
                    # alt-preferred winner).
                    winners_branch[pidx] = int(branch_idx_list[best])
                    winner_rows[pidx] = best
                    winning_roots[pidx] = int(root_token_list[best])
                    winning_expanded_rows[pidx] = best
                    winning_score_per_parent[pidx] = best_s
                else:
                    branch_counts_retained.append(1)
                    winners_branch[pidx] = 0
                    winner_rows[pidx] = branch0_row
                    winning_roots[pidx] = int(root_token_list[branch0_row])
                    winning_expanded_rows[pidx] = branch0_row

            selected_count = sum(1 for c in branch_counts_retained if c > 1)
            b_score = batch_size + selected_count
            assert b_score == len(retained_abs_rows), (b_score, len(retained_abs_rows))

            # Branch-0-first invariant assertion.
            by_parent_rows_retained: list[list[int]] = [[] for _ in range(batch_size)]
            for r, pidx in enumerate(parent_index_retained):
                by_parent_rows_retained[pidx].append(r)
            for pidx, rows_p in enumerate(by_parent_rows_retained):
                assert rows_p, pidx
                assert int(branch_index_retained[rows_p[0]]) == 0, (
                    pidx,
                    rows_p,
                    branch_index_retained,
                )

            # Slice speculations / logits to retained rows.
            retained_rows_t = torch.tensor(
                retained_abs_rows, dtype=torch.long, device=self.device
            )
            out_speculations = speculations_exp.index_select(0, retained_rows_t)
            out_logits_q = torch.stack(
                [
                    logits_j.index_select(0, retained_rows_t)
                    for logits_j in logits_q_rows
                ],
                dim=1,
            )

            # Allocate target / inter COW for each retained alt row.
            t_bm: BlockManager = self.scheduler.block_manager
            target_copy_src_all: list[int] = []
            target_copy_dst_all: list[int] = []
            target_copy_valid_all: list[int] = []
            inter_copy_src_all: list[int] = []
            inter_copy_dst_all: list[int] = []
            inter_copy_valid_all: list[int] = []

            for pidx in range(batch_size):
                alt_row = alt_winner_row_per_parent[pidx]
                if alt_row is None:
                    continue
                parent = seqs[pidx]
                alt_seq = expanded_seqs[alt_row]
                alt_state = branch_states[alt_row]
                assert alt_seq is not None and alt_state is not None
                clen = committed_len_per_parent[pidx]
                t_plan, i_plan = self._allocate_target_cow_for_alt(parent, clen)

                # Patch alt sequence's target / inter block tables and patch
                # the BranchForkState to advertise target ownership.
                alt_seq.block_table = t_plan.fork_block_table
                if i_plan is not None:
                    alt_seq.inter_block_table = i_plan.fork_block_table

                if t_plan.copy_src_block_ids:
                    target_copy_src_all.extend(t_plan.copy_src_block_ids)
                    target_copy_dst_all.extend(t_plan.copy_dst_block_ids)
                    target_copy_valid_all.extend(t_plan.copy_valid_tokens)
                    num_target_cow_copy_blocks += len(t_plan.copy_src_block_ids)
                if i_plan is not None and i_plan.copy_src_block_ids:
                    inter_copy_src_all.extend(i_plan.copy_src_block_ids)
                    inter_copy_dst_all.extend(i_plan.copy_dst_block_ids)
                    inter_copy_valid_all.extend(i_plan.copy_valid_tokens)
                    num_inter_cow_copy_blocks += len(i_plan.copy_src_block_ids)

                alt_state.target_shared_prefix_blocks = t_plan.shared_prefix_blocks
                alt_state.target_private_tail_block_ids = t_plan.private_tail_block_ids
                alt_state.target_kv_owned = True
                if i_plan is not None:
                    alt_state.inter_shared_prefix_blocks = i_plan.shared_prefix_blocks
                    alt_state.inter_private_tail_block_ids = i_plan.private_tail_block_ids
                    alt_state.inter_kv_owned = True

            self._cuda_sync_for_pivot_cost()
            t_target_copy0 = perf_counter()
            if target_copy_src_all:
                self.target_model_runner.call(
                    "copy_kv_blocks",
                    target_copy_src_all,
                    target_copy_dst_all,
                    target_copy_valid_all,
                    "target",
                )
            if (
                inter_copy_src_all
                and self.intermediate_runner is not None
            ):
                self.intermediate_runner.call(
                    "copy_kv_blocks",
                    inter_copy_src_all,
                    inter_copy_dst_all,
                    inter_copy_valid_all,
                    "intermediate",
                )
            self._cuda_sync_for_pivot_cost()
            target_cow_copy_s = perf_counter() - t_target_copy0

            # Release dropped non-zero non-alt-winner draft forks. Retained
            # alt's draft fork is left intact; PivotExecutorFlat will release
            # or graft it after target collapse. Branch 0 is parent-in-place.
            self._release_score_expansion_loser_draft_forks(
                seqs,
                expanded_seqs,
                branch_states,
                set(retained_abs_rows),
            )

            # Build filtered host plan that only contains retained rows.
            host_src = plan.host
            assert host_src is not None
            criteria_scores_h = (
                list(host_src.criteria_scores)
                if host_src.criteria_scores is not None
                else None
            )
            top1_probs_h = (
                list(host_src.top1_probs)
                if host_src.top1_probs is not None
                else None
            )
            residual_scores_h = (
                list(host_src.residual_scores)
                if host_src.residual_scores is not None
                else None
            )
            slope_h = (
                list(host_src.dynamic_expansion_slope_scores)
                if host_src.dynamic_expansion_slope_scores is not None
                else None
            )
            expand_mask_h = [c > 1 for c in branch_counts_retained]
            retained_host = PivotHostPlan(
                parent_index_per_branch=list(parent_index_retained),
                branch_index_per_parent=list(branch_index_retained),
                root_token_ids=list(root_token_retained),
                branch_counts=list(branch_counts_retained),
                expand_mask=expand_mask_h,
                criteria_scores=criteria_scores_h,
                root_token_probs=list(root_prob_retained) if host_src.root_token_probs is not None else None,
                top1_probs=top1_probs_h,
                residual_scores=residual_scores_h,
                dynamic_expansion_slope_scores=slope_h,
            )

            # Retained Sequence and BranchForkState lists (one entry per
            # retained row, in retained order). PivotExecutorFlat raises if
            # expanded_seqs is None — score_expansion always populates these,
            # including branch-0-only rows for no-expansion parents.
            retained_seqs_for_bundle = [expanded_seqs[r] for r in retained_abs_rows]
            retained_states_for_bundle = [branch_states[r] for r in retained_abs_rows]
            for s in retained_seqs_for_bundle:
                assert s is not None
            for st in retained_states_for_bundle:
                assert st is not None

            after_collapse_bsz = b_score
            target_verify_bsz = b_score

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
            selected_root_rank_per_parent=(
                list(self._cached_selected_ranks_per_parent)
                if self.precollapse_selection == "slope"
                and self._cached_selected_ranks_per_parent is not None
                else None
            ),
            slope_score_per_parent=(
                list(self._cached_slope_scores_per_parent)
                if self.precollapse_selection == "slope"
                and self._cached_slope_scores_per_parent is not None
                else None
            ),
        )

        if dbg and not score_expansion:
            for pidx, parent in enumerate(seqs):
                clen = committed_len_per_parent[pidx]
                assert parent.num_cached_tokens == clen
                assert parent.num_draft_cached_tokens == parent.num_tokens
                tail = list(parent.token_ids[clen:])
                cand = [int(x) for x in out_speculations[pidx].detach().cpu().tolist()]
                assert tail == cand, (tail, cand)

        if score_expansion:
            branch_bundle = PivotBranchBundle(
                parent_batch_size=batch_size,
                host_plan=retained_host,
                expanded_seqs=retained_seqs_for_bundle,
                branch_states=retained_states_for_bundle,
                precollapse_decision=decision,
            )
        else:
            branch_bundle = PivotBranchBundle(
                parent_batch_size=batch_size,
                host_plan=plan.host,
                expanded_seqs=None,
                branch_states=None,
                precollapse_decision=decision,
            )

        t_final_pack1 = perf_counter()
        branch_construct_s = t_branch_construct1 - t_branch_construct0
        draft_cow_copy_s = t_draft_copy1 - t_draft_copy0
        inter_cow_copy_s = 0.0
        cow_copy_s = draft_cow_copy_s + target_cow_copy_s
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
                "pivot_after_collapse_batch_size": int(after_collapse_bsz),
                "pivot_target_verify_batch_size": int(target_verify_bsz),
                "expansion_ratio": float(b_exp / batch_size) if batch_size > 0 else 0.0,
                "num_nonzero_branches": int(num_nonzero_branches),
                "num_target_cow_copy_blocks": int(num_target_cow_copy_blocks),
                "pivot_num_target_cow_copy_blocks": int(num_target_cow_copy_blocks),
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
            speculations=out_speculations,
            logits_q=out_logits_q,
            cache_hits=None,
            branch_bundle=branch_bundle,
        )
