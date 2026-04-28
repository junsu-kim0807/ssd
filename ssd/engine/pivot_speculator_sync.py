from __future__ import annotations

import json
import numpy as np
import os
from time import perf_counter
import torch

from ssd.engine.block_manager import BlockManager, CowForkPlan
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_branch_planner import (
    PivotExpansionConfig,
    PivotExpansionPlan,
    PivotHostPlan,
    build_pivot_expansion_plan,
)
from ssd.engine.pivot_types import BranchForkState, PivotBranchBundle
from ssd.engine.sequence import Sequence
from ssd.engine.speculator_sync import SpeculatorSync


class PivotRootSpeculatorSync(SpeculatorSync):
    """Sync speculator with planner-driven one-time root expansion."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        draft_model_runner,
        target_model_runner,
        intermediate_runner,
        scheduler,
        expansion_cfg: PivotExpansionConfig,
        max_expand_rows: int | None = None,
        enable_profile_trace: bool = False,
    ):
        super().__init__(lookahead, device, draft_model_runner)
        self.target_model_runner = target_model_runner
        self.intermediate_runner = intermediate_runner
        self.scheduler = scheduler
        self.expansion_cfg = expansion_cfg
        self.max_expand_rows = max_expand_rows
        self.enable_profile_trace = enable_profile_trace
        self._cached_root_tokens_per_parent: list[list[int]] | None = None
        self._pivot_cost_step_id = -1
        self._pivot_cost_writer_pid = os.getpid()

    def reset_step_state(self) -> None:
        self._cached_root_tokens_per_parent = None
        self._pivot_cost_step_id += 1

    def _pivot_cost_enabled(self) -> bool:
        cfg = getattr(self.scheduler, "config", None)
        if cfg is None:
            return False
        out_dir = getattr(cfg, "profiler_output_dir", None)
        if not out_dir or not str(out_dir).strip():
            return False
        return getattr(cfg, "profiler_mode", "") in {"cost_breakdown", "cost_metadata"}

    @staticmethod
    def _cuda_sync_for_pivot_cost() -> None:
        if os.environ.get("SSD_PROFILE_PIVOT_SYNC", "0") == "1":
            torch.cuda.synchronize()

    def _write_pivot_cost_row(self, row: dict) -> None:
        # Disabled: do not write legacy pivot microcost JSONL rows.
        _ = row
        return

    def _debug_pivot_expansion(
        self,
        *,
        first_logits_q: torch.Tensor,
        plan: PivotExpansionPlan,
    ) -> None:
        cfg = getattr(self.scheduler, "config", None)
        if not bool(getattr(cfg, "debug_mode", False)):
            return
        probs = torch.softmax(first_logits_q.float(), dim=-1)
        topk = min(int(self.expansion_cfg.topk), int(probs.shape[-1]))
        topk_probs, topk_ids = torch.topk(probs, k=topk, dim=-1)
        selected_parent_ids = torch.nonzero(plan.expand_mask, as_tuple=False).view(-1).tolist()
        row = {
            "pivot_round_debug": True,
            "stage": "expansion",
            "expansion_criteria": str(self.expansion_cfg.criteria),
            "before_expansion_input_shape": list(first_logits_q.shape),
            "after_expansion_input_shape": [
                int(plan.expanded_batch_size),
                int(self.lookahead + 1),
            ],
            "selected_request_ids": [int(i) for i in selected_parent_ids],
            "parent_token_ids": [int(x) for x in topk_ids[:, 0].tolist()],
            "parent_token_confidence": [float(x) for x in plan.criteria_scores.tolist()],
            "topk_token_ids": [[int(tok) for tok in toks] for toks in topk_ids.tolist()],
            "topk_token_probs": [[float(p) for p in ps] for ps in topk_probs.tolist()],
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)

    @staticmethod
    def _num_blocks(tokens: int, block_size: int) -> int:
        return (max(0, tokens) + block_size - 1) // block_size

    @staticmethod
    def _private_blocks_needed(
        cached_tokens: int,
        required_tokens: int,
        block_size: int,
    ) -> int:
        assert required_tokens >= cached_tokens, (
            f"required_tokens must be >= cached_tokens, got "
            f"required={required_tokens}, cached={cached_tokens}"
        )
        required_blocks = (required_tokens + block_size - 1) // block_size
        full_shared_blocks = cached_tokens // block_size
        return max(0, required_blocks - full_shared_blocks)

    def _override_branch0_roots(
        self,
        plan: PivotExpansionPlan,
        first_token_ids: list[int],
        first_logits_q: torch.Tensor,
    ) -> None:
        """Force branch 0's root token to match vanilla draft sampler output.

        ``build_pivot_expansion_plan`` derives root tokens from
        ``torch.topk(first_logits_q.float(), ...)``. The float upcast and any
        sampler-side temperature handling can yield a different argmax than
        ``first_token_ids`` (which came through the production sampler). Branch
        0 is supposed to be the parent's natural continuation - i.e. identical
        to the vanilla single-branch draft - so override its root in-place on
        both the host plan and the GPU tensor mirror. Branches >= 1 keep the
        topk candidates from the planner.
        """
        host = plan.host
        if host is None or plan.expanded_batch_size == 0:
            return
        # Collect (row, parent) pairs where branch_idx == 0.
        rows_to_patch: list[int] = []
        new_tokens: list[int] = []
        for row_idx, bidx in enumerate(host.branch_index_per_parent):
            if bidx == 0:
                pidx = host.parent_index_per_branch[row_idx]
                tok = int(first_token_ids[pidx])
                if host.root_token_ids[row_idx] != tok:
                    host.root_token_ids[row_idx] = tok
                    rows_to_patch.append(row_idx)
                    new_tokens.append(tok)
        if not rows_to_patch:
            return
        # One scatter into the GPU tensor; avoids per-row .item() / index_put_.
        rows_t = torch.tensor(rows_to_patch, dtype=torch.long, device=plan.root_token_ids.device)
        toks_t = torch.tensor(new_tokens, dtype=plan.root_token_ids.dtype, device=plan.root_token_ids.device)
        plan.root_token_ids.index_copy_(0, rows_t, toks_t)
        # Keep profile metadata consistent: when branch-0 roots are forced to the
        # sampler output, update both host/GPU root probs for patched rows only.
        if host.root_token_probs is not None:
            probs = torch.softmax(first_logits_q.float(), dim=-1)
            pids_t = torch.tensor(
                [host.parent_index_per_branch[r] for r in rows_to_patch],
                dtype=torch.long,
                device=plan.root_token_ids.device,
            )
            probs_t = probs[pids_t, toks_t].to(torch.float32)
            plan.root_token_probs.index_copy_(0, rows_t, probs_t)
            probs_host = probs_t.detach().cpu().tolist()
            for row_idx, prob in zip(rows_to_patch, probs_host):
                host.root_token_probs[row_idx] = float(prob)

    def _apply_kv_capacity_limit(
        self,
        seqs: list[Sequence],
        plan: PivotExpansionPlan,
        *,
        fork_target_kv: bool = True,
    ) -> PivotExpansionPlan:
        """Clamp expansion by BM free blocks and max_model_len in addition to row cap.

        All ordering / per-parent decisions run on the host plan to avoid GPU
        syncs. Only the final mask rebuild touches GPU tensors when the clamp
        actually changes the plan.
        """
        bsz = len(seqs)
        if bsz == 0:
            return plan
        host = plan.host
        assert host is not None, "PivotExpansionPlan.host required for capacity clamp"
        topk = max(plan.branch_counts) if plan.branch_counts else 1

        # Start from planner-selected expand mask (host bool list).
        expand_mask_host = list(host.expand_mask)

        # Enforce model length at request-level.
        for i, seq in enumerate(seqs):
            if seq.num_tokens + self.lookahead > self.scheduler.max_model_len:
                expand_mask_host[i] = False

        # Clamp by aggregate free-block budgets (target/draft/intermediate).
        t_bm: BlockManager = self.scheduler.block_manager
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager

        free_t = len(t_bm.free_block_ids)
        free_d = len(d_bm.free_block_ids)
        free_i = len(i_bm.free_block_ids) if i_bm is not None else 10**12

        block_size0 = int(seqs[0].block_size)
        nscratch0 = (self.lookahead + 1 + block_size0 - 1) // block_size0
        if not fork_target_kv:
            # Phase-1A: every expanded row allocates ``nscratch`` target scratch blocks.
            # Reserve one row per parent up front; loop below charges (topk-1) extras
            # when expansion is approved for a parent.
            baseline_t = bsz * nscratch0
            scratch_baseline_ok = baseline_t <= free_t
            if scratch_baseline_ok:
                free_t -= baseline_t
        else:
            scratch_baseline_ok = True

        # Deterministic keep order by ascending criteria score: lower logit
        # margin == higher uncertainty, so high-uncertainty parents are
        # allocated KV budget first. Tie-break by index for determinism.
        scores_host = host.criteria_scores
        assert scores_host is not None
        order = sorted(range(bsz), key=lambda i: (scores_host[i], i))
        kept_host = [False] * bsz

        for idx in order:
            if not expand_mask_host[idx]:
                continue
            seq = seqs[idx]
            required_tokens = seq.num_tokens + self.lookahead

            if fork_target_kv:
                per_extra_t = self._private_blocks_needed(
                    seq.num_cached_tokens,
                    required_tokens,
                    seq.block_size,
                )
            per_extra_d = self._private_blocks_needed(
                seq.num_draft_cached_tokens,
                required_tokens,
                seq.block_size,
            )
            per_extra_i = (
                self._private_blocks_needed(
                    seq.num_inter_cached_tokens,
                    required_tokens,
                    seq.block_size,
                )
                if i_bm is not None
                else 0
            )
            extras = topk - 1
            if fork_target_kv:
                need_t = per_extra_t * extras
            elif scratch_baseline_ok:
                need_t = nscratch0 * extras
            else:
                need_t = 10**18
            need_d = per_extra_d * extras
            need_i = per_extra_i * extras
            if need_t <= free_t and need_d <= free_d and need_i <= free_i:
                kept_host[idx] = True
                free_t -= need_t
                free_d -= need_d
                free_i -= need_i

        # Apply ``max_expand_rows`` row-cap on host. Mirrors apply_capacity_limit().
        clamped_host = self._apply_row_cap_host(kept_host, scores_host, topk)

        if clamped_host == host.expand_mask:
            return plan

        # Rebuild plan from host_plan + clamped mask. Reuse cached per-parent
        # candidate roots from the original plan (no .tolist() — host already has them).
        per_parent_roots: list[list[tuple[int, float]]] = [[] for _ in range(bsz)]
        rp_src = host.root_token_probs  # may be None when profile trace is off
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
            host=new_host,
        )

    def _apply_row_cap_host(
        self,
        kept: list[bool],
        scores: list[float],
        topk: int,
    ) -> list[bool]:
        """Host-side equivalent of ``apply_capacity_limit``.

        Mirrors the GPU implementation: when ``max_expand_rows`` is set, retain
        at most ``(max_rows - bsz) // (topk - 1)`` expanded parents, prioritized
        by ascending criteria score (low-uncertainty first).
        """
        if self.max_expand_rows is None:
            return list(kept)
        if topk <= 1:
            return [False] * len(kept)
        bsz = len(kept)
        max_rows = int(self.max_expand_rows)
        if max_rows <= bsz:
            return [False] * bsz
        max_expand_reqs = (max_rows - bsz) // (topk - 1)
        max_expand_reqs = max(0, min(max_expand_reqs, bsz))
        if max_expand_reqs == 0:
            return [False] * bsz
        # Order by ascending score, then index for determinism.
        candidate_idxs = [i for i, k in enumerate(kept) if k]
        candidate_idxs.sort(key=lambda i: (scores[i], i))
        keep_set = set(candidate_idxs[:max_expand_reqs])
        return [i in keep_set for i in range(bsz)]

    @staticmethod
    def _copy_kv_block(
        kv_cache: torch.Tensor,
        src_block_id: int,
        dst_block_id: int,
        valid_tokens: int,
    ) -> None:
        if valid_tokens <= 0:
            return
        kv_cache[:, :, dst_block_id, :valid_tokens].copy_(kv_cache[:, :, src_block_id, :valid_tokens])

    def _copy_partial_cow_block(
        self,
        *,
        copy_src_block_ids: list[int],
        copy_dst_block_ids: list[int],
        copy_valid_tokens: list[int],
        kv_cache: torch.Tensor,
    ) -> None:
        for src_block, dst_block, valid_tokens in zip(
            copy_src_block_ids,
            copy_dst_block_ids,
            copy_valid_tokens,
        ):
            self._copy_kv_block(kv_cache, src_block, dst_block, valid_tokens)

    def speculate(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
        fork_target_kv: bool = True,
    ) -> SpeculateResult:
        assert not verify_result.eagle_acts, "Eagle is not supported for pivot sync speculation"
        batch_size = len(seqs)
        if batch_size == 0:
            return SpeculateResult(
                speculations=torch.zeros(0, self.lookahead + 1, dtype=torch.int64, device=self.device),
                logits_q=torch.zeros(0, self.lookahead, 0, dtype=torch.float32, device=self.device),
            )

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

        first_token_ids, first_logits_q = self.draft_model_runner.call(
            "run", seqs, False, True, True
        )
        for s in seqs:
            s.num_draft_cached_tokens += 1

        t_plan0 = perf_counter()
        if self._cached_root_tokens_per_parent is None:
            plan = build_pivot_expansion_plan(
                first_logits_q,
                self.expansion_cfg,
                max_expand_rows=self.max_expand_rows,
                materialize_host=True,
                profile_metadata=self.enable_profile_trace,
            )
            t_plan1 = perf_counter()
            t_cap0 = perf_counter()
            plan = self._apply_kv_capacity_limit(seqs, plan, fork_target_kv=fork_target_kv)
            t_cap1 = perf_counter()
            # Vanilla parity: branch 0 must equal the sampler's first_token_ids,
            # not the planner's float-topk argmax. Apply on every speculate()
            # call before populating the cached-root structure.
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
        else:
            parent_index_per_branch: list[int] = []
            branch_index_per_parent: list[int] = []
            root_token_ids: list[int] = []
            branch_counts: list[int] = []
            for pidx, roots in enumerate(self._cached_root_tokens_per_parent):
                branch_counts.append(len(roots))
                for bidx, tok in enumerate(roots):
                    parent_index_per_branch.append(pidx)
                    branch_index_per_parent.append(bidx)
                    # ``roots`` is already list[int] from the prior speculate() call.
                    root_token_ids.append(tok)

            b_exp_cached = len(parent_index_per_branch)
            # Batch root_token_probs gather: 1 sync instead of O(B_exp).
            # Skip materialization entirely when profile trace is off.
            if self.enable_profile_trace and b_exp_cached > 0:
                probs = torch.softmax(first_logits_q.float(), dim=-1)
                pids_t = torch.tensor(parent_index_per_branch, dtype=torch.long, device=self.device)
                rids_t = torch.tensor(root_token_ids, dtype=torch.long, device=self.device)
                root_token_probs_t = probs[pids_t, rids_t].to(torch.float32)
            else:
                root_token_probs_t = torch.zeros(b_exp_cached, dtype=torch.float32, device=self.device)

            expand_mask_host = [len(r) > 1 for r in self._cached_root_tokens_per_parent]
            # Cached path has no fresh logits-derived ordering signal; use zero
            # scores so capacity clamp falls back to deterministic index order.
            host = PivotHostPlan(
                parent_index_per_branch=list(parent_index_per_branch),
                branch_index_per_parent=list(branch_index_per_parent),
                root_token_ids=list(root_token_ids),
                branch_counts=list(branch_counts),
                expand_mask=expand_mask_host,
                criteria_scores=[0.0] * batch_size,
                root_token_probs=(
                    root_token_probs_t.tolist() if self.enable_profile_trace else None
                ),
                top1_probs=([0.0] * batch_size) if self.enable_profile_trace else None,
                residual_scores=([0.0] * batch_size) if self.enable_profile_trace else None,
            )
            plan = PivotExpansionPlan(
                parent_batch_size=batch_size,
                expanded_batch_size=b_exp_cached,
                expand_mask=torch.tensor(
                    expand_mask_host,
                    dtype=torch.bool,
                    device=self.device,
                ),
                parent_index_per_branch=torch.tensor(parent_index_per_branch, dtype=torch.int64, device=self.device),
                branch_index_per_parent=torch.tensor(branch_index_per_parent, dtype=torch.int64, device=self.device),
                root_token_ids=torch.tensor(root_token_ids, dtype=torch.int64, device=self.device),
                root_token_probs=root_token_probs_t,
                criteria_scores=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                top1_probs=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                residual_scores=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                branch_counts=branch_counts,
                branch_counts_tensor=torch.tensor(branch_counts, dtype=torch.int64, device=self.device),
                host=host,
            )
            t_plan1 = perf_counter()
            t_cap0 = perf_counter()
            plan = self._apply_kv_capacity_limit(seqs, plan, fork_target_kv=fork_target_kv)
            t_cap1 = perf_counter()
            # Same vanilla parity override on the cached path: the cached
            # branches >=1 stay fixed across rounds, but branch 0 must follow
            # the freshly-sampled ``first_token_ids`` for this round.
            self._override_branch0_roots(plan, first_token_ids, first_logits_q)
            t_cap2 = perf_counter()

        self._debug_pivot_expansion(first_logits_q=first_logits_q, plan=plan)

        # Two-pass branch construction:
        #   Pass 1: clone every B>0 branch BEFORE the parent gets its root appended.
        #           Otherwise ``parent.clone_spec()`` would copy a tape that already
        #           contains branch 0's root token.
        #   Pass 2: take ``branch 0`` in-place on the parent (no fork) so its draft/
        #           target KV is written directly into the parent's block table —
        #           which is what ``scheduler.may_append`` already pre-allocated for.
        # Reuse the already-materialized host plan; no extra .tolist() syncs.
        assert plan.host is not None
        host = plan.host
        parent_idx_list = host.parent_index_per_branch
        branch_idx_list = host.branch_index_per_parent
        root_token_list = host.root_token_ids
        # ``root_token_probs`` is profile-only — fall back to zeros when absent
        # so ``BranchForkState.root_confidence`` stays well-defined.
        root_prob_list = (
            host.root_token_probs
            if host.root_token_probs is not None
            else [0.0] * len(parent_idx_list)
        )
        b_exp = len(parent_idx_list)
        expanded_seqs: list[Sequence] = [None] * b_exp  # type: ignore[list-item]
        branch_states: list[BranchForkState] = [None] * b_exp  # type: ignore[list-item]

        t_bm: BlockManager = self.scheduler.block_manager
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager
        target_copy_src_all: list[int] = []
        target_copy_dst_all: list[int] = []
        target_copy_valid_all: list[int] = []
        draft_copy_src_all: list[int] = []
        draft_copy_dst_all: list[int] = []
        draft_copy_valid_all: list[int] = []
        inter_copy_src_all: list[int] = []
        inter_copy_dst_all: list[int] = []
        inter_copy_valid_all: list[int] = []
        num_nonzero_branches = 0
        num_target_cow_copy_blocks = 0
        num_draft_cow_copy_blocks = 0
        num_inter_cow_copy_blocks = 0

        # Pass 1: clones for B > 0. ``parent.token_ids`` still ends at ``recovery``
        # at this point so each clone inherits a clean ``[..., recovery]`` tape.
        t_branch_construct0 = perf_counter()
        for row_idx, (parent_idx, branch_idx, root_token) in enumerate(
            zip(parent_idx_list, branch_idx_list, root_token_list)
        ):
            if branch_idx == 0:
                continue
            num_nonzero_branches += 1
            parent = seqs[parent_idx]
            target_cached0 = parent.num_cached_tokens
            draft_cached0 = parent.num_draft_cached_tokens
            inter_cached0 = parent.num_inter_cached_tokens
            branch_seq = parent.clone_spec()
            required_tokens = parent.num_tokens + self.lookahead
            if fork_target_kv:
                t_plan = t_bm.make_cow_fork_block_table(
                    parent.block_table,
                    cached_tokens=target_cached0,
                    required_total_tokens=required_tokens,
                )
            else:
                # Phase-1: share parent's committed target prefix only (no target COW).
                n_share = min(
                    target_cached0 // parent.block_size,
                    len(parent.block_table),
                )
                fork_bt = t_bm.fork_shared_prefix(parent.block_table, n_share)
                t_plan = CowForkPlan(
                    fork_block_table=list(fork_bt),
                    private_tail_block_ids=[],
                    shared_prefix_blocks=int(n_share),
                    copy_src_block_ids=[],
                    copy_dst_block_ids=[],
                    copy_valid_tokens=[],
                )
            d_plan = d_bm.make_cow_fork_block_table(
                parent.draft_block_table,
                cached_tokens=draft_cached0,
                required_total_tokens=required_tokens,
            )
            i_plan: CowForkPlan | None = None
            if i_bm is not None and self.intermediate_runner is not None:
                i_plan = i_bm.make_cow_fork_block_table(
                    parent.inter_block_table,
                    cached_tokens=inter_cached0,
                    required_total_tokens=required_tokens,
                )

            branch_seq.block_table = list(t_plan.fork_block_table)
            branch_seq.draft_block_table = list(d_plan.fork_block_table)
            if i_plan is not None:
                branch_seq.inter_block_table = list(i_plan.fork_block_table)

            if t_plan.copy_src_block_ids:
                target_copy_src_all.extend(t_plan.copy_src_block_ids)
                target_copy_dst_all.extend(t_plan.copy_dst_block_ids)
                target_copy_valid_all.extend(t_plan.copy_valid_tokens)
                num_target_cow_copy_blocks += len(t_plan.copy_src_block_ids)
            if d_plan.copy_src_block_ids:
                draft_copy_src_all.extend(d_plan.copy_src_block_ids)
                draft_copy_dst_all.extend(d_plan.copy_dst_block_ids)
                draft_copy_valid_all.extend(d_plan.copy_valid_tokens)
                num_draft_cow_copy_blocks += len(d_plan.copy_src_block_ids)
            if i_plan is not None and i_plan.copy_src_block_ids:
                inter_copy_src_all.extend(i_plan.copy_src_block_ids)
                inter_copy_dst_all.extend(i_plan.copy_dst_block_ids)
                inter_copy_valid_all.extend(i_plan.copy_valid_tokens)
                num_inter_cow_copy_blocks += len(i_plan.copy_src_block_ids)
            branch_seq.append_token(int(root_token))
            expanded_seqs[row_idx] = branch_seq
            st = BranchForkState(
                parent_seq_idx=int(parent_idx),
                branch_idx=int(branch_idx),
                root_token_id=int(root_token),
                root_confidence=root_prob_list[row_idx],
                target_shared_prefix_blocks=t_plan.shared_prefix_blocks,
                draft_shared_prefix_blocks=d_plan.shared_prefix_blocks,
                inter_shared_prefix_blocks=(
                    i_plan.shared_prefix_blocks if i_plan is not None else 0
                ),
                draft_private_tail_block_ids=list(d_plan.private_tail_block_ids),
                target_private_tail_block_ids=list(t_plan.private_tail_block_ids),
                inter_private_tail_block_ids=(
                    list(i_plan.private_tail_block_ids) if i_plan is not None else []
                ),
                is_parent_inplace=False,
            )
            assert st.target_shared_prefix_blocks == target_cached0 // parent.block_size
            assert st.draft_shared_prefix_blocks == draft_cached0 // parent.block_size
            if i_plan is not None:
                assert st.inter_shared_prefix_blocks == inter_cached0 // parent.block_size
            branch_states[row_idx] = st
        t_branch_construct1 = perf_counter()

        self._cuda_sync_for_pivot_cost()
        t_cow_copy0 = perf_counter()
        if target_copy_src_all:
            # Must go through ``call`` so every target TP rank copies its shard.
            self.target_model_runner.call(
                "copy_kv_blocks",
                target_copy_src_all,
                target_copy_dst_all,
                target_copy_valid_all,
                "target",
            )
        if draft_copy_src_all:
            self._copy_partial_cow_block(
                copy_src_block_ids=draft_copy_src_all,
                copy_dst_block_ids=draft_copy_dst_all,
                copy_valid_tokens=draft_copy_valid_all,
                kv_cache=self.draft_model_runner.kv_cache,
            )
        if inter_copy_src_all and self.intermediate_runner is not None:
            self.intermediate_runner.call(
                "copy_kv_blocks",
                inter_copy_src_all,
                inter_copy_dst_all,
                inter_copy_valid_all,
                "intermediate",
            )
        self._cuda_sync_for_pivot_cost()
        t_cow_copy1 = perf_counter()

        # Pass 2: branch 0 = parent in-place. Mutating ``parent.token_ids`` here is
        # safe — Pass 1 finished before this, and the outer step rolls the tape
        # back after verify.
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
                # Shared/private fields are unused for parent-in-place branches but
                # populated with conservative values so accidental release is a no-op.
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
        # Build speculations on host (numpy) to avoid B_exp x lookahead H2D
        # scalar copies; upload once at the end.
        spec_np = np.empty((b_exp, self.lookahead + 1), dtype=np.int64)
        if b_exp > 0:
            spec_np[:, 0] = [recovery_tokens[parent_idx_list[i]] for i in range(b_exp)]
            spec_np[:, 1] = root_token_list

        # K logits rows: first row copied from parent (GPU gather), subsequent
        # rows from expanded rollout. Keep logits gather on GPU.
        logits_q_rows = [first_logits_q[plan.parent_index_per_branch]]
        t_initial_pack1 = perf_counter()

        t_tail_draft0 = perf_counter()
        if self.lookahead >= 2:
            for k in range(1, self.lookahead):
                token_ids_k, logits_k = self.draft_model_runner.call(
                    "run", expanded_seqs, False, True, True
                )
                for s in expanded_seqs:
                    s.num_draft_cached_tokens += 1
                logits_q_rows.append(logits_k)
                for row_idx, (seq, tok) in enumerate(zip(expanded_seqs, token_ids_k)):
                    seq.append_token(tok)
                    spec_np[row_idx, k + 1] = tok
        t_tail_draft1 = perf_counter()

        # Extra draft forward to write the LAST speculative token's draft KV.
        # ``SpeculatorSync`` runs ``lookahead + 1`` draft forwards (last logits
        # are dropped) for exactly this reason; without this step ``parent`` /
        # branch draft KV stops one token short of ``num_tokens`` and the next
        # decode would see ``num_draft_cached_tokens`` ahead of the actual KV
        # frontier after we commit the winner branch.
        t_extra_draft0 = perf_counter()
        if b_exp > 0:
            self.draft_model_runner.call(
                "run", expanded_seqs, False, True, True
            )
            for s in expanded_seqs:
                s.num_draft_cached_tokens += 1
        t_extra_draft1 = perf_counter()

        t_final_pack0 = perf_counter()
        logits_q = torch.stack(logits_q_rows, dim=1)
        # Single H2D upload of the fully-built host buffer.
        speculations = torch.from_numpy(spec_np).to(self.device, non_blocking=True)
        # Bundle reuses the already-materialized host plan; no further .tolist().
        assert plan.host is not None
        branch_bundle = PivotBranchBundle(
            parent_batch_size=batch_size,
            host_plan=plan.host,
            expanded_seqs=expanded_seqs,
            branch_states=branch_states,
        )
        t_final_pack1 = perf_counter()
        branch_construct_s = t_branch_construct1 - t_branch_construct0
        cow_copy_s = t_cow_copy1 - t_cow_copy0
        branch0_setup_s = t_branch0_setup1 - t_branch0_setup0
        initial_pack_s = t_initial_pack1 - t_initial_pack0
        tail_draft_s = t_tail_draft1 - t_tail_draft0
        extra_draft_s = t_extra_draft1 - t_extra_draft0
        final_pack_s = t_final_pack1 - t_final_pack0
        self._write_pivot_cost_row(
            {
                "kind": "pivot_draft_microcost",
                "step_id": int(self._pivot_cost_step_id),
                "parent_batch_size": int(batch_size),
                "expanded_batch_size": int(b_exp),
                "expansion_ratio": float(b_exp / batch_size) if batch_size > 0 else 0.0,
                "num_nonzero_branches": int(num_nonzero_branches),
                "num_target_cow_copy_blocks": int(num_target_cow_copy_blocks),
                "num_draft_cow_copy_blocks": int(num_draft_cow_copy_blocks),
                "num_inter_cow_copy_blocks": int(num_inter_cow_copy_blocks),
                "pivot_plan_build_s": float(t_plan1 - t_plan0),
                "pivot_capacity_clamp_s": float(t_cap1 - t_cap0),
                "pivot_branch0_override_s": float(t_cap2 - t_cap1),
                "pivot_branch_construct_s": float(branch_construct_s),
                "pivot_cow_copy_s": float(cow_copy_s),
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
            speculations=speculations,
            logits_q=logits_q,
            cache_hits=None,
            branch_bundle=branch_bundle,
        )
