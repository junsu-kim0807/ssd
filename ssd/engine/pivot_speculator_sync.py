from __future__ import annotations

import json
import torch

from ssd.engine.block_manager import BlockManager, CowForkPlan
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_branch_planner import (
    PivotExpansionConfig,
    PivotExpansionPlan,
    apply_capacity_limit,
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
    ):
        super().__init__(lookahead, device, draft_model_runner)
        self.target_model_runner = target_model_runner
        self.intermediate_runner = intermediate_runner
        self.scheduler = scheduler
        self.expansion_cfg = expansion_cfg
        self.max_expand_rows = max_expand_rows
        self._cached_root_tokens_per_parent: list[list[int]] | None = None

    def reset_step_state(self) -> None:
        self._cached_root_tokens_per_parent = None

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

    def _apply_kv_capacity_limit(self, seqs: list[Sequence], plan: PivotExpansionPlan) -> PivotExpansionPlan:
        """Clamp expansion by BM free blocks and max_model_len in addition to row cap."""
        bsz = len(seqs)
        if bsz == 0:
            return plan
        topk = max(plan.branch_counts) if plan.branch_counts else 1
        expand_mask = plan.expand_mask.clone()

        # First, enforce model length at request-level.
        for i, seq in enumerate(seqs):
            if seq.num_tokens + self.lookahead > self.scheduler.max_model_len:
                expand_mask[i] = False

        # Then clamp by aggregate free-block budgets (target/draft/intermediate).
        t_bm: BlockManager = self.scheduler.block_manager
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager

        free_t = len(t_bm.free_block_ids)
        free_d = len(d_bm.free_block_ids)
        free_i = len(i_bm.free_block_ids) if i_bm is not None else 10**12

        # Build deterministic keep order by low uncertainty score first.
        scores = plan.criteria_scores
        order = sorted(range(bsz), key=lambda x: (float(scores[x].item()), x))
        kept = torch.zeros_like(expand_mask)

        for idx in order:
            if not bool(expand_mask[idx].item()):
                continue
            seq = seqs[idx]
            required_tokens = seq.num_tokens + self.lookahead

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
            # Flat pivot currently uses the same required token horizon across roles.
            # Hierarchical pivot may require a role-specific frontier in future paths.
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
            need_t = per_extra_t * extras
            need_d = per_extra_d * extras
            need_i = per_extra_i * extras
            if need_t <= free_t and need_d <= free_d and need_i <= free_i:
                kept[idx] = True
                free_t -= need_t
                free_d -= need_d
                free_i -= need_i

        clamped = apply_capacity_limit(
            kept,
            criteria_scores=plan.criteria_scores,
            topk=topk,
            max_expand_rows=self.max_expand_rows,
        )
        if torch.equal(clamped, plan.expand_mask):
            return plan

        # Rebuild plan tensors from clamped mask while preserving top-k candidates.
        root_ids = plan.root_token_ids
        root_probs = plan.root_token_probs
        # Recompute compact view using original per-parent candidate arrays.
        per_parent_roots: list[list[tuple[int, float]]] = [[] for _ in range(bsz)]
        for pidx, bidx, rid, rp in zip(
            plan.parent_index_per_branch.tolist(),
            plan.branch_index_per_parent.tolist(),
            root_ids.tolist(),
            root_probs.tolist(),
        ):
            while len(per_parent_roots[pidx]) <= bidx:
                per_parent_roots[pidx].append((rid, rp))
            per_parent_roots[pidx][bidx] = (rid, rp)

        pib: list[int] = []
        bip: list[int] = []
        rid_out: list[int] = []
        rp_out: list[float] = []
        bcounts: list[int] = []
        for pidx in range(bsz):
            count = len(per_parent_roots[pidx]) if bool(clamped[pidx].item()) else 1
            bcounts.append(count)
            for bidx in range(count):
                pib.append(pidx)
                bip.append(bidx)
                rid, rp = per_parent_roots[pidx][bidx]
                rid_out.append(int(rid))
                rp_out.append(float(rp))

        return PivotExpansionPlan(
            parent_batch_size=bsz,
            expanded_batch_size=len(pib),
            expand_mask=clamped,
            parent_index_per_branch=torch.tensor(pib, dtype=torch.int64, device=self.device),
            branch_index_per_parent=torch.tensor(bip, dtype=torch.int64, device=self.device),
            root_token_ids=torch.tensor(rid_out, dtype=torch.int64, device=self.device),
            root_token_probs=torch.tensor(rp_out, dtype=torch.float32, device=self.device),
            criteria_scores=plan.criteria_scores,
            top1_probs=plan.top1_probs,
            residual_scores=plan.residual_scores,
            branch_counts=bcounts,
        )

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

        if self._cached_root_tokens_per_parent is None:
            plan = build_pivot_expansion_plan(
                first_logits_q,
                self.expansion_cfg,
                max_expand_rows=self.max_expand_rows,
            )
            plan = self._apply_kv_capacity_limit(seqs, plan)
            roots_per_parent: list[list[int]] = [[] for _ in range(batch_size)]
            for pidx, bidx, tok in zip(
                plan.parent_index_per_branch.tolist(),
                plan.branch_index_per_parent.tolist(),
                plan.root_token_ids.tolist(),
            ):
                while len(roots_per_parent[pidx]) <= bidx:
                    roots_per_parent[pidx].append(tok)
                roots_per_parent[pidx][bidx] = tok
            self._cached_root_tokens_per_parent = roots_per_parent
        else:
            parent_index_per_branch: list[int] = []
            branch_index_per_parent: list[int] = []
            root_token_ids: list[int] = []
            root_token_probs: list[float] = []
            branch_counts: list[int] = []
            probs = torch.softmax(first_logits_q.float(), dim=-1)
            for pidx, roots in enumerate(self._cached_root_tokens_per_parent):
                branch_counts.append(len(roots))
                for bidx, tok in enumerate(roots):
                    parent_index_per_branch.append(pidx)
                    branch_index_per_parent.append(bidx)
                    root_token_ids.append(int(tok))
                    root_token_probs.append(float(probs[pidx, int(tok)].item()))
            from ssd.engine.pivot_branch_planner import PivotExpansionPlan

            plan = PivotExpansionPlan(
                parent_batch_size=batch_size,
                expanded_batch_size=len(parent_index_per_branch),
                expand_mask=torch.tensor(
                    [len(r) > 1 for r in self._cached_root_tokens_per_parent],
                    dtype=torch.bool,
                    device=self.device,
                ),
                parent_index_per_branch=torch.tensor(parent_index_per_branch, dtype=torch.int64, device=self.device),
                branch_index_per_parent=torch.tensor(branch_index_per_parent, dtype=torch.int64, device=self.device),
                root_token_ids=torch.tensor(root_token_ids, dtype=torch.int64, device=self.device),
                root_token_probs=torch.tensor(root_token_probs, dtype=torch.float32, device=self.device),
                criteria_scores=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                top1_probs=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                residual_scores=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
                branch_counts=branch_counts,
            )
            plan = self._apply_kv_capacity_limit(seqs, plan)

        self._debug_pivot_expansion(first_logits_q=first_logits_q, plan=plan)

        # Two-pass branch construction:
        #   Pass 1: clone every B>0 branch BEFORE the parent gets its root appended.
        #           Otherwise ``parent.clone_spec()`` would copy a tape that already
        #           contains branch 0's root token.
        #   Pass 2: take ``branch 0`` in-place on the parent (no fork) so its draft/
        #           target KV is written directly into the parent's block table —
        #           which is what ``scheduler.may_append`` already pre-allocated for.
        parent_idx_list = [int(x) for x in plan.parent_index_per_branch.tolist()]
        branch_idx_list = [int(x) for x in plan.branch_index_per_parent.tolist()]
        root_token_list = [int(x) for x in plan.root_token_ids.tolist()]
        root_prob_list = [float(x) for x in plan.root_token_probs.tolist()]
        b_exp = len(parent_idx_list)
        expanded_seqs: list[Sequence] = [None] * b_exp  # type: ignore[list-item]
        branch_states: list[BranchForkState] = [None] * b_exp  # type: ignore[list-item]

        t_bm: BlockManager = self.scheduler.block_manager
        d_bm: BlockManager = self.scheduler.draft_block_manager
        i_bm: BlockManager | None = self.scheduler.intermediate_block_manager

        # Pass 1: clones for B > 0. ``parent.token_ids`` still ends at ``recovery``
        # at this point so each clone inherits a clean ``[..., recovery]`` tape.
        for row_idx, (parent_idx, branch_idx, root_token) in enumerate(
            zip(parent_idx_list, branch_idx_list, root_token_list)
        ):
            if branch_idx == 0:
                continue
            parent = seqs[parent_idx]
            target_cached0 = parent.num_cached_tokens
            draft_cached0 = parent.num_draft_cached_tokens
            inter_cached0 = parent.num_inter_cached_tokens
            branch_seq = parent.clone_spec()
            required_tokens = parent.num_tokens + self.lookahead
            t_plan = t_bm.make_cow_fork_block_table(
                parent.block_table,
                cached_tokens=target_cached0,
                required_total_tokens=required_tokens,
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
                # Must go through ``call`` so every target TP rank copies its shard.
                self.target_model_runner.call(
                    "copy_kv_blocks",
                    t_plan.copy_src_block_ids,
                    t_plan.copy_dst_block_ids,
                    t_plan.copy_valid_tokens,
                    "target",
                )
            if d_plan.copy_src_block_ids:
                self._copy_partial_cow_block(
                    copy_src_block_ids=d_plan.copy_src_block_ids,
                    copy_dst_block_ids=d_plan.copy_dst_block_ids,
                    copy_valid_tokens=d_plan.copy_valid_tokens,
                    kv_cache=self.draft_model_runner.kv_cache,
                )
            if i_plan is not None and i_plan.copy_src_block_ids:
                self.intermediate_runner.call(
                    "copy_kv_blocks",
                    i_plan.copy_src_block_ids,
                    i_plan.copy_dst_block_ids,
                    i_plan.copy_valid_tokens,
                    "intermediate",
                )
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

        # Pass 2: branch 0 = parent in-place. Mutating ``parent.token_ids`` here is
        # safe — Pass 1 finished before this, and the outer step rolls the tape
        # back after verify.
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

        speculations = torch.zeros(
            b_exp, self.lookahead + 1, dtype=torch.int64, device=self.device
        )
        speculations[:, 0] = torch.tensor(
            [recovery_tokens[parent_idx_list[i]] for i in range(b_exp)],
            dtype=torch.int64,
            device=self.device,
        )
        speculations[:, 1] = plan.root_token_ids.to(torch.int64)

        # K logits rows: first row copied from parent, subsequent rows from expanded rollout.
        logits_q_rows = []
        logits_q_rows.append(first_logits_q[plan.parent_index_per_branch])

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
                    speculations[row_idx, k + 1] = int(tok)

        # Extra draft forward to write the LAST speculative token's draft KV.
        # ``SpeculatorSync`` runs ``lookahead + 1`` draft forwards (last logits
        # are dropped) for exactly this reason; without this step ``parent`` /
        # branch draft KV stops one token short of ``num_tokens`` and the next
        # decode would see ``num_draft_cached_tokens`` ahead of the actual KV
        # frontier after we commit the winner branch.
        if b_exp > 0:
            self.draft_model_runner.call(
                "run", expanded_seqs, False, True, True
            )
            for s in expanded_seqs:
                s.num_draft_cached_tokens += 1

        logits_q = torch.stack(logits_q_rows, dim=1)
        branch_bundle = PivotBranchBundle(
            parent_batch_size=batch_size,
            parent_index_per_branch=[int(x) for x in plan.parent_index_per_branch.tolist()],
            branch_index_per_parent=[int(x) for x in plan.branch_index_per_parent.tolist()],
            branch_counts=list(plan.branch_counts),
            root_token_ids=[int(x) for x in plan.root_token_ids.tolist()],
            root_token_probs=[float(x) for x in plan.root_token_probs.tolist()],
            criteria_scores=[float(x) for x in plan.criteria_scores.tolist()],
            top1_probs=[float(x) for x in plan.top1_probs.tolist()],
            residual_scores=[float(x) for x in plan.residual_scores.tolist()],
            expanded_seqs=expanded_seqs,
            branch_states=branch_states,
        )
        return SpeculateResult(
            speculations=speculations,
            logits_q=logits_q,
            cache_hits=None,
            branch_bundle=branch_bundle,
        )
