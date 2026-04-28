from __future__ import annotations
import math
from time import perf_counter

import torch

from ssd.engine.helpers.pivot_tree_helpers import (
    DraftScratchPackedInputs,
    build_rowwise_prefix_candidate_mask,
    build_target_scratch_packed_inputs_from_paths,
    build_target_scratch_packed_inputs,
    can_use_draft_scratch_phase2a,
    can_use_target_scratch_phase1a,
)
from ssd.engine.pivot_branch_planner import (
    PivotExpansionConfig,
    PivotExpansionPlan,
    PivotHostPlan,
    build_pivot_expansion_plan,
)
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_speculator_sync import PivotRootSpeculatorSync
from ssd.engine.pivot_types import PivotTreeNode, PivotTreeScratchBundle, ScratchOwner


class PivotTreeScratchSpeculator(PivotRootSpeculatorSync):
    """Pivot root expansion with optional target/draft scratch KV paths.

    Phase 0: flat fallback with tree-shaped metadata.
    Phase 1A: flat draft rollout + target scratch verify.
    Phase 2A: draft scratch rollout + target scratch verify, no branch Sequence/COW.
    """

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
        metrics: dict | None = None,
    ):
        super().__init__(
            lookahead=lookahead,
            device=device,
            draft_model_runner=draft_model_runner,
            target_model_runner=target_model_runner,
            intermediate_runner=intermediate_runner,
            scheduler=scheduler,
            expansion_cfg=expansion_cfg,
            max_expand_rows=max_expand_rows,
            enable_profile_trace=enable_profile_trace,
        )
        self.metrics = metrics if metrics is not None else {}

    def prefill(self, seqs, verify_result: VerifyResult) -> SpeculateResult:
        return super().prefill(seqs, verify_result)

    def _phase1a_target_scratch_eligible(self, seqs) -> bool:
        if not seqs:
            return False
        if not bool(getattr(self.target_model_runner, "enforce_eager", False)):
            return False
        return all(can_use_target_scratch_phase1a(s) for s in seqs)

    def _phase2_draft_scratch_enabled(self, seqs) -> bool:
        cfg = getattr(self.scheduler, "config", None)
        if not bool(getattr(cfg, "enable_pivot_draft_scratch_phase2", False)):
            return False
        if not seqs:
            return False
        if not bool(getattr(self.target_model_runner, "enforce_eager", False)):
            return False
        if not bool(getattr(self.draft_model_runner, "enforce_eager", False)):
            return False
        if bool(getattr(cfg, "use_eagle", False)):
            return False
        if getattr(cfg, "spec_policy", "") != "pivot_tree_scratch":
            return False
        for seq in seqs:
            tgt_temp = float(getattr(seq, "temperature", 0.0))
            dr_temp = (
                seq.draft_temperature
                if seq.draft_temperature is not None
                else seq.temperature
            )
            if tgt_temp != 0.0 or float(dr_temp) != 0.0:
                return False
        if self._phase2_scratch_max_expand_rows(seqs) < len(seqs):
            return False
        return all(can_use_target_scratch_phase1a(s) for s in seqs) and all(
            can_use_draft_scratch_phase2a(s) for s in seqs
        )

    def _sample_phase2_draft_tokens_greedy(self, logits: torch.Tensor) -> list[int]:
        return torch.argmax(logits, dim=-1).detach().cpu().tolist()

    def _phase2_scratch_max_expand_rows(self, seqs) -> int:
        block_size = int(seqs[0].block_size)
        k1 = int(self.lookahead) + 1
        nscratch = (k1 + block_size - 1) // block_size
        free_t = len(self.scheduler.block_manager.free_block_ids)
        free_d = len(self.scheduler.draft_block_manager.free_block_ids)
        max_rows_by_target = free_t // nscratch
        max_rows_by_draft = free_d // nscratch
        return int(min(max_rows_by_target, max_rows_by_draft))

    def _apply_phase2_scratch_capacity_limit(
        self,
        seqs,
        plan: PivotExpansionPlan,
    ) -> PivotExpansionPlan:
        if plan.expanded_batch_size == 0:
            return plan
        max_rows = self._phase2_scratch_max_expand_rows(seqs)
        if plan.expanded_batch_size <= max_rows:
            return plan
        if max_rows <= 0:
            max_rows = len(seqs)
        topk = max(1, int(self.expansion_cfg.topk))
        if topk <= 1:
            return plan
        max_expand_reqs = max(0, (int(max_rows) - len(seqs)) // (topk - 1))
        assert plan.host is not None
        host = plan.host
        criteria = (
            host.criteria_scores
            if host.criteria_scores is not None
            else [float(x) for x in plan.criteria_scores.detach().cpu().tolist()]
        )
        expandable = [i for i, c in enumerate(host.branch_counts) if int(c) > 1]
        expandable.sort(key=lambda i: (float(criteria[i]), int(i)))
        keep_expanded_set = set(expandable[:max_expand_reqs])
        new_branch_counts = [
            topk if i in keep_expanded_set else 1 for i in range(len(host.branch_counts))
        ]
        pids: list[int] = []
        bids: list[int] = []
        for pidx, cnt in enumerate(new_branch_counts):
            for bidx in range(int(cnt)):
                pids.append(int(pidx))
                bids.append(int(bidx))
        device = plan.parent_index_per_branch.device
        cursor = 0
        root_by_parent: list[list[int]] = []
        prob_by_parent: list[list[float]] = []
        root_probs_host = (
            [float(x) for x in plan.root_token_probs.detach().cpu().tolist()]
            if plan.root_token_probs.numel() == int(plan.expanded_batch_size)
            else [0.0] * int(plan.expanded_batch_size)
        )
        for c in host.branch_counts:
            cnt = int(c)
            root_by_parent.append([int(x) for x in host.root_token_ids[cursor : cursor + cnt]])
            prob_by_parent.append([float(x) for x in root_probs_host[cursor : cursor + cnt]])
            cursor += cnt
        root_ids_t = torch.tensor(
            [int(root_by_parent[p][b]) for p, b in zip(pids, bids)],
            dtype=torch.int64,
            device=device,
        )
        root_probs_t = torch.tensor(
            [float(prob_by_parent[p][b]) for p, b in zip(pids, bids)],
            dtype=torch.float32,
            device=device,
        )
        expand_mask_host = [c > 1 for c in new_branch_counts]
        new_host = PivotHostPlan(
            parent_index_per_branch=list(pids),
            branch_index_per_parent=list(bids),
            root_token_ids=[int(x) for x in root_ids_t.detach().cpu().tolist()],
            branch_counts=[int(x) for x in new_branch_counts],
            expand_mask=[bool(x) for x in expand_mask_host],
            criteria_scores=list(criteria),
            root_token_probs=(
                [float(x) for x in root_probs_t.detach().cpu().tolist()]
                if self.enable_profile_trace
                else None
            ),
            top1_probs=host.top1_probs,
            residual_scores=host.residual_scores,
        )
        return PivotExpansionPlan(
            parent_batch_size=int(plan.parent_batch_size),
            expanded_batch_size=len(pids),
            expand_mask=torch.tensor(expand_mask_host, dtype=torch.bool, device=device),
            parent_index_per_branch=torch.tensor(pids, dtype=torch.int64, device=device),
            branch_index_per_parent=torch.tensor(bids, dtype=torch.int64, device=device),
            root_token_ids=root_ids_t,
            root_token_probs=root_probs_t,
            criteria_scores=plan.criteria_scores,
            top1_probs=plan.top1_probs,
            residual_scores=plan.residual_scores,
            branch_counts=[int(x) for x in new_branch_counts],
            branch_counts_tensor=torch.tensor(new_branch_counts, dtype=torch.int64, device=device),
            host=new_host,
        )

    def _run_phase2_parent_recovery_logits(
        self,
        seqs,
        recovery_tokens: list[int],
    ) -> torch.Tensor:
        bsz = len(seqs)
        block_size = int(seqs[0].block_size)
        bm = self.scheduler.draft_block_manager
        input_ids = torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device)
        positions = torch.empty((bsz,), dtype=torch.int64, device=self.device)
        slot_mapping = torch.empty((bsz,), dtype=torch.int64, device=self.device)
        context_lens = torch.empty((bsz,), dtype=torch.int32, device=self.device)
        block_tables_host: list[list[int]] = []
        scratch_blocks_all: list[int] = []
        for i, seq in enumerate(seqs):
            pos0 = int(seq.num_tokens)
            assert int(seq.num_draft_cached_tokens) == pos0
            n_pref = pos0 // block_size
            prefix = [int(x) for x in seq.draft_block_table[:n_pref]]
            sb = bm.allocate_scratch_blocks(1)
            scratch_blocks_all.extend(sb)
            row_bt = prefix + sb
            block_tables_host.append(row_bt)
            positions[i] = pos0
            context_lens[i] = pos0 + 1
            bidx = pos0 // block_size
            off = pos0 % block_size
            slot_mapping[i] = int(row_bt[bidx]) * block_size + off
        max_blocks = max(len(r) for r in block_tables_host)
        block_tables = torch.full((bsz, max_blocks), -1, dtype=torch.int32, device=self.device)
        for i, row in enumerate(block_tables_host):
            block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=self.device)
        cu = torch.arange(0, bsz + 1, dtype=torch.int32, device=self.device)
        pos0s = [int(s.num_tokens) for s in seqs]
        mask = build_rowwise_prefix_candidate_mask(pos0s, 1, device=self.device)
        try:
            logits = self.draft_model_runner.call(
                "run_packed_tree_decode",
                input_ids,
                positions,
                slot_mapping,
                context_lens,
                block_tables,
                cu,
                1,
                mask,
            )
        finally:
            ScratchOwner(target_block_ids=[], draft_block_ids=scratch_blocks_all).release_unreleased(
                self.scheduler.block_manager,
                self.scheduler.draft_block_manager,
            )
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        return logits

    def _build_tree_bundle_from_super_out(self, out, *, enable_target_scratch: bool):
        bundle = out.branch_bundle
        if bundle is None:
            return out
        spec_rows = out.speculations.detach().cpu().tolist()
        b_exp = len(spec_rows)
        k1 = len(spec_rows[0]) if b_exp else 0
        path_node_ids: list[list[int]] = []
        path_token_ids: list[list[int]] = []
        nodes: list[PivotTreeNode] = []
        nid = 0
        for r, row in enumerate(spec_rows):
            cur_ids: list[int] = []
            parent_node_id = None
            for depth, tok in enumerate(row):
                cur_ids.append(nid)
                parent_idx = bundle.parent_index_per_branch[r]
                branch_idx = bundle.branch_index_per_parent[r]
                nodes.append(
                    PivotTreeNode(
                        node_id=nid,
                        parent_seq_idx=int(parent_idx),
                        parent_node_id=parent_node_id,
                        branch_idx=int(branch_idx),
                        depth=int(depth),
                        token_id=int(tok),
                        position=int(depth),
                    )
                )
                parent_node_id = nid
                nid += 1
            path_node_ids.append(cur_ids)
            path_token_ids.append([int(t) for t in row])

        tgt_slot: dict[int, tuple[int, int]] = {}
        scratch_owner = None
        target_packed = None
        if enable_target_scratch and bundle.expanded_seqs is not None:
            target_packed = build_target_scratch_packed_inputs(
                bundle.expanded_seqs,
                path_token_ids,
                path_node_ids,
                self.scheduler.block_manager,
                block_size=int(bundle.expanded_seqs[0].block_size),
                device=self.device,
                lookahead=int(self.lookahead),
            )
            tgt_slot = target_packed.target_node_to_slot
            scratch_owner = target_packed.scratch_owner

        out.branch_bundle = PivotTreeScratchBundle(
            parent_batch_size=bundle.parent_batch_size,
            expanded_batch_size=b_exp,
            host_plan=bundle.host_plan,
            nodes=nodes,
            path_node_ids=path_node_ids,
            path_token_ids=path_token_ids,
            path_parent_seq_idx=[int(x) for x in bundle.parent_index_per_branch],
            path_branch_idx=[int(x) for x in bundle.branch_index_per_parent],
            logits_q=out.logits_q,
            target_node_to_slot=tgt_slot,
            draft_node_to_slot={},
            expanded_seqs=bundle.expanded_seqs,
            branch_states=bundle.branch_states,
            scratch_owner=scratch_owner,
            target_scratch_packed=target_packed,
            draft_scratch_packed=None,
        )
        assert out.speculations.shape == (b_exp, k1)
        assert out.logits_q.shape[:2] == (b_exp, max(0, k1 - 1))
        return out

    def _speculate_phase2a_draft_scratch(
        self,
        seqs,
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
    ) -> SpeculateResult:
        assert not recovery_already_appended, (
            "Phase-2A draft scratch currently assumes recovery is not appended to parent"
        )
        recovery_token_ids = [int(seq.recovery_token_id) for seq in seqs]
        t_recovery0 = perf_counter()
        first_logits_q = self._run_phase2_parent_recovery_logits(seqs, recovery_token_ids)
        t_recovery1 = perf_counter()
        first_token_ids = self._sample_phase2_draft_tokens_greedy(first_logits_q)
        scratch_cap_rows = self._phase2_scratch_max_expand_rows(seqs)
        phase2_max_expand_rows = (
            scratch_cap_rows
            if self.max_expand_rows is None
            else min(int(self.max_expand_rows), scratch_cap_rows)
        )
        plan = build_pivot_expansion_plan(
            first_logits_q,
            self.expansion_cfg,
            max_expand_rows=phase2_max_expand_rows,
            materialize_host=True,
            profile_metadata=self.enable_profile_trace,
        )
        self._override_branch0_roots(plan, first_token_ids, first_logits_q)
        plan = self._apply_kv_capacity_limit(seqs, plan, fork_target_kv=False)
        plan = self._apply_phase2_scratch_capacity_limit(seqs, plan)
        assert plan.host is not None
        host_plan = plan.host
        b_exp = int(plan.expanded_batch_size)
        k1 = int(self.lookahead) + 1
        block_size = int(seqs[0].block_size)
        nscratch = int(math.ceil(k1 / block_size))
        needed_blocks = b_exp * nscratch
        assert needed_blocks <= len(self.scheduler.block_manager.free_block_ids), (
            f"insufficient target scratch capacity: need={needed_blocks}, "
            f"free={len(self.scheduler.block_manager.free_block_ids)}"
        )
        assert needed_blocks <= len(self.scheduler.draft_block_manager.free_block_ids), (
            f"insufficient draft scratch capacity: need={needed_blocks}, "
            f"free={len(self.scheduler.draft_block_manager.free_block_ids)}"
        )

        path_parent_seq_idx = [int(x) for x in host_plan.parent_index_per_branch]
        path_branch_idx = [int(x) for x in host_plan.branch_index_per_parent]
        root_token_ids = [int(x) for x in host_plan.root_token_ids]
        path_token_ids = [[-1] * k1 for _ in range(b_exp)]
        path_node_ids = [[r * k1 + d for d in range(k1)] for r in range(b_exp)]
        for r in range(b_exp):
            pidx = int(path_parent_seq_idx[r])
            path_token_ids[r][0] = int(recovery_token_ids[pidx])
            if k1 > 1:
                path_token_ids[r][1] = int(root_token_ids[r])

        draft_bm = self.scheduler.draft_block_manager
        draft_node_to_slot: dict[int, tuple[int, int]] = {}
        draft_row_block_tables: list[list[int]] = []
        draft_pos0: list[int] = []
        draft_scratch_all: list[int] = []
        target_packed = None
        try:
            for r in range(b_exp):
                pidx = int(path_parent_seq_idx[r])
                seq = seqs[pidx]
                pos0 = int(seq.num_tokens)
                assert int(seq.num_draft_cached_tokens) == pos0
                assert pos0 % block_size == 0
                n_pref = pos0 // block_size
                prefix = [int(x) for x in seq.draft_block_table[:n_pref]]
                scratch = draft_bm.allocate_scratch_blocks(nscratch)
                draft_scratch_all.extend(scratch)
                row_bt = prefix + scratch
                draft_row_block_tables.append(row_bt)
                draft_pos0.append(pos0)
                for d in range(k1):
                    pos = pos0 + d
                    bidx = pos // block_size
                    off = pos % block_size
                    bid = int(row_bt[bidx])
                    draft_node_to_slot[int(path_node_ids[r][d])] = (bid, off)

            logits_q_rows: list[torch.Tensor] = []
            max_blocks = max(len(r) for r in draft_row_block_tables) if draft_row_block_tables else 0
            block_tables = torch.full((b_exp, max_blocks), -1, dtype=torch.int32, device=self.device)
            for i, row in enumerate(draft_row_block_tables):
                block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=self.device)
            cu = torch.arange(0, b_exp + 1, dtype=torch.int32, device=self.device)
            t_rollout0 = perf_counter()
            for depth in range(k1):
                input_ids = torch.tensor(
                    [int(path_token_ids[r][depth]) for r in range(b_exp)],
                    dtype=torch.int64,
                    device=self.device,
                )
                positions = torch.tensor(
                    [int(draft_pos0[r] + depth) for r in range(b_exp)],
                    dtype=torch.int64,
                    device=self.device,
                )
                context_lens = torch.tensor(
                    [int(draft_pos0[r] + depth + 1) for r in range(b_exp)],
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.tensor(
                    [
                        int(draft_node_to_slot[int(path_node_ids[r][depth])][0] * block_size)
                        + int(draft_node_to_slot[int(path_node_ids[r][depth])][1])
                        for r in range(b_exp)
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                mask = build_rowwise_prefix_candidate_mask(
                    [int(draft_pos0[r] + depth) for r in range(b_exp)], 1, device=self.device
                )
                q_lens = (cu[1:] - cu[:-1]).to(torch.int64)
                kv_lens = context_lens.to(torch.int64)
                expected_mask_len = int((q_lens * kv_lens).sum().item())
                actual_mask_len = int(mask.numel()) if mask is not None else -1
                q_total = int(q_lens.sum().item())
                assert int(input_ids.numel()) == q_total, (
                    f"phase2 draft input_ids mismatch: input_ids={int(input_ids.numel())}, "
                    f"q_total={q_total}"
                )
                assert int(positions.numel()) == q_total
                assert int(slot_mapping.numel()) == q_total
                assert actual_mask_len == expected_mask_len, (
                    "phase2 draft mask length mismatch: "
                    f"expected={expected_mask_len}, actual={actual_mask_len}, depth={depth}"
                )
                assert int(block_tables.shape[0]) == b_exp
                logits = self.draft_model_runner.call(
                    "run_packed_tree_decode",
                    input_ids,
                    positions,
                    slot_mapping,
                    context_lens,
                    block_tables,
                    cu,
                    1,
                    mask,
                )
                if logits.ndim == 2:
                    assert int(logits.shape[0]) == b_exp, (
                        "phase2 draft logits rows mismatch: "
                        f"got={int(logits.shape[0])}, expected={b_exp}"
                    )
                elif logits.ndim == 3:
                    assert tuple(logits.shape[:2]) == (b_exp, 1), (
                        "phase2 draft logits shape mismatch: "
                        f"got={list(logits.shape)}, expected=[{b_exp}, 1, V]"
                    )
                    logits = logits[:, 0, :]
                else:
                    raise AssertionError(
                        f"unexpected phase2 draft logits shape: {list(logits.shape)}"
                    )
                if depth < self.lookahead:
                    logits_q_rows.append(logits)
                if depth == 0:
                    # Root token is fixed by pivot plan.
                    # Only write recovery KV into branch scratch geometry.
                    continue
                if depth < self.lookahead:
                    sampled = self._sample_phase2_draft_tokens_greedy(logits)
                    for r in range(b_exp):
                        path_token_ids[r][depth + 1] = int(sampled[r])
            t_rollout1 = perf_counter()
            self.metrics.setdefault("phase2_recovery_plan_forward_s", []).append(
                float(t_recovery1 - t_recovery0)
            )
            self.metrics.setdefault("phase2_draft_rollout_forward_s", []).append(
                float(t_rollout1 - t_rollout0)
            )

            speculations = torch.tensor(path_token_ids, dtype=torch.int64, device=self.device)
            logits_q = (
                torch.stack(logits_q_rows, dim=1)
                if logits_q_rows
                else torch.empty((b_exp, 0, first_logits_q.shape[-1]), dtype=first_logits_q.dtype, device=self.device)
            )
            target_packed = build_target_scratch_packed_inputs_from_paths(
                seqs,
                path_parent_seq_idx,
                path_token_ids,
                path_node_ids,
                self.scheduler.block_manager,
                block_size=block_size,
                device=self.device,
                lookahead=int(self.lookahead),
            )
        except Exception:
            ScratchOwner(
                target_block_ids=(
                    list(target_packed.scratch_owner.target_block_ids)
                    if target_packed is not None
                    else []
                ),
                draft_block_ids=list(draft_scratch_all),
            ).release_unreleased(
                self.scheduler.block_manager,
                self.scheduler.draft_block_manager,
            )
            raise
        scratch_owner = target_packed.scratch_owner
        try:
            scratch_owner.draft_block_ids.extend(draft_scratch_all)
            draft_input_ids = torch.tensor(path_token_ids, dtype=torch.int64, device=self.device).reshape(-1)
            draft_positions = torch.tensor(
                [draft_pos0[r] + d for r in range(b_exp) for d in range(k1)],
                dtype=torch.int64,
                device=self.device,
            )
            draft_slot_mapping = torch.tensor(
                [
                    draft_node_to_slot[int(path_node_ids[r][d])][0] * block_size
                    + draft_node_to_slot[int(path_node_ids[r][d])][1]
                    for r in range(b_exp)
                    for d in range(k1)
                ],
                dtype=torch.int64,
                device=self.device,
            )
            draft_context_lens = torch.tensor(
                [draft_pos0[r] + k1 for r in range(b_exp)],
                dtype=torch.int32,
                device=self.device,
            )
            draft_mask = build_rowwise_prefix_candidate_mask(draft_pos0, k1, device=self.device)
            draft_packed = DraftScratchPackedInputs(
                input_ids=draft_input_ids,
                positions=draft_positions,
                slot_mapping=draft_slot_mapping,
                context_lens=draft_context_lens,
                block_tables=block_tables,
                cu_seqlens_q=torch.arange(0, b_exp + 1, dtype=torch.int32, device=self.device) * k1,
                max_seqlen_q=k1,
                tree_attn_mask=draft_mask,
                draft_node_to_slot=draft_node_to_slot,
                scratch_owner=scratch_owner,
            )
            nodes: list[PivotTreeNode] = []
            for r in range(b_exp):
                pidx = int(path_parent_seq_idx[r])
                br = int(path_branch_idx[r])
                parent_node_id = None
                for d in range(k1):
                    nid = int(path_node_ids[r][d])
                    nodes.append(
                        PivotTreeNode(
                            node_id=nid,
                            parent_seq_idx=pidx,
                            parent_node_id=parent_node_id,
                            branch_idx=br,
                            depth=d,
                            token_id=int(path_token_ids[r][d]),
                            position=d,
                        )
                    )
                    parent_node_id = nid
            bundle = PivotTreeScratchBundle(
                parent_batch_size=len(seqs),
                expanded_batch_size=b_exp,
                host_plan=host_plan,
                nodes=nodes,
                path_node_ids=path_node_ids,
                path_token_ids=path_token_ids,
                path_parent_seq_idx=path_parent_seq_idx,
                path_branch_idx=path_branch_idx,
                logits_q=logits_q,
                target_node_to_slot=target_packed.target_node_to_slot,
                draft_node_to_slot=draft_node_to_slot,
                expanded_seqs=None,
                branch_states=None,
                scratch_owner=scratch_owner,
                target_scratch_packed=target_packed,
                draft_scratch_packed=draft_packed,
            )
            assert bundle.expanded_seqs is None
            assert bundle.branch_states is None
            assert bundle.target_node_to_slot
            assert bundle.draft_node_to_slot
            assert bundle.target_scratch_packed is not None
            assert bundle.draft_scratch_packed is not None
            assert speculations.shape == (b_exp, k1)
            assert logits_q.shape[:2] == (b_exp, self.lookahead)
            assert all(-1 not in row for row in path_token_ids), path_token_ids
            return SpeculateResult(
                speculations=speculations,
                logits_q=logits_q,
                cache_hits=None,
                branch_bundle=bundle,
            )
        except Exception:
            scratch_owner.release_unreleased(
                self.scheduler.block_manager,
                self.scheduler.draft_block_manager,
            )
            raise

    def speculate(
        self,
        seqs,
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
    ) -> SpeculateResult:
        phase2a = self._phase2_draft_scratch_enabled(seqs)
        if phase2a:
            self.metrics.setdefault("pivot_tree_phase2a_steps", []).append(1)
            self.metrics.setdefault("pivot_tree_phase1a_steps", []).append(0)
            self.metrics.setdefault("pivot_tree_fallback_steps", []).append(0)
            return self._speculate_phase2a_draft_scratch(
                seqs,
                verify_result,
                recovery_already_appended=recovery_already_appended,
            )
        phase1a = self._phase1a_target_scratch_eligible(seqs)
        if phase1a:
            self.metrics.setdefault("pivot_tree_phase2a_steps", []).append(0)
            self.metrics.setdefault("pivot_tree_phase1a_steps", []).append(1)
            self.metrics.setdefault("pivot_tree_fallback_steps", []).append(0)
        else:
            self.metrics.setdefault("pivot_tree_phase2a_steps", []).append(0)
            self.metrics.setdefault("pivot_tree_phase1a_steps", []).append(0)
            self.metrics.setdefault("pivot_tree_fallback_steps", []).append(1)
        if phase1a and seqs:
            bs0 = int(seqs[0].block_size)
            k1 = int(self.lookahead) + 1
            nscratch = (k1 + bs0 - 1) // bs0
            bsz = len(seqs)
            topk = max(1, int(self.expansion_cfg.topk))
            cap = self.max_expand_rows
            worst_rows = bsz * topk if cap is None else min(int(cap), bsz * topk)
            free_scratch = len(self.scheduler.block_manager.free_block_ids)
            if worst_rows * nscratch > free_scratch:
                phase1a = False
        out = super().speculate(
            seqs,
            verify_result,
            recovery_already_appended=recovery_already_appended,
            fork_target_kv=not phase1a,
        )
        return self._build_tree_bundle_from_super_out(out, enable_target_scratch=phase1a)
