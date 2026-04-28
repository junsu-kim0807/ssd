from __future__ import annotations

from ssd.engine.helpers.pivot_tree_helpers import (
    build_target_scratch_packed_inputs,
    can_use_target_scratch_phase1a,
)
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_speculator_sync import PivotRootSpeculatorSync
from ssd.engine.pivot_types import PivotTreeNode, PivotTreeScratchBundle


class PivotTreeScratchSpeculator(PivotRootSpeculatorSync):
    """Phase-0/1 bridge: reuse pivot root planning while exposing tree bundle API.

    This class intentionally keeps draft branch rollout from ``PivotRootSpeculatorSync``
    for early phases while migrating verifier/postprocess contracts to tree-oriented
    bundle types.

    Phase-1A (``enforce_eager`` + block-aligned ``num_cached_tokens``): skips target COW
    on non-zero branches; target verify KV is packed into scratch slots (see
    ``build_target_scratch_packed_inputs``).
    """

    def prefill(self, seqs, verify_result: VerifyResult) -> SpeculateResult:
        return super().prefill(seqs, verify_result)

    def _phase1a_target_scratch_eligible(self, seqs) -> bool:
        if not seqs:
            return False
        if not bool(getattr(self.target_model_runner, "enforce_eager", False)):
            return False
        return all(can_use_target_scratch_phase1a(s) for s in seqs)

    def speculate(
        self,
        seqs,
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
    ) -> SpeculateResult:
        phase1a = self._phase1a_target_scratch_eligible(seqs)
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
        if phase1a and b_exp > 0 and bundle.expanded_seqs is not None:
            target_packed = build_target_scratch_packed_inputs(
                bundle.expanded_seqs,
                path_token_ids,
                path_node_ids,
                self.scheduler.block_manager,
                block_size=int(seqs[0].block_size),
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
        )
        assert out.speculations.shape == (b_exp, k1)
        assert out.logits_q.shape[:2] == (b_exp, max(0, k1 - 1))
        return out
