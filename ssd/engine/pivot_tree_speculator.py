from __future__ import annotations

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult
from ssd.engine.pivot_speculator_sync import PivotRootSpeculatorSync
from ssd.engine.pivot_types import PivotTreeNode, PivotTreeScratchBundle


class PivotTreeScratchSpeculator(PivotRootSpeculatorSync):
    """Phase-0/1 bridge: reuse pivot root planning while exposing tree bundle API.

    This class intentionally keeps draft branch rollout from ``PivotRootSpeculatorSync``
    for early phases while migrating verifier/postprocess contracts to tree-oriented
    bundle types.
    """

    def prefill(self, seqs, verify_result: VerifyResult) -> SpeculateResult:
        return super().prefill(seqs, verify_result)

    def speculate(
        self,
        seqs,
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
    ) -> SpeculateResult:
        out = super().speculate(
            seqs,
            verify_result,
            recovery_already_appended=recovery_already_appended,
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
            target_node_to_slot={},
            draft_node_to_slot={},
            expanded_seqs=bundle.expanded_seqs,
            branch_states=bundle.branch_states,
            scratch_owner=None,
        )
        assert out.speculations.shape == (b_exp, k1)
        assert out.logits_q.shape[:2] == (b_exp, max(0, k1 - 1))
        return out
