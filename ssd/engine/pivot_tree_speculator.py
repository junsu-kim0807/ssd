from __future__ import annotations

import json
import torch

from ssd.engine.helpers.pivot_tree_helpers import (
    build_draft_scratch_packed_inputs,
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

    def _phase2_draft_scratch_enabled(self, seqs) -> bool:
        """Draft scratch is gated off until depth-wise scratch rollout is implemented."""
        return False

    def _phase2_draft_scratch_shadow_enabled(self, seqs) -> bool:
        cfg = getattr(self.scheduler, "config", None)
        if not bool(getattr(cfg, "enable_pivot_draft_scratch_shadow", False)):
            return False
        if not seqs:
            return False
        return bool(getattr(self.draft_model_runner, "enforce_eager", False))

    def _debug_compare_draft_scratch_shadow(
        self,
        *,
        expanded_seqs,
        path_token_ids: list[list[int]],
        path_node_ids: list[list[int]],
        logits_q_ref: torch.Tensor,
    ) -> None:
        packed = build_draft_scratch_packed_inputs(
            expanded_seqs,
            path_token_ids,
            path_node_ids,
            self.scheduler.draft_block_manager,
            block_size=int(expanded_seqs[0].block_size),
            device=self.device,
            lookahead=int(self.lookahead),
        )
        try:
            logits_shadow = self.draft_model_runner.call(
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
            n_rows = len(path_token_ids)
            row_len = len(path_token_ids[0]) if n_rows else 0
            if logits_shadow.ndim == 2:
                logits_shadow = logits_shadow.view(n_rows, row_len, -1)
            elif logits_shadow.ndim == 3:
                assert tuple(logits_shadow.shape[:2]) == (n_rows, row_len), (
                    f"unexpected draft scratch shadow shape[:2]={tuple(logits_shadow.shape[:2])}, "
                    f"expected={(n_rows, row_len)}"
                )
            else:
                raise AssertionError(
                    f"unexpected draft scratch logits shape={list(logits_shadow.shape)}"
                )
            logits_q_shadow = logits_shadow[:, : self.lookahead, :]
            assert logits_q_shadow.shape == logits_q_ref.shape, (
                f"draft scratch shadow shape mismatch: "
                f"shadow={list(logits_q_shadow.shape)}, ref={list(logits_q_ref.shape)}"
            )
            diff = float((logits_q_shadow.float() - logits_q_ref.float()).abs().max().item())
            print(
                json.dumps(
                    {
                        "pivot_tree_phase2_draft_shadow_compare": True,
                        "max_abs_diff": diff,
                        "batch_expanded": int(n_rows),
                        "lookahead": int(self.lookahead),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if bool(
                getattr(
                    getattr(self.scheduler, "config", None),
                    "debug_pivot_draft_scratch_compare",
                    False,
                )
            ):
                assert diff < 1e-3, f"draft scratch shadow mismatch: max_abs_diff={diff}"
        finally:
            packed.scratch_owner.release_unreleased(
                self.scheduler.block_manager,
                self.scheduler.draft_block_manager,
            )

    def speculate(
        self,
        seqs,
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
    ) -> SpeculateResult:
        phase1a = self._phase1a_target_scratch_eligible(seqs)
        phase2a = self._phase2_draft_scratch_enabled(seqs)
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
        if (
            self._phase2_draft_scratch_shadow_enabled(seqs)
            and b_exp > 0
            and bundle.expanded_seqs is not None
        ):
            self._debug_compare_draft_scratch_shadow(
                expanded_seqs=bundle.expanded_seqs,
                path_token_ids=path_token_ids,
                path_node_ids=path_node_ids,
                logits_q_ref=out.logits_q,
            )
        tgt_slot: dict[int, tuple[int, int]] = {}
        dr_slot: dict[int, tuple[int, int]] = {}
        scratch_owner = None
        target_packed = None
        draft_packed = None
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
        if phase2a:
            # Safety guard: keep Phase-2 off until draft scratch forward writes KV.
            dr_slot = {}
            draft_packed = None

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
            draft_node_to_slot=dr_slot,
            expanded_seqs=bundle.expanded_seqs,
            branch_states=bundle.branch_states,
            scratch_owner=scratch_owner,
            target_scratch_packed=target_packed,
            draft_scratch_packed=draft_packed,
        )
        assert out.speculations.shape == (b_exp, k1)
        assert out.logits_q.shape[:2] == (b_exp, max(0, k1 - 1))
        return out
