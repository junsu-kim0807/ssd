"""Flat-pivot-only target-scratch dataclasses.

These types are intentionally separate from the PivotTree scratch infrastructure
(``pivot_types.ScratchOwner`` / ``PivotTreeCommitBundle`` /
``pivot_tree_helpers``). The flat pivot target-scratch path must NOT import any
of those: keeping the type lattice disjoint prevents PivotTree semantics from
leaking into flat pivot and lets either side evolve independently.

Lifecycle invariant: every ``FlatPivotTargetScratchOwner`` instance is created
exactly once per step (inside ``_verify_expanded_target_scratch``) and is
released exactly once via ``release_unreleased`` from one of three sites:

  1. verify-time exception handler (allocation succeeded, downstream raised);
  2. ``postprocess_pivot_target_scratch`` finally clause (normal path);
  3. ``step.py`` outer finally as a defensive fallback.

The ``released`` flag makes ``release_unreleased`` idempotent so multiple call
sites cannot double-free. Do not mutate ``target_block_ids`` after construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlatPivotTargetScratchOwner:
    """Owns the target scratch block ids allocated for one flat-pivot step.

    ``target_block_manager`` here is the *target* BlockManager (the same object
    ``Scheduler.block_manager`` points at). Draft / intermediate scratch are
    not the concern of this owner.
    """
    target_block_ids: list[int]
    released: bool = False

    def release_unreleased(self, target_block_manager) -> None:
        if self.released:
            return
        if self.target_block_ids:
            target_block_manager.release_scratch_blocks(self.target_block_ids)
        self.released = True


@dataclass
class FlatPivotTargetScratchPackedInputs:
    """Packed inputs for one flat-pivot target-scratch verify forward.

    Shapes are described in ``build_flat_pivot_target_scratch_packed_inputs``.
    ``path_node_ids`` and ``node_to_slot`` are the bookkeeping that lets the
    postprocess slot-copy phase find each accepted token's source slot in the
    scratch KV.
    """
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    attn_mask: torch.Tensor | None
    path_node_ids: list[list[int]]
    node_to_slot: dict[int, tuple[int, int]]
    scratch_owner: FlatPivotTargetScratchOwner


@dataclass
class FlatPivotTargetScratchCommitBundle:
    """Carries the verify-time scratch bookkeeping into postprocess.

    Indexing convention:
      * ``winner_target_node_ids[i]`` = node ids for parent ``i``'s winning
        expanded row, truncated to the post-EOS suffix length at postprocess
        time. The verify-side fills it with the raw winner-row prefix; the
        postprocess slices to ``len(new_suffix)``.
      * ``target_node_slot[node_id] = (scratch_block_id, intra_block_offset)``.
        Together with the winner's parent block table this gives src/dst pairs
        for ``copy_kv_slots``.
      * ``raw_suffix_lens[i]`` = pre-EOS-truncation suffix length, kept so that
        postprocess can assert the truncation never *grows* the suffix.
    """
    winner_target_node_ids: list[list[int]]
    target_node_slot: dict[int, tuple[int, int]]
    raw_suffix_lens: list[int]
    scratch_owner: FlatPivotTargetScratchOwner
    num_scratch_blocks: int
    num_committed_slots: int = 0
