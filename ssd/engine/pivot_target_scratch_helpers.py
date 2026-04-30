"""Flat-pivot target-scratch packed-input builder.

This module is intentionally self-contained: it MUST NOT import any
``ssd.engine.helpers.pivot_tree_helpers`` symbol nor any PivotTree dataclass.
The only shared infrastructure is the BlockManager scratch API
(``allocate_scratch_blocks`` / ``release_scratch_blocks``) and the ModelRunner
``copy_kv_blocks`` partial-block copy primitive — both generic, not PivotTree-
specific.

Geometry (one expanded row, parent KV cached up to ``c0``):
  * ``partial = c0 % block_size``
  * ``n_full = c0 // block_size``  (full prefix blocks already in parent.block_table)
  * ``L = lookahead + 1``  (verify candidate length)
  * ``nscratch = ceil((partial + L) / block_size)``
  * Page table for this row: ``parent.block_table[:n_full] + scratch_blocks``
    (parent's partial last block is REPLACED by ``scratch_blocks[0]``).
  * If ``partial > 0``: pre-copy parent.block_table[n_full] -> scratch_blocks[0]
    for slots ``[0, partial)`` so attention reads of those positions see valid
    parent KV. Verify writes fill slots ``[partial, partial+L)``.
  * ``slot_mapping[row, j] = row_bt[(c0+j) // block_size] * block_size + (c0+j) % block_size``
    which by construction lands inside the scratch portion (one or two blocks).
  * ``context_lens[row] = c0 + L`` and the causal mask filters non-causal keys.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ssd.engine.pivot_flat_scratch_types import (
    FlatPivotTargetScratchOwner,
    FlatPivotTargetScratchPackedInputs,
)

if TYPE_CHECKING:
    from ssd.engine.block_manager import BlockManager
    from ssd.engine.sequence import Sequence


def _build_flat_causal_mask(
    pos0_per_row: list[int],
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-row causal mask, prefix fully visible.

    For row with prefix length ``pos0`` and query length ``L``, key length is
    ``pos0 + L``; query ``j`` attends to keys ``[0, pos0 + j]``. Returned shape
    is the flat concatenation across rows (matches the format
    ``run_packed_tree_decode`` consumes via ``custom_mask``).
    """
    if not pos0_per_row:
        return torch.empty((0,), dtype=torch.bool, device=device)
    chunks: list[torch.Tensor] = []
    for pos0 in pos0_per_row:
        p0 = int(pos0)
        kv_len = p0 + L
        m = torch.zeros((L, kv_len), dtype=torch.bool, device=device)
        for j in range(L):
            m[j, : p0 + j + 1] = True
        chunks.append(m.reshape(-1))
    return torch.cat(chunks).to(torch.bool)


def build_flat_pivot_target_scratch_packed_inputs(
    *,
    parent_seqs: list["Sequence"],
    parent_index_per_branch: list[int],
    speculations: torch.Tensor,  # [B_exp, L]
    block_size: int,
    target_block_manager: "BlockManager",
    target_model_runner,
    device: torch.device,
) -> FlatPivotTargetScratchPackedInputs:
    """Build packed verify inputs for flat-pivot target scratch.

    Side effects:
      * Allocates ``nscratch_per_row`` target scratch blocks for each expanded row.
      * For rows with non-zero partial, issues a target ``copy_kv_blocks`` call
        copying ``partial`` slots from parent's last partial block into
        ``scratch_blocks[0]``. This is the only target-side KV write we do at
        verify-prep time (winner accepted-suffix slot copy happens later in
        postprocess).

    On any exception inside this function, the caller is expected to release
    the partially-built ``scratch_owner.target_block_ids``. To keep that simple
    we accumulate scratch ids into a single owner created at the start and
    raise without any cleanup — the caller handles release via the owner.
    """
    b_exp = len(parent_index_per_branch)
    L = int(speculations.shape[1]) if b_exp > 0 else 0

    # Owner is created up-front so partial allocations are still releasable.
    scratch_owner = FlatPivotTargetScratchOwner(target_block_ids=[])

    if b_exp == 0:
        z = torch.empty((0,), dtype=torch.int64, device=device)
        zi = torch.zeros((1,), dtype=torch.int32, device=device)
        return FlatPivotTargetScratchPackedInputs(
            input_ids=z,
            positions=z,
            slot_mapping=z,
            context_lens=torch.empty((0,), dtype=torch.int32, device=device),
            block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            cu_seqlens_q=zi,
            max_seqlen_q=0,
            attn_mask=None,
            path_node_ids=[],
            node_to_slot={},
            scratch_owner=scratch_owner,
        )

    # Speculations -> host once.
    spec_host = speculations.to("cpu", dtype=torch.int64).tolist()
    assert len(spec_host) == b_exp and all(len(r) == L for r in spec_host)

    # Per-row scratch alloc + (optional) partial-block copy plan.
    block_tables_host: list[list[int]] = []
    pos0_per_row: list[int] = []
    positions_host: list[int] = []
    slot_mapping_host: list[int] = []
    input_ids_host: list[int] = []
    path_node_ids: list[list[int]] = []
    node_to_slot: dict[int, tuple[int, int]] = {}

    partial_copy_src: list[int] = []
    partial_copy_dst: list[int] = []
    partial_copy_valid: list[int] = []

    for row in range(b_exp):
        parent = parent_seqs[parent_index_per_branch[row]]
        c0 = int(parent.num_cached_tokens)
        partial = c0 % block_size
        n_full = c0 // block_size
        nscratch = (partial + L + block_size - 1) // block_size

        scratch_blocks = target_block_manager.allocate_scratch_blocks(nscratch)
        # Persist into the owner immediately so a downstream raise is recoverable.
        scratch_owner.target_block_ids.extend(int(x) for x in scratch_blocks)

        # If parent has a partial last block, mirror its first ``partial`` slots
        # into scratch_blocks[0] so positions [n_full*block_size, c0) read
        # consistent KV through the same logical block index.
        if partial > 0:
            assert n_full < len(parent.block_table), (
                "flat target scratch: parent has partial tokens but no partial block in block_table"
            )
            partial_copy_src.append(int(parent.block_table[n_full]))
            partial_copy_dst.append(int(scratch_blocks[0]))
            partial_copy_valid.append(int(partial))

        # Page table: full prefix blocks + scratch (replaces partial last parent block).
        row_bt = [int(b) for b in parent.block_table[:n_full]] + [int(b) for b in scratch_blocks]
        block_tables_host.append(row_bt)
        pos0_per_row.append(c0)

        # Per-token packing.
        for j in range(L):
            pos = c0 + j
            bidx = pos // block_size
            off = pos % block_size
            assert bidx < len(row_bt), (
                f"flat target scratch row_bt OOB: bidx={bidx} len={len(row_bt)} "
                f"c0={c0} L={L} partial={partial} n_full={n_full} nscratch={nscratch}"
            )
            bid = int(row_bt[bidx])
            slot_mapping_host.append(bid * block_size + off)
            positions_host.append(pos)
            input_ids_host.append(int(spec_host[row][j]))
            node_id = row * L + j
            if len(path_node_ids) <= row:
                path_node_ids.append([])
            path_node_ids[row].append(node_id)
            node_to_slot[node_id] = (bid, off)

    # Issue partial-block copies in one batched call. ``copy_kv_blocks`` uses
    # the COW copy mode env (defaults to full_block which over-copies but is a
    # single kernel; here the per-call src/dst lists are tiny so either works).
    if partial_copy_src:
        target_model_runner.call(
            "copy_kv_blocks",
            partial_copy_src,
            partial_copy_dst,
            partial_copy_valid,
            "target",
        )

    # Tensorize.
    max_blocks = max(len(t) for t in block_tables_host)
    block_tables_t = torch.full((b_exp, max_blocks), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(block_tables_host):
        if row:
            block_tables_t[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)

    input_ids_t = torch.tensor(input_ids_host, dtype=torch.int64, device=device)
    positions_t = torch.tensor(positions_host, dtype=torch.int64, device=device)
    slot_mapping_t = torch.tensor(slot_mapping_host, dtype=torch.int64, device=device)
    context_lens_t = torch.tensor(
        [pos0 + L for pos0 in pos0_per_row], dtype=torch.int32, device=device
    )
    cu_seqlens_q_t = torch.arange(0, b_exp + 1, dtype=torch.int32, device=device) * L
    attn_mask = _build_flat_causal_mask(pos0_per_row, L, device=device)

    return FlatPivotTargetScratchPackedInputs(
        input_ids=input_ids_t,
        positions=positions_t,
        slot_mapping=slot_mapping_t,
        context_lens=context_lens_t,
        block_tables=block_tables_t,
        cu_seqlens_q=cu_seqlens_q_t,
        max_seqlen_q=L,
        attn_mask=attn_mask,
        path_node_ids=path_node_ids,
        node_to_slot=node_to_slot,
        scratch_owner=scratch_owner,
    )
