from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ssd.engine.pivot_types import ScratchOwner
from ssd.engine.sequence import Sequence

if TYPE_CHECKING:
    from ssd.engine.block_manager import BlockManager


def can_use_target_scratch_phase1a(seq: Sequence) -> bool:
    """Phase-1A gate: target scratch verify without partial-block COW on parent."""
    bs = int(seq.block_size)
    return bs > 0 and int(seq.num_cached_tokens) % bs == 0


def can_use_draft_scratch_phase2a(seq: Sequence) -> bool:
    """Phase-2A gate: draft scratch only on block-aligned draft frontier."""
    bs = int(seq.block_size)
    return bs > 0 and int(seq.num_draft_cached_tokens) % bs == 0


def gather_logits_by_path(
    logits_tree_flat: torch.Tensor,
    path_node_ids: list[list[int]],
) -> torch.Tensor:
    """Index packed-tree logits by per-row node ids; output [B_exp, K+1, V]."""
    if not path_node_ids:
        return logits_tree_flat.view(0, 0, logits_tree_flat.shape[-1])
    row_len = len(path_node_ids[0])
    flat_ids = [int(nid) for row in path_node_ids for nid in row]
    ids = torch.tensor(flat_ids, dtype=torch.long, device=logits_tree_flat.device)
    out = logits_tree_flat[ids]
    return out.view(len(path_node_ids), row_len, -1)


@dataclass
class PackedTreeDecodeInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    tree_attn_mask: torch.Tensor | None


@dataclass
class TargetScratchPackedInputs:
    """Packed target verify tensors for Phase-1 scratch (row-wise scratch slots)."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    tree_attn_mask: torch.Tensor | None
    target_node_to_slot: dict[int, tuple[int, int]]
    scratch_owner: ScratchOwner


@dataclass
class DraftScratchPackedInputs:
    """Packed draft tensors for Phase-2 scratch path (row-wise baseline)."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    tree_attn_mask: torch.Tensor | None
    draft_node_to_slot: dict[int, tuple[int, int]]
    scratch_owner: ScratchOwner


def build_tree_mask(path_node_ids: list[list[int]], *, device: torch.device) -> torch.Tensor:
    """Phase-0 debug packer: q×q mask over flattened packed queries (same row only).

    For Phase-1A target scratch use :func:`build_rowwise_prefix_candidate_mask` instead;
    FlashInfer expects per-row masks sized ``(K+1) * (pos0 + K + 1)`` flattened.
    """
    if not path_node_ids:
        return torch.empty((0,), dtype=torch.bool, device=device)
    row_len = len(path_node_ids[0])
    n_rows = len(path_node_ids)
    q = n_rows * row_len
    # Conservative mask: causal lower-triangular over each row.
    mask = torch.zeros((q, q), dtype=torch.bool, device=device)
    for ridx in range(n_rows):
        base = ridx * row_len
        for j in range(row_len):
            cur = base + j
            mask[cur, base : cur + 1] = True
    return mask.flatten()


def build_rowwise_prefix_candidate_mask(
    pos0_per_row: list[int],
    k1: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Per-row causal mask over KV positions ``0 .. pos0+K`` for packed tree verify.

    Query row has length ``k1`` (K+1). For expanded row with committed prefix length
    ``pos0`` (``num_cached_tokens`` at verify), key length is ``kv_len = pos0 + k1``.
    Query ``j`` may attend to keys ``0 .. pos0 + j`` inclusive.
    """
    if not pos0_per_row:
        return torch.empty((0,), dtype=torch.bool, device=device)
    chunks: list[torch.Tensor] = []
    for pos0 in pos0_per_row:
        p0 = int(pos0)
        kv_len = p0 + k1
        m = torch.zeros((k1, kv_len), dtype=torch.bool, device=device)
        for j in range(k1):
            m[j, : p0 + j + 1] = True
        chunks.append(m.reshape(-1))
    return torch.cat(chunks).to(torch.bool)


def build_phase0_packed_inputs(
    seqs: list[Sequence],
    path_token_ids: list[list[int]],
    *,
    block_size: int,
    device: torch.device,
    use_draft_table: bool,
) -> PackedTreeDecodeInputs:
    """Pack expanded path tokens into a simple varlen verify batch.

    This is a correctness-first (Phase 0) packer. It does not attempt cross-row
    tree reuse and simply preserves row-wise K+1 geometry.
    """
    if not path_token_ids:
        z = torch.empty((0,), dtype=torch.int64, device=device)
        zi = torch.zeros((1,), dtype=torch.int32, device=device)
        return PackedTreeDecodeInputs(
            input_ids=z,
            positions=z,
            slot_mapping=z,
            context_lens=torch.empty((0,), dtype=torch.int32, device=device),
            block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            cu_seqlens_q=zi,
            max_seqlen_q=0,
            tree_attn_mask=None,
        )
    row_len = len(path_token_ids[0])
    input_ids = torch.tensor(path_token_ids, dtype=torch.int64, device=device).reshape(-1)
    bsz = len(path_token_ids)
    context_lens = torch.empty((bsz,), dtype=torch.int32, device=device)
    block_tables_host: list[list[int]] = []
    positions_host: list[int] = []
    slot_mapping_host: list[int] = []
    pos0_per_row: list[int] = []
    for i, toks in enumerate(path_token_ids):
        seq = seqs[i if i < len(seqs) else 0]
        c = int(seq.num_tokens)
        pos0 = c - row_len
        pos0_per_row.append(pos0)
        assert pos0 >= 0, (
            f"phase0 pack invariant failed: negative pos0={pos0} "
            f"(num_tokens={c}, row_len={row_len})"
        )
        assert int(seq.num_cached_tokens) == pos0, (
            f"phase0 pack invariant failed: "
            f"num_cached_tokens={seq.num_cached_tokens}, pos0={pos0}, "
            f"num_tokens={seq.num_tokens}, row_len={row_len}"
        )
        context_lens[i] = c
        bt = seq.draft_block_table if use_draft_table else seq.block_table
        block_tables_host.append([int(x) for x in bt])
        for j in range(len(toks)):
            p = pos0 + j
            positions_host.append(p)
            bidx = p // block_size
            off = p % block_size
            assert bidx < len(bt), (
                f"phase0 pack block-table OOB: bidx={bidx}, len(bt)={len(bt)}, "
                f"p={p}, block_size={block_size}, pos0={pos0}, row_len={row_len}"
            )
            slot_mapping_host.append(int(bt[bidx]) * block_size + off)
    max_blocks = max(len(t) for t in block_tables_host)
    block_tables = torch.full((bsz, max_blocks), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(block_tables_host):
        if row:
            block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_host, dtype=torch.int64, device=device)
    slot_mapping = torch.tensor(slot_mapping_host, dtype=torch.int64, device=device)
    cu = torch.arange(0, bsz + 1, dtype=torch.int32, device=device) * row_len
    mask = build_rowwise_prefix_candidate_mask(pos0_per_row, row_len, device=device)
    return PackedTreeDecodeInputs(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu,
        max_seqlen_q=row_len,
        tree_attn_mask=mask,
    )


def build_target_scratch_packed_inputs(
    expanded_seqs: list[Sequence],
    path_token_ids: list[list[int]],
    path_node_ids: list[list[int]],
    block_manager: "BlockManager",
    *,
    block_size: int,
    device: torch.device,
    lookahead: int,
) -> TargetScratchPackedInputs:
    """Phase-1A: parent full-block prefix in page table; (K+1) verify writes in scratch slots.

    Requires ``expanded_seqs[r].num_cached_tokens == pos0`` with
    ``pos0 = num_tokens - (lookahead+1)`` (same invariant as flat target verify).
    """
    if not path_token_ids:
        z = torch.empty((0,), dtype=torch.int64, device=device)
        zi = torch.zeros((1,), dtype=torch.int32, device=device)
        return TargetScratchPackedInputs(
            input_ids=z,
            positions=z,
            slot_mapping=z,
            context_lens=torch.empty((0,), dtype=torch.int32, device=device),
            block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            cu_seqlens_q=zi,
            max_seqlen_q=0,
            tree_attn_mask=None,
            target_node_to_slot={},
            scratch_owner=ScratchOwner(target_block_ids=[], draft_block_ids=[]),
        )
    row_len = lookahead + 1
    assert len(path_token_ids[0]) == row_len
    bsz = len(path_token_ids)
    flat_nodes = [int(n) for row in path_node_ids for n in row]
    assert flat_nodes == list(range(len(flat_nodes))), (
        "build_target_scratch_packed_inputs: expect contiguous node_id 0..N-1 row-major"
    )
    input_ids = torch.tensor(path_token_ids, dtype=torch.int64, device=device).reshape(-1)
    context_lens = torch.empty((bsz,), dtype=torch.int32, device=device)
    block_tables_host: list[list[int]] = []
    positions_host: list[int] = []
    slot_mapping_host: list[int] = []
    target_node_to_slot: dict[int, tuple[int, int]] = {}
    all_scratch: list[int] = []
    k1 = row_len
    nscratch = (k1 + block_size - 1) // block_size
    pos0_per_row: list[int] = []
    for r in range(bsz):
        exp = expanded_seqs[r]
        pos0 = int(exp.num_tokens) - k1
        pos0_per_row.append(pos0)
        assert int(exp.num_cached_tokens) == pos0, (
            f"phase1 scratch pack: num_cached_tokens={exp.num_cached_tokens} != pos0={pos0} "
            f"(num_tokens={exp.num_tokens}, k1={k1})"
        )
        assert int(exp.num_cached_tokens) % block_size == 0
        context_lens[r] = int(exp.num_tokens)
        n_pref = (pos0 + block_size - 1) // block_size
        prefix_blocks = [int(x) for x in exp.block_table[:n_pref]]
        scratch_blocks = block_manager.allocate_scratch_blocks(nscratch)
        all_scratch.extend(scratch_blocks)
        row_bt = prefix_blocks + scratch_blocks
        block_tables_host.append(row_bt)
        for j in range(k1):
            pos = pos0 + j
            positions_host.append(pos)
            bidx = pos // block_size
            off = pos % block_size
            bid = int(row_bt[bidx])
            slot_mapping_host.append(bid * block_size + off)
            nid = int(path_node_ids[r][j])
            target_node_to_slot[nid] = (bid, off)
    max_blocks = max(len(t) for t in block_tables_host)
    block_tables = torch.full((bsz, max_blocks), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(block_tables_host):
        if row:
            block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_host, dtype=torch.int64, device=device)
    slot_mapping = torch.tensor(slot_mapping_host, dtype=torch.int64, device=device)
    cu = torch.arange(0, bsz + 1, dtype=torch.int32, device=device) * k1
    mask = build_rowwise_prefix_candidate_mask(pos0_per_row, k1, device=device)
    owner = ScratchOwner(target_block_ids=list(all_scratch), draft_block_ids=[])
    return TargetScratchPackedInputs(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu,
        max_seqlen_q=k1,
        tree_attn_mask=mask,
        target_node_to_slot=target_node_to_slot,
        scratch_owner=owner,
    )


def build_target_scratch_packed_inputs_from_paths(
    parent_seqs: list[Sequence],
    path_parent_seq_idx: list[int],
    path_token_ids: list[list[int]],
    path_node_ids: list[list[int]],
    block_manager: "BlockManager",
    *,
    block_size: int,
    device: torch.device,
    lookahead: int,
) -> TargetScratchPackedInputs:
    """Phase-2A target scratch packer using parent sequences (no expanded_seqs)."""
    if not path_token_ids:
        z = torch.empty((0,), dtype=torch.int64, device=device)
        zi = torch.zeros((1,), dtype=torch.int32, device=device)
        return TargetScratchPackedInputs(
            input_ids=z,
            positions=z,
            slot_mapping=z,
            context_lens=torch.empty((0,), dtype=torch.int32, device=device),
            block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            cu_seqlens_q=zi,
            max_seqlen_q=0,
            tree_attn_mask=None,
            target_node_to_slot={},
            scratch_owner=ScratchOwner(target_block_ids=[], draft_block_ids=[]),
        )
    row_len = lookahead + 1
    bsz = len(path_token_ids)
    assert len(path_node_ids) == bsz
    input_ids = torch.tensor(path_token_ids, dtype=torch.int64, device=device).reshape(-1)
    context_lens = torch.empty((bsz,), dtype=torch.int32, device=device)
    block_tables_host: list[list[int]] = []
    positions_host: list[int] = []
    slot_mapping_host: list[int] = []
    target_node_to_slot: dict[int, tuple[int, int]] = {}
    all_scratch: list[int] = []
    nscratch = (row_len + block_size - 1) // block_size
    pos0_per_row: list[int] = []
    try:
        for r in range(bsz):
            pidx = int(path_parent_seq_idx[r])
            seq = parent_seqs[pidx]
            pos0 = int(seq.num_tokens)
            pos0_per_row.append(pos0)
            assert int(seq.num_cached_tokens) == pos0
            assert pos0 % block_size == 0
            context_lens[r] = pos0 + row_len
            n_pref = pos0 // block_size
            prefix_blocks = [int(x) for x in seq.block_table[:n_pref]]
            scratch_blocks = block_manager.allocate_scratch_blocks(nscratch)
            all_scratch.extend(scratch_blocks)
            row_bt = prefix_blocks + scratch_blocks
            block_tables_host.append(row_bt)
            for j in range(row_len):
                pos = pos0 + j
                positions_host.append(pos)
                bidx = pos // block_size
                off = pos % block_size
                bid = int(row_bt[bidx])
                slot_mapping_host.append(bid * block_size + off)
                nid = int(path_node_ids[r][j])
                target_node_to_slot[nid] = (bid, off)
    except Exception:
        block_manager.release_scratch_blocks(all_scratch)
        raise
    max_blocks = max(len(t) for t in block_tables_host)
    block_tables = torch.full((bsz, max_blocks), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(block_tables_host):
        if row:
            block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_host, dtype=torch.int64, device=device)
    slot_mapping = torch.tensor(slot_mapping_host, dtype=torch.int64, device=device)
    cu = torch.arange(0, bsz + 1, dtype=torch.int32, device=device) * row_len
    mask = build_rowwise_prefix_candidate_mask(pos0_per_row, row_len, device=device)
    owner = ScratchOwner(target_block_ids=list(all_scratch), draft_block_ids=[])
    return TargetScratchPackedInputs(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu,
        max_seqlen_q=row_len,
        tree_attn_mask=mask,
        target_node_to_slot=target_node_to_slot,
        scratch_owner=owner,
    )


def build_draft_scratch_packed_inputs(
    expanded_seqs: list[Sequence],
    path_token_ids: list[list[int]],
    path_node_ids: list[list[int]],
    block_manager: "BlockManager",
    *,
    block_size: int,
    device: torch.device,
    lookahead: int,
) -> DraftScratchPackedInputs:
    """Phase-2 row-wise draft scratch packing.

    This is correctness-first scaffolding for Phase-2A. It mirrors the target
    scratch packer geometry and allocates draft scratch slots per row.
    """
    if not path_token_ids:
        z = torch.empty((0,), dtype=torch.int64, device=device)
        zi = torch.zeros((1,), dtype=torch.int32, device=device)
        return DraftScratchPackedInputs(
            input_ids=z,
            positions=z,
            slot_mapping=z,
            context_lens=torch.empty((0,), dtype=torch.int32, device=device),
            block_tables=torch.empty((0, 0), dtype=torch.int32, device=device),
            cu_seqlens_q=zi,
            max_seqlen_q=0,
            tree_attn_mask=None,
            draft_node_to_slot={},
            scratch_owner=ScratchOwner(target_block_ids=[], draft_block_ids=[]),
        )
    row_len = lookahead + 1
    assert len(path_token_ids[0]) == row_len
    bsz = len(path_token_ids)
    input_ids = torch.tensor(path_token_ids, dtype=torch.int64, device=device).reshape(-1)
    context_lens = torch.empty((bsz,), dtype=torch.int32, device=device)
    block_tables_host: list[list[int]] = []
    positions_host: list[int] = []
    slot_mapping_host: list[int] = []
    draft_node_to_slot: dict[int, tuple[int, int]] = {}
    all_scratch: list[int] = []
    k1 = row_len
    nscratch = (k1 + block_size - 1) // block_size
    pos0_per_row: list[int] = []
    for r in range(bsz):
        exp = expanded_seqs[r]
        pos0 = int(exp.num_tokens) - k1
        pos0_per_row.append(pos0)
        assert int(exp.num_draft_cached_tokens) == pos0, (
            f"phase2 draft scratch pack: num_draft_cached_tokens={exp.num_draft_cached_tokens} "
            f"!= pos0={pos0} (num_tokens={exp.num_tokens}, k1={k1})"
        )
        assert int(exp.num_draft_cached_tokens) % block_size == 0
        context_lens[r] = int(exp.num_tokens)
        n_pref = (pos0 + block_size - 1) // block_size
        prefix_blocks = [int(x) for x in exp.draft_block_table[:n_pref]]
        scratch_blocks = block_manager.allocate_scratch_blocks(nscratch)
        all_scratch.extend(scratch_blocks)
        row_bt = prefix_blocks + scratch_blocks
        block_tables_host.append(row_bt)
        for j in range(k1):
            pos = pos0 + j
            positions_host.append(pos)
            bidx = pos // block_size
            off = pos % block_size
            bid = int(row_bt[bidx])
            slot_mapping_host.append(bid * block_size + off)
            nid = int(path_node_ids[r][j])
            draft_node_to_slot[nid] = (bid, off)
    max_blocks = max(len(t) for t in block_tables_host)
    block_tables = torch.full((bsz, max_blocks), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(block_tables_host):
        if row:
            block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_host, dtype=torch.int64, device=device)
    slot_mapping = torch.tensor(slot_mapping_host, dtype=torch.int64, device=device)
    cu = torch.arange(0, bsz + 1, dtype=torch.int32, device=device) * k1
    mask = build_rowwise_prefix_candidate_mask(pos0_per_row, k1, device=device)
    owner = ScratchOwner(target_block_ids=[], draft_block_ids=list(all_scratch))
    return DraftScratchPackedInputs(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu,
        max_seqlen_q=k1,
        tree_attn_mask=mask,
        draft_node_to_slot=draft_node_to_slot,
        scratch_owner=owner,
    )
