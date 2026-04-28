from __future__ import annotations

from dataclasses import dataclass

import torch

from ssd.engine.sequence import Sequence


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


def build_tree_mask(path_node_ids: list[list[int]], *, device: torch.device) -> torch.Tensor:
    """Build a simple path-only keep mask for packed tree queries."""
    if not path_node_ids:
        return torch.empty((0,), dtype=torch.uint8, device=device)
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
    return mask.flatten().to(torch.uint8)


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
    for i, toks in enumerate(path_token_ids):
        seq = seqs[i if i < len(seqs) else 0]
        c = int(seq.num_tokens)
        context_lens[i] = c
        bt = seq.draft_block_table if use_draft_table else seq.block_table
        block_tables_host.append([int(x) for x in bt])
        for j in range(len(toks)):
            p = c + j
            positions_host.append(p)
            bidx = p // block_size
            off = p % block_size
            slot_mapping_host.append(int(bt[bidx]) * block_size + off)
    max_blocks = max(len(t) for t in block_tables_host)
    block_tables = torch.full((bsz, max_blocks), -1, dtype=torch.int32, device=device)
    for i, row in enumerate(block_tables_host):
        if row:
            block_tables[i, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
    positions = torch.tensor(positions_host, dtype=torch.int64, device=device)
    slot_mapping = torch.tensor(slot_mapping_host, dtype=torch.int64, device=device)
    cu = torch.arange(0, bsz + 1, dtype=torch.int32, device=device) * row_len
    mask = build_tree_mask(
        [list(range(i * row_len, (i + 1) * row_len)) for i in range(bsz)],
        device=device,
    )
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
