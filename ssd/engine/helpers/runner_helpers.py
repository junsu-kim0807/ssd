import torch
import torch.distributed as dist

from ssd.engine.sequence import Sequence

def prepare_prefill_payload(
    input_id_list: list[list[int]],
    eagle_acts: torch.Tensor,
    device: torch.device,
    max_blocks: int,
    draft_block_tables: list[list[int]] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_flat = []
    num_tokens = []
    for input_ids in input_id_list:
        input_ids_flat.extend(input_ids)
        num_tokens.append(len(input_ids))
    input_ids_flat = torch.tensor(input_ids_flat, dtype=torch.int64, device=device)
    num_tokens = torch.tensor(num_tokens, dtype=torch.int64, device=device)
    if isinstance(draft_block_tables, list):
        draft_block_table = torch.tensor(
            [dbt + [-1] * (max_blocks - len(dbt)) for dbt in draft_block_tables],
            dtype=torch.int32, device=device,
        )
    else:
        assert draft_block_tables.shape == (len(input_id_list), max_blocks), (
            f"draft_block_tables shape mismatch: expected ({len(input_id_list), max_blocks}), got {draft_block_tables.shape}"
        )
        draft_block_table = draft_block_tables

    # 3) send cmd=1
    cmd = torch.tensor([1], dtype=torch.int64, device=device)

    # 4) send metadata for tensor reconstruction
    metadata = torch.tensor([
        input_ids_flat.size(0),
        len(input_id_list),  # batch_size
        max_blocks,
        1 if eagle_acts is not None else 0,
        eagle_acts.shape[1] if eagle_acts is not None else 0,
    ], dtype=torch.int64, device=device)

    if eagle_acts is not None:
        assert eagle_acts.shape[0] == input_ids_flat.shape[0], (
            f"Eagle activations length {eagle_acts.shape[0]} != input_ids_flat length {input_ids_flat.shape[0]}"
        )

    return cmd, metadata, input_ids_flat, num_tokens, draft_block_table, eagle_acts

def _kv_block_table_and_cached(seq: Sequence, is_draft: bool, is_intermediate: bool):
    assert not (is_draft and is_intermediate), "draft and intermediate KV are mutually exclusive"
    if is_intermediate:
        return seq.inter_block_table, seq.num_inter_cached_tokens
    if is_draft:
        return seq.draft_block_table, seq.num_draft_cached_tokens
    return seq.block_table, seq.num_cached_tokens


def prepare_decode_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool,
    verify: bool = False,
    k: int = -1,
    is_intermediate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    if not verify:  # normal decoding or draft fwd in speculation
        assert k == -1, "k should be -1 for normal decoding or draft fwd in speculation"
        for seq in seqs:
            block_table, num_cached_tokens = _kv_block_table_and_cached(seq, is_draft, is_intermediate)
            assert len(seq) // block_size <= len(block_table), "in sync spec draft decode, not enough blocks allocated"
            expected = len(seq) - 1 + (seq.hv_num_provisional_tokens if is_draft else 0)
            assert num_cached_tokens == expected, (
                f"num_cached_tokens should be {expected} in sq decode path "
                f"(got {num_cached_tokens}, len={len(seq)}, prov={seq.hv_num_provisional_tokens})"
            )
            if is_draft and seq.hv_num_provisional_tokens > 0:
                # Logical frontier after HV provisional tail (tokens not in seq.token_ids).
                pt = seq.hv_num_provisional_tokens
                prov = seq.hv_provisional_token_ids
                assert len(prov) == pt, "hv_num_provisional_tokens must match provisional list length"
                # Only the first draft forward of this speculate may sit purely on the provisional
                # stem (recovery skipped onto ``token_ids``). Later forwards must use the token
                # just appended so autoregressive draft advances.
                if seq.num_tokens == seq.num_cached_tokens:
                    last_tok = prov[-1]
                else:
                    last_tok = seq.last_token
                logical_pos = len(seq) - 1 + pt
                context_len = len(seq) + pt
            else:
                last_tok = seq.last_token
                logical_pos = len(seq) - 1
                context_len = len(seq)
            input_ids.append(last_tok)
            positions.append(logical_pos)
            context_lens.append(context_len)

            block_idx = logical_pos // block_size
            pos_in_block = logical_pos % block_size
            slot_mapping.append(block_table[block_idx] * block_size + pos_in_block)
    else:  # verify and glue decode prep both go here
        assert not is_draft, "verify path only supported for target model" # we prep tensors to send to draft for glue on the target 
        assert k > 0, "k should be > 0 for target fwd in verify"

        for seq_idx, seq in enumerate(seqs):
            # can hardcode block_table here for target since this is only target codepath 
            assert (seq.num_tokens - 1) // block_size <= len(seq.block_table), "in sync spec target verify, not enough blocks allocated"
            
            pos0 = seq.num_tokens - (k+1)
            input_ids.extend(seq[pos0:])
            positions.extend(list(range(pos0, pos0 + k + 1)))
            assert seq.num_cached_tokens == pos0, f"num_cached_tokens={seq.num_cached_tokens} != pos0={pos0} (num_tokens={seq.num_tokens}, k={k})"
            context_lens.append(len(seq))  

            for j in range(k + 1):
                pos = pos0 + j
                block_idx = pos // block_size
                block_id = seq.block_table[block_idx]
                pos_in_block = pos % block_size
                slot_mapping.append(
                    block_id * block_size + pos_in_block)

    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(
        context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    return input_ids, positions, slot_mapping, context_lens


def prepare_verify_tensors_varlen(
    seqs: list[Sequence],
    block_size: int,
    verify_tokens_per_seq: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed variable-length target verify: one row per seq, query starts at seq.num_cached_tokens.

    `verify_tokens_per_seq[i]` is the full candidate suffix (length L_i) scored against KV
    whose context ends at num_cached_tokens (committed frontier, not including temp speculate).
    """
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    seqlen_q_list: list[int] = []
    for seq, cand in zip(seqs, verify_tokens_per_seq):
        c0 = seq.num_cached_tokens
        L = len(cand)
        assert L >= 1
        # Match fixed verify path: constant context length for the whole batch row.
        context_lens.append(len(seq))
        seqlen_q_list.append(L)
        for j, tok in enumerate(cand):
            pos = c0 + j
            input_ids.append(int(tok))
            positions.append(pos)
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            block_id = seq.block_table[block_idx]
            slot_mapping.append(block_id * block_size + pos_in_block)

    input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    _dev = torch.device("cuda")
    cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=_dev)
    q_lens = torch.tensor(seqlen_q_list, dtype=torch.int32, device=_dev)
    cu_seqlens_q[1:] = torch.cumsum(q_lens, dim=0)
    max_seqlen_q = max(seqlen_q_list) if seqlen_q_list else 0
    return input_ids_t, positions_t, slot_mapping_t, context_lens_t, cu_seqlens_q, max_seqlen_q


def prepare_intermediate_verify_suffix_tensors(
    seqs: list[Sequence],
    block_size: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One packed forward over the speculative tail on the *intermediate* KV.

    The chain is always length ``k+1``: recovery then ``k`` draft tokens. When HV
    ``skip_append`` left recovery only on ``hv_provisional_token_ids[-1]``, the tail is
    ``[prov[-1]] + token_ids[num_cached : num_cached + k]``; otherwise it is
    ``token_ids[num_cached : num_cached + k + 1]``.

    RoPE positions start at ``num_inter_cached_tokens`` (intermediate KV depth before this
    forward), which must match prior intermediate accepts + provisional depth — not only
    ``num_cached_tokens`` — so intermediate scoring aligns with the draft's logical chain.
    """
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    seqlen_q_list: list[int] = []
    for seq in seqs:
        c0 = seq.num_cached_tokens
        base_pos = seq.num_inter_cached_tokens
        assert base_pos >= c0, (
            f"intermediate verify: num_inter_cached_tokens must be >= num_cached_tokens, "
            f"got {base_pos} vs {c0}"
        )
        pt = seq.hv_num_provisional_tokens
        if pt > 0:
            tail = [seq.hv_provisional_token_ids[-1]] + list(seq.token_ids[c0 : c0 + k])
        else:
            tail = list(seq.token_ids[c0 : c0 + k + 1])
        assert len(tail) == k + 1, (
            f"intermediate verify: need K+1 tokens on tail, got {len(tail)} (K={k})"
        )
        seqlen_q_list.append(k + 1)
        context_lens.append(base_pos + k + 1)
        for j, tok in enumerate(tail):
            pos = base_pos + j
            input_ids.append(int(tok))
            positions.append(pos)
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            bid = seq.inter_block_table[block_idx]
            slot_mapping.append(bid * block_size + pos_in_block)

    input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    _dev = torch.device("cuda")
    cu_seqlens_q = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=_dev)
    q_lens = torch.tensor(seqlen_q_list, dtype=torch.int32, device=_dev)
    cu_seqlens_q[1:] = torch.cumsum(q_lens, dim=0)
    max_seqlen_q = k + 1
    return input_ids_t, positions_t, slot_mapping_t, context_lens_t, cu_seqlens_q, max_seqlen_q


def prepare_block_tables_from_seqs(
    seqs: list[Sequence],
    is_draft: bool = False,
    is_intermediate: bool = False,
) -> torch.Tensor:
        assert not (is_draft and is_intermediate)
        if is_intermediate:
            max_len = max(len(seq.inter_block_table) for seq in seqs)
            block_tables = [seq.inter_block_table + [-1] * (max_len - len(seq.inter_block_table)) for seq in seqs]
        elif is_draft:
            max_len = max(len(seq.draft_block_table) for seq in seqs)
            block_tables = [seq.draft_block_table + [-1] * (max_len - len(seq.draft_block_table)) for seq in seqs]
        else:
            max_len = max(len(seq.block_table) for seq in seqs)
            block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

def prepare_prefill_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool = False,
    skip_first_token: int = 0,
    is_intermediate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert skip_first_token in (0, 1)
    assert not (is_draft and is_intermediate)
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    
    for seq in seqs:
        seqlen = len(seq)
        block_table, num_cached_tokens = _kv_block_table_and_cached(seq, is_draft, is_intermediate)

        start = num_cached_tokens + (skip_first_token if is_draft else 0)
        input_ids.extend(seq[start:])
        pos_offset = -skip_first_token if is_draft else 0
        positions.extend(list(range(start + pos_offset, seqlen + pos_offset)))
        seqlen_q = seqlen - start
        seqlen_k = seqlen + pos_offset
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not block_table:  # first prefill
            continue

        # new: emit exactly one slot for each *new* token
        #    map each token index -> (block_id * block_size + offset)
        for pos in range(start + pos_offset, seq.num_tokens + pos_offset):
            block_i = pos // block_size
            offset = pos % block_size
            slot = block_table[block_i] * block_size + offset
            slot_mapping.append(slot)

    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(
        cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(
        cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping
