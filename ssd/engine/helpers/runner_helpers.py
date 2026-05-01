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
    *,
    hv_block_debug: bool = False,
    decode_lookahead_hint: int = 0,
    use_eagle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # EAGLE draft prefill consumes ``skip_first_token=1``, leaving draft KV one slot
    # short of the logical token count. Apply pos_offset=-1 in sync draft decode so
    # the recovery token lands at slot N-1 (matching async jit_speculate semantics).
    eagle_pos_offset = -1 if (is_draft and use_eagle and not verify) else 0
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    if not verify:  # normal decoding or draft fwd in speculation
        assert k == -1, "k should be -1 for normal decoding or draft fwd in speculation"
        for seq in seqs:
            block_table, num_cached_tokens = _kv_block_table_and_cached(seq, is_draft, is_intermediate)
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
                logical_pos = len(seq) - 1 + eagle_pos_offset
                context_len = len(seq) + eagle_pos_offset
            required_blocks_now = (context_len + block_size - 1) // block_size
            required_blocks_with_lookahead = (
                (context_len + decode_lookahead_hint + block_size - 1) // block_size
                if decode_lookahead_hint > 0
                else required_blocks_now
            )

            if hv_block_debug and (is_draft or is_intermediate):
                _prov = int(getattr(seq, "hv_num_provisional_tokens", 0))
                _dctx = (len(seq) - 1 + _prov) if is_draft else None
                print(
                    "[HV_BLOCK_DEBUG:prepare_decode] "
                    f"seq_id={seq.seq_id} "
                    f"is_draft={is_draft} "
                    f"is_intermediate={is_intermediate} "
                    f"num_tokens={seq.num_tokens} "
                    f"num_draft_cached_tokens={seq.num_draft_cached_tokens} "
                    f"hv_num_provisional_tokens={_prov} "
                    f"hv_round_idx={getattr(seq, 'hv_round_idx', None)} "
                    f"len_draft_block_table={len(seq.draft_block_table)} "
                    f"draft_context_len={_dctx if is_draft else 'n/a'} "
                    f"context_len={context_len} "
                    f"logical_pos={logical_pos} "
                    f"required_blocks={required_blocks_now} "
                    f"decode_lookahead_hint={decode_lookahead_hint} "
                    f"required_blocks_with_lookahead={required_blocks_with_lookahead} "
                    f"block_table_len={len(block_table)} "
                    f"num_inter_cached_tokens={getattr(seq, 'num_inter_cached_tokens', None)} "
                    f"len_inter_block_table={len(getattr(seq, 'inter_block_table', []))}",
                    flush=True,
                )

            if is_draft:
                assert required_blocks_now <= len(block_table), (
                    "[HV_BLOCK_ASSERT:prepare_decode] "
                    f"seq_id={seq.seq_id} "
                    f"num_tokens={seq.num_tokens} "
                    f"num_draft_cached_tokens={seq.num_draft_cached_tokens} "
                    f"hv_num_provisional_tokens={seq.hv_num_provisional_tokens} "
                    f"context_len={context_len} "
                    f"logical_pos={logical_pos} "
                    f"block_idx={(logical_pos // block_size)} "
                    f"required_blocks={required_blocks_now} "
                    f"block_table_len={len(block_table)} "
                    f"block_table={block_table}"
                )
            else:
                assert len(seq) // block_size <= len(block_table), (
                    "in sync spec decode: committed tape spans more blocks than allocated"
                )

            input_ids.append(last_tok)
            positions.append(logical_pos)
            context_lens.append(context_len)

            block_idx = logical_pos // block_size
            pos_in_block = logical_pos % block_size
            assert block_idx < len(block_table), (
                "[HV_BLOCK_ASSERT:block_idx] "
                f"seq_id={seq.seq_id} "
                f"is_draft={is_draft} "
                f"is_intermediate={is_intermediate} "
                f"block_idx={block_idx} "
                f"block_table_len={len(block_table)} "
                f"context_len={context_len} "
                f"logical_pos={logical_pos} "
                f"num_draft_cached_tokens={seq.num_draft_cached_tokens} "
                f"hv_num_provisional_tokens={getattr(seq, 'hv_num_provisional_tokens', 0)} "
                f"num_inter_cached_tokens={getattr(seq, 'num_inter_cached_tokens', None)}"
            )
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

    ``context_lens`` must be the logical length ``c0 + L`` (committed + candidate), not
    ``len(seq)``: hierarchical target candidates can include ``hv_provisional_token_ids`` not
    present on ``token_ids`` while ``positions`` use ``c0 + j`` for ``j < L``.
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
        context_lens.append(c0 + L)
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


def prepare_verify_tensors_varlen_bucketed(
    seqs: list[Sequence],
    block_size: int,
    verify_tokens_per_seq: list[list[int]],
    bucket_q_len: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    list[int],
]:
    """Trailing bucket padding for hierarchical target verify CUDAGraph (fixed ``bucket_q_len`` per seq).

    Padding rows use ``cand[-1]`` and positions ``c0+j`` for ``j >= L``; verifier must use
    ``logits[:L]`` only. ``context_lens`` is ``c0 + bucket_q_len`` per sequence.
    """
    actual_lens: list[int] = []
    for cand in verify_tokens_per_seq:
        L = len(cand)
        assert L >= 1
        assert L <= bucket_q_len, (
            f"prepare_verify_tensors_varlen_bucketed: L={L} > bucket_q_len={bucket_q_len}"
        )
        actual_lens.append(L)

    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []

    for seq, cand in zip(seqs, verify_tokens_per_seq):
        c0 = seq.num_cached_tokens
        L = len(cand)
        context_lens.append(c0 + bucket_q_len)
        for j in range(bucket_q_len):
            tok = int(cand[j]) if j < L else int(cand[-1])
            pos = c0 + j
            input_ids.append(tok)
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
    q_lens = torch.full((len(seqs),), bucket_q_len, dtype=torch.int32, device=_dev)
    cu_seqlens_q[1:] = torch.cumsum(q_lens, dim=0)
    return input_ids_t, positions_t, slot_mapping_t, context_lens_t, cu_seqlens_q, bucket_q_len, actual_lens


def build_intermediate_verify_row(seq: Sequence, k: int) -> tuple[list[int], int, int]:
    """Per-sequence intermediate verify row: ``(full_tokens, base_pos, score_start)``."""
    c0 = seq.num_cached_tokens
    nic = seq.num_inter_cached_tokens
    pt = seq.hv_num_provisional_tokens

    if nic < c0:
        assert pt == 0, "intermediate verify gap path requires empty provisional tape"
        gap = list(seq.token_ids[nic:c0])
        tail = list(seq.token_ids[c0 : c0 + k + 1])
        full = gap + tail
        score_start = len(gap)
        base_pos = nic
    elif pt > 0:
        full = [seq.hv_provisional_token_ids[-1]] + list(seq.token_ids[c0 : c0 + k])
        score_start = 0
        base_pos = nic
    else:
        full = list(seq.token_ids[c0 : c0 + k + 1])
        score_start = 0
        base_pos = nic

    assert len(full) == score_start + k + 1, (
        f"intermediate verify: expected len={score_start + k + 1}, got {len(full)} "
        f"(nic={nic}, c0={c0}, pt={pt}, K={k})"
    )
    return full, base_pos, score_start


def prepare_intermediate_verify_gapaware_bucketed_tensors(
    seqs: list[Sequence],
    block_size: int,
    k: int,
    bucket_q_len: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    list[int],
    list[int],
    list[int],
]:
    """Trailing bucket padding for intermediate verify CUDAGraph (fixed ``bucket_q_len`` per seq).

    Scored slice remains ``score_start:score_start+k+1``; padding is only after the actual row.
    """
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    score_starts: list[int] = []
    packed_q_lens: list[int] = []
    actual_q_lens: list[int] = []

    for seq in seqs:
        full_actual, base_pos, score_start = build_intermediate_verify_row(seq, k)
        actual_q = len(full_actual)
        assert actual_q <= bucket_q_len, (
            f"intermediate verify bucket: actual_q={actual_q} > bucket_q_len={bucket_q_len}"
        )
        score_starts.append(score_start)
        packed_q_lens.append(bucket_q_len)
        actual_q_lens.append(actual_q)
        context_lens.append(base_pos + bucket_q_len)
        pad_tok = int(full_actual[-1])
        for j in range(bucket_q_len):
            tok = int(full_actual[j]) if j < actual_q else pad_tok
            pos = base_pos + j
            input_ids.append(tok)
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
    q_lens_t = torch.full((len(seqs),), bucket_q_len, dtype=torch.int32, device=_dev)
    cu_seqlens_q[1:] = torch.cumsum(q_lens_t, dim=0)
    return (
        input_ids_t,
        positions_t,
        slot_mapping_t,
        context_lens_t,
        cu_seqlens_q,
        bucket_q_len,
        score_starts,
        packed_q_lens,
        actual_q_lens,
    )


def prepare_intermediate_verify_gapaware_tensors(
    seqs: list[Sequence],
    block_size: int,
    k: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    list[int],
    list[int],
]:
    """Packed varlen intermediate verify: optional ``[nic:c0)`` gap + scored ``(K+1)`` tail.

    Per sequence:
    - If ``nic < c0`` (target committed ahead of intermediate KV): ``full = token_ids[nic:c0]
      + token_ids[c0:c0+k+1]``, ``score_start = c0 - nic`` (warmup rows only).
    - Elif provisional depth ``pt > 0``: ``full = [prov[-1]] + token_ids[c0:c0+k]``, ``score_start = 0``.
    - Else: ``full = token_ids[c0:c0+k+1]``, ``score_start = 0``.

    RoPE starts at ``nic``; ``context_lens`` is ``nic + len(full)`` per row pack. Verification
    uses logits rows ``[score_start : score_start + k + 1]`` only (same contract as the old
    fixed ``K+1`` path when ``score_start == 0``).
    """
    input_ids: list[int] = []
    positions: list[int] = []
    slot_mapping: list[int] = []
    context_lens: list[int] = []
    seqlen_q_list: list[int] = []
    score_starts: list[int] = []
    for seq in seqs:
        full, base_pos, score_start = build_intermediate_verify_row(seq, k)
        q_len = len(full)
        seqlen_q_list.append(q_len)
        score_starts.append(score_start)
        context_lens.append(base_pos + q_len)
        for j, tok in enumerate(full):
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
    q_lens_t = torch.tensor(seqlen_q_list, dtype=torch.int32, device=_dev)
    cu_seqlens_q[1:] = torch.cumsum(q_lens_t, dim=0)
    max_seqlen_q = max(seqlen_q_list) if seqlen_q_list else 0
    return (
        input_ids_t,
        positions_t,
        slot_mapping_t,
        context_lens_t,
        cu_seqlens_q,
        max_seqlen_q,
        score_starts,
        seqlen_q_list,
    )


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

        raw_start = num_cached_tokens + (skip_first_token if is_draft else 0)
        # Prefix cache can make ``raw_start >= seqlen``, so ``seq[raw_start:]`` would be empty and
        # prefill would run zero queries (breaks ``last_only`` lm_head). Re-score the last prompt
        # token once at its true position (do not apply draft ``skip_first_token`` position shift).
        if raw_start >= seqlen:
            start = max(seqlen - 1, 0)
            effective_pos_offset = 0
        else:
            start = raw_start
            effective_pos_offset = -skip_first_token if is_draft else 0
        input_ids.extend(seq[start:])
        positions.extend(list(range(start + effective_pos_offset, seqlen + effective_pos_offset)))
        seqlen_q = seqlen - start
        seqlen_k = seqlen + effective_pos_offset
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not block_table:  # first prefill
            continue

        # new: emit exactly one slot for each *new* token
        #    map each token index -> (block_id * block_size + offset)
        for pos in range(start + effective_pos_offset, seq.num_tokens + effective_pos_offset):
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
