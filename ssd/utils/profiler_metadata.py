"""Assemble per-request profiler JSON rows (metadata / cost_metadata).

Per-row ``num_draft`` / ``num_verification``: for decode steps, the profiler's
per-step batch counts (typically ``batch_size`` when each sequence runs one draft
and one verify in that step). Prefill rows use the same batch size for both.

Decode rows use ``step_id = profiler.decode_metadata_step_id()`` (1-based decode
ordinal). Prefill rows still use the global engine ``profiler.step_id`` so
prefill engine steps do not create gaps in decode ``step_id`` values.

Hierarchical intermediate rounds (``verification_model == "intermediate"``): the
verify trace's shared ``token_ids_per_position`` / ``token_confidence_per_position``
fields hold the **intermediate** model chain (length K+1), not the target verifier
chain. They are written as ``intermediate_verify_chain_*``; ``target_*`` columns
are null. Per-position accept/recovery/bonus for that round remain in ``inter_*``.

Target hierarchical rows may include ``inter_target_prefix_accept_len``: greedy
acceptance count restricted to candidate indices before the last ``K`` draft tail
tokens (``K = num_speculative_token``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def profile_greedy_token_confidence(logits_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy token id per slot and its probability under ``softmax(logits)`` (T=1).

    Used for profiler traces when ``temperature==0`` would otherwise use a one-hot
    ``target_probs_p_batched`` view, making ``max(dim=-1)`` always ``1.0``.
    """
    greedy = logits_p.argmax(dim=-1)
    probs = torch.softmax(logits_p.to(torch.float32), dim=-1)
    conf = probs.gather(2, greedy.unsqueeze(-1)).squeeze(-1)
    return greedy, conf


def draft_metadata_from_logits(
    logits_q: torch.Tensor,
    speculations: torch.Tensor,
    k: int,
) -> tuple[
    list[list[int]],
    list[list[float]],
    list[list[int]],
    list[list[float]],
]:
    """Draft fields: raw-softmax top-5 at position 0; chosen-token conf per draft position (k rows)."""
    device = logits_q.device
    b, kq, v = logits_q.shape
    assert kq == k, (kq, k)
    probs0 = F.softmax(logits_q[:, 0, :].float(), dim=-1)
    tk = min(5, v)
    top5 = torch.topk(probs0, k=tk, dim=-1)
    first_ids = top5.indices.cpu().tolist()
    first_conf = top5.values.cpu().tolist()
    pad = 5 - tk
    if pad > 0:
        first_ids = [row + [0] * pad for row in first_ids]
        first_conf = [row + [0.0] * pad for row in first_conf]

    probs_all = F.softmax(logits_q.float(), dim=-1)
    chosen = speculations[:, 1 : k + 1].to(device).long().clamp(0, v - 1)
    gather_idx = chosen.unsqueeze(-1)
    conf_pos = probs_all.gather(2, gather_idx).squeeze(-1).cpu().tolist()
    draft_ids = chosen.cpu().tolist()
    draft_ids_out = [[int(draft_ids[i][j]) for j in range(k)] for i in range(b)]
    conf_pos_out = [[float(conf_pos[i][j]) for j in range(k)] for i in range(b)]
    return first_ids, first_conf, draft_ids_out, conf_pos_out


def prefill_metadata_rows(
    *,
    profiler: Any,
    seqs: list[Any],
    speculate_k: int,
    spec_policy: str,
    draft_async: bool,
    cost_fields: bool,
) -> list[dict[str, Any]]:
    """One JSON row per request for a speculative prefill step (no decode logits)."""
    st = profiler.current_step_state
    B = len(seqs)
    step_wall = profiler.current_step_elapsed_s()
    draft_s = float(st.draft_time_s) if st is not None else 0.0
    ver_s = float(st.verification_time_s) if st is not None else 0.0
    sync_s = float(st.sync_time_s) if st is not None else 0.0
    nd, nv = B, B
    rows = []
    for bi, seq in enumerate(seqs):
        rows.append(
            trace_to_row_indexed(
                profiler=profiler,
                seq=seq,
                batch_index=bi,
                batch_size=B,
                is_prefill=True,
                speculate_k=speculate_k,
                spec_policy=spec_policy,
                draft_async=draft_async,
                cache_hit=None,
                trace=None,
                first_draft_token_ids=[0, 0, 0, 0, 0],
                first_draft_token_confidence=[0.0, 0.0, 0.0, 0.0, 0.0],
                draft_token_ids_per_position=[0] * speculate_k,
                draft_token_confidence_per_position=[0.0] * speculate_k,
                step_wall_time_s=step_wall,
                draft_time_s=draft_s,
                verification_time_s=ver_s,
                sync_time_s=sync_s,
                num_draft=nd,
                num_verification=nv,
                cost_fields=cost_fields,
            )
        )
    return rows


def trace_to_row_indexed(
    *,
    profiler: Any,
    seq: Any,
    batch_index: int,
    batch_size: int,
    is_prefill: bool,
    speculate_k: int,
    spec_policy: str,
    draft_async: bool,
    cache_hit: int | None,
    trace: Any | None,
    first_draft_token_ids: list[int],
    first_draft_token_confidence: list[float],
    draft_token_ids_per_position: list[int],
    draft_token_confidence_per_position: list[float],
    step_wall_time_s: float,
    draft_time_s: float,
    verification_time_s: float,
    sync_time_s: float,
    num_draft: int,
    num_verification: int,
    cost_fields: bool,
    hv_fused_subround_idx: int | None = None,
    hv_fused_engine_step_id: int | None = None,
) -> dict[str, Any]:
    inter_r, tgt_r = profiler.inter_target_counts_for_seq(seq.seq_id)
    # Decode JSONL uses a decode-only counter so prefill engine steps do not create gaps in ids.
    _sid = int(profiler.decode_metadata_step_id()) if not is_prefill else int(profiler.step_id)
    row: dict[str, Any] = {
        "step_id": _sid,
        "request_id": seq.seq_id,
        "batch_size": batch_size,
        "is_prefill": is_prefill,
        "spec_policy": spec_policy,
        "spec_mode": "async" if draft_async else "sync",
        "intermediate_verification_round": inter_r,
        "target_verification_round": tgt_r,
        "num_speculative_token": speculate_k,
        "first_draft_token_ids": first_draft_token_ids,
        "first_draft_token_confidence": first_draft_token_confidence,
        "draft_token_ids_per_position": draft_token_ids_per_position,
        "draft_token_confidence_per_position": draft_token_confidence_per_position,
        "cache_hit": cache_hit,
    }
    if trace is not None:
        i = batch_index
        vm = trace.verification_models[i]
        row["verification_model"] = vm
        # Hierarchical intermediate: VerifyProfileTrace shared fields are intermediate-model
        # K+1 chain, not target; do not map them into target_*.
        if vm == "intermediate":
            row["target_token_ids_per_position"] = None
            row["target_token_confidence_per_position"] = None
            row["target_accept_len"] = None
            row["target_recovery_token"] = None
            row["target_bonus_token"] = None
            row["inter_target_prefix_accept_len"] = None
            tid = trace.token_ids_per_position[i]
            tcf = trace.token_confidence_per_position[i]
            row["intermediate_verify_chain_token_ids_per_position"] = list(tid) if tid else None
            row["intermediate_verify_chain_token_confidence_per_position"] = (
                list(tcf) if tcf else None
            )
        else:
            row["intermediate_verify_chain_token_ids_per_position"] = None
            row["intermediate_verify_chain_token_confidence_per_position"] = None
            row["target_token_ids_per_position"] = (
                list(trace.token_ids_per_position[i]) if trace.token_ids_per_position[i] else None
            )
            row["target_token_confidence_per_position"] = (
                list(trace.token_confidence_per_position[i]) if trace.token_confidence_per_position[i] else None
            )
            al = trace.accept_len
            if al is not None and i < len(al):
                row["target_accept_len"] = al[i]
            else:
                row["target_accept_len"] = None
            row["target_recovery_token"] = trace.recovery_tokens[i]
            row["target_bonus_token"] = trace.bonus_tokens[i]
            itp = getattr(trace, "inter_target_prefix_accept_len", None)
            row["inter_target_prefix_accept_len"] = (
                int(itp[i]) if itp is not None and i < len(itp) else None
            )
        if trace.inter_token_ids_per_position is not None:
            row["inter_token_ids_per_position"] = trace.inter_token_ids_per_position[i]
            row["inter_token_confidence_per_position"] = trace.inter_token_confidence_per_position[i]
            row["inter_accept_len"] = trace.inter_accept_len[i]
            row["inter_recovery_token"] = trace.inter_recovery_token[i]
            row["inter_bonus_token"] = trace.inter_bonus_token[i]
        else:
            row["inter_token_ids_per_position"] = None
            row["inter_token_confidence_per_position"] = None
            row["inter_accept_len"] = None
            row["inter_recovery_token"] = None
            row["inter_bonus_token"] = None
        pcs = getattr(trace, "pivot_criteria_score", None)
        ptop1 = getattr(trace, "pivot_top1_prob", None)
        pres = getattr(trace, "pivot_residual_score", None)
        pexp = getattr(trace, "pivot_expanded", None)
        pbc = getattr(trace, "pivot_branch_count", None)
        psel = getattr(trace, "pivot_selected_branch_idx", None)
        psel_tok = getattr(trace, "pivot_selected_root_token_id", None)
        row["pivot_criteria_score"] = None if pcs is None else pcs[i]
        row["pivot_top1_prob"] = None if ptop1 is None else ptop1[i]
        row["pivot_residual_score"] = None if pres is None else pres[i]
        row["pivot_expanded"] = None if pexp is None else pexp[i]
        if pexp is None:
            row["pivot_expanded_request_count_in_step"] = None
            row["pivot_expanded_request_probability_in_step"] = None
        else:
            expanded_count = int(sum(1 for x in pexp if bool(x)))
            total_count = int(len(pexp))
            row["pivot_expanded_request_count_in_step"] = expanded_count
            row["pivot_expanded_request_probability_in_step"] = (
                float(expanded_count) / float(total_count) if total_count > 0 else 0.0
            )
        row["pivot_branch_count"] = None if pbc is None else pbc[i]
        row["pivot_selected_branch_idx"] = None if psel is None else psel[i]
        row["pivot_selected_root_token_id"] = None if psel_tok is None else psel_tok[i]
    else:
        row["verification_model"] = None
        row["target_token_ids_per_position"] = None
        row["target_token_confidence_per_position"] = None
        row["target_accept_len"] = None
        row["target_recovery_token"] = None
        row["target_bonus_token"] = None
        row["inter_target_prefix_accept_len"] = None
        row["intermediate_verify_chain_token_ids_per_position"] = None
        row["intermediate_verify_chain_token_confidence_per_position"] = None
        row["inter_token_ids_per_position"] = None
        row["inter_token_confidence_per_position"] = None
        row["inter_accept_len"] = None
        row["inter_recovery_token"] = None
        row["inter_bonus_token"] = None
        row["pivot_criteria_score"] = None
        row["pivot_top1_prob"] = None
        row["pivot_residual_score"] = None
        row["pivot_expanded"] = None
        row["pivot_expanded_request_count_in_step"] = None
        row["pivot_expanded_request_probability_in_step"] = None
        row["pivot_branch_count"] = None
        row["pivot_selected_branch_idx"] = None
        row["pivot_selected_root_token_id"] = None

    if cost_fields:
        row["step_wall_time_s"] = step_wall_time_s
        row["draft_time_s"] = draft_time_s
        row["verification_time_s"] = verification_time_s
        row["sync_time_s"] = sync_time_s
        row["num_draft"] = num_draft
        row["num_verification"] = num_verification
    if hv_fused_subround_idx is not None:
        row["hv_fused_subround_idx"] = int(hv_fused_subround_idx)
    if hv_fused_engine_step_id is not None:
        row["hv_fused_engine_step_id"] = int(hv_fused_engine_step_id)
    return row
