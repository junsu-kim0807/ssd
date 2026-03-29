#!/usr/bin/env python3
"""
Batch profile script for speculative decoding characteristics.

Methods:
- vanila: draft model drafts K tokens, then intermediate/target verify.
- bump: first draft token is produced by intermediate, remaining K-1 by draft model,
  then intermediate/target verify.

This script performs true tensor batching (padding + attention masks) across requests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from run_intermediate_verifier_profile import (
    DEFAULT_DRAFT,
    DEFAULT_INTERMEDIATE,
    DEFAULT_TARGET,
    _decode_generated_ids,
    _get_stop_token_ids,
    _get_model_input_device,
    _is_main_process,
    _maybe_barrier,
    _maybe_init_target_tp,
    _model_path,
    _rank0_print,
    _resolve_device_arg,
    _truncate_at_stop_token,
    _token_and_topk_probs_from_logits,
    build_model_run_dir,
    build_summary,
    compute_intermediate_precision_against_target,
    compute_position_stats,
    configure_reproducibility,
    count_profile_requests_for_record,
    get_dataset_max_new_tokens,
    get_dataset_records,
    get_tokenizer,
    make_metric_state,
    merge_metric_states,
    parse_csv_list,
    prepare_record_prompt_entries,
    resolve_prompt_style,
    save_summary,
)

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    raise ImportError("Install transformers: pip install transformers")


def _normalize_method(method: str) -> str:
    low = method.strip().lower()
    if low == "vanilla":
        return "vanila"
    if low not in {"vanila", "bump", "morphable"}:
        raise ValueError(f"Unsupported --method={method}. Use vanila, bump, or morphable.")
    return low


def _normalize_bonus_method(bonus_method: str) -> str:
    low = bonus_method.strip().lower()
    if low not in {"proactive", "conservative", "adaptive"}:
        raise ValueError(f"Unsupported --bonus-method={bonus_method}. Use proactive, conservative, or adaptive.")
    return low


def _should_include_bonus_recovery(
    *,
    bonus_method: str,
    bonus_threshold: float,
    recovery_confidence: float,
) -> bool:
    if bonus_method == "proactive":
        return True
    if bonus_method == "conservative":
        return False
    return recovery_confidence >= bonus_threshold


def _pad_sequences(seqs: list[list[int]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int], int]:
    if not seqs:
        raise ValueError("seqs must be non-empty")
    lengths = [len(x) for x in seqs]
    max_len = max(lengths)
    input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        n = len(seq)
        input_ids[i, :n] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
    return input_ids, attention_mask, lengths, max_len


def _batched_next_logits(model, seqs: list[list[int]], device_fallback: str) -> tuple[torch.Tensor, int, int]:
    model_device = _get_model_input_device(model, device_fallback)
    input_ids, attention_mask, lengths, max_len = _pad_sequences(seqs, device=model_device)
    with torch.inference_mode():
        if input_ids.shape[0] > 1:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        else:
            out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = out.logits
    row_idx = torch.arange(logits.size(0), device=logits.device)
    col_idx = torch.tensor([n - 1 for n in lengths], dtype=torch.long, device=logits.device)
    next_logits = logits[row_idx, col_idx]
    total_tokens = sum(lengths)
    total_slots = len(lengths) * max_len
    return next_logits, total_tokens, total_slots


def _sequential_next_logits(model, seqs: list[list[int]], device_fallback: str) -> tuple[torch.Tensor, int, int]:
    if not seqs:
        raise ValueError("seqs must be non-empty")
    model_device = _get_model_input_device(model, device_fallback)
    logits_rows: list[torch.Tensor] = []
    total_tokens = 0
    total_slots = 0
    with torch.inference_mode():
        for seq in seqs:
            inp = torch.tensor([seq], dtype=torch.long, device=model_device)
            attn = torch.ones_like(inp, dtype=torch.long, device=model_device)
            out = model(input_ids=inp, attention_mask=attn, use_cache=False, return_dict=True)
            logits_rows.append(out.logits[0, len(seq) - 1])
            total_tokens += len(seq)
            total_slots += len(seq)
    return torch.stack(logits_rows, dim=0), total_tokens, total_slots


def _batched_logits_at_positions(
    model,
    seqs: list[list[int]],
    positions_per_sample: list[list[int]],
    device_fallback: str,
) -> tuple[torch.Tensor, int, int]:
    if len(seqs) != len(positions_per_sample):
        raise ValueError("seqs and positions_per_sample must have same batch size")
    if not seqs:
        raise ValueError("seqs must be non-empty")
    each_len = [len(x) for x in positions_per_sample]
    if len(set(each_len)) != 1:
        raise ValueError("positions_per_sample must have the same length for all samples")

    model_device = _get_model_input_device(model, device_fallback)
    input_ids, attention_mask, lengths, max_len = _pad_sequences(seqs, device=model_device)
    with torch.inference_mode():
        if input_ids.shape[0] > 1:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        else:
            out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = out.logits

    bsz = len(seqs)
    num_pos = each_len[0]
    pos_tensor = torch.tensor(positions_per_sample, dtype=torch.long, device=logits.device)
    row_idx = torch.arange(bsz, device=logits.device).unsqueeze(1).expand(bsz, num_pos)
    gathered = logits[row_idx, pos_tensor]
    total_tokens = sum(lengths)
    total_slots = len(lengths) * max_len
    return gathered, total_tokens, total_slots


def _batched_logits_at_positions_padded(
    model,
    seqs: list[list[int]],
    positions_per_sample: list[list[int]],
    device_fallback: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if len(seqs) != len(positions_per_sample):
        raise ValueError("seqs and positions_per_sample must have same batch size")
    if not seqs:
        raise ValueError("seqs must be non-empty")

    model_device = _get_model_input_device(model, device_fallback)
    input_ids, attention_mask, lengths, max_len = _pad_sequences(seqs, device=model_device)
    with torch.inference_mode():
        if input_ids.shape[0] > 1:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        else:
            out = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = out.logits

    bsz = len(seqs)
    max_pos_len = max(len(x) for x in positions_per_sample)
    pos_tensor = torch.zeros((bsz, max_pos_len), dtype=torch.long, device=logits.device)
    valid_mask = torch.zeros((bsz, max_pos_len), dtype=torch.bool, device=logits.device)
    for i, pos_list in enumerate(positions_per_sample):
        if not pos_list:
            continue
        n = len(pos_list)
        pos_tensor[i, :n] = torch.tensor(pos_list, dtype=torch.long, device=logits.device)
        valid_mask[i, :n] = True

    row_idx = torch.arange(bsz, device=logits.device).unsqueeze(1).expand(bsz, max_pos_len)
    gathered = logits[row_idx, pos_tensor]
    total_tokens = sum(lengths)
    total_slots = len(lengths) * max_len
    return gathered, valid_mask, total_tokens, total_slots


def _sequential_logits_at_positions_padded(
    model,
    seqs: list[list[int]],
    positions_per_sample: list[list[int]],
    device_fallback: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if len(seqs) != len(positions_per_sample):
        raise ValueError("seqs and positions_per_sample must have same batch size")
    if not seqs:
        raise ValueError("seqs must be non-empty")

    model_device = _get_model_input_device(model, device_fallback)
    bsz = len(seqs)
    max_pos_len = max(len(x) for x in positions_per_sample)
    gathered_rows: list[torch.Tensor] = []
    valid_mask = torch.zeros((bsz, max_pos_len), dtype=torch.bool)
    total_tokens = 0
    total_slots = 0

    with torch.inference_mode():
        for i, (seq, pos_list) in enumerate(zip(seqs, positions_per_sample)):
            inp = torch.tensor([seq], dtype=torch.long, device=model_device)
            attn = torch.ones_like(inp, dtype=torch.long, device=model_device)
            out = model(input_ids=inp, attention_mask=attn, use_cache=False, return_dict=True)
            logits = out.logits[0]
            row = torch.zeros((max_pos_len, logits.size(-1)), dtype=logits.dtype, device=logits.device)
            if pos_list:
                pos_tensor = torch.tensor(pos_list, dtype=torch.long, device=logits.device)
                n = len(pos_list)
                row[:n] = logits[pos_tensor]
                valid_mask[i, :n] = True
            gathered_rows.append(row)
            total_tokens += len(seq)
            total_slots += len(seq)

    gathered = torch.stack(gathered_rows, dim=0)
    return gathered, valid_mask.to(device=gathered.device), total_tokens, total_slots


def _topk_info(logits_1d: torch.Tensor, k: int = 5) -> dict[str, Any]:
    probs = torch.softmax(logits_1d.float(), dim=-1)
    vals, idx = torch.topk(probs, k=min(k, probs.numel()))
    return {
        "top1_token_id": int(idx[0].item()),
        "top1_prob": float(vals[0].item()),
        "top5_token_ids": [int(x) for x in idx.cpu().tolist()],
        "top5_probs": [float(x) for x in vals.cpu().tolist()],
    }


def run_one_verify_round_batch(
    *,
    method: str,
    draft_model,
    inter_model,
    target_model,
    prompt_ids_batch: list[list[int]],
    recovery_token_ids: list[int],
    k: int,
    device_draft: str,
    device_inter: str,
    device_target: str,
    confidence_threshold: float,
    force_target_batch: list[bool],
) -> dict[str, Any]:
    bsz = len(prompt_ids_batch)
    seqs = [list(prompt_ids_batch[i]) + [int(recovery_token_ids[i])] for i in range(bsz)]
    draft_tokens_batch: list[list[int]] = [[] for _ in range(bsz)]
    draft_step_stats_batch: list[list[dict[str, Any]]] = [[] for _ in range(bsz)]
    draft_confidence_batch: list[list[float]] = [[] for _ in range(bsz)]
    draft_confidence_cumprod_batch: list[list[float]] = [[] for _ in range(bsz)]
    threshold_cross_position_batch: list[int | None] = [None for _ in range(bsz)]
    draft_confidence_cumprod_avg_per_position: list[float] = []
    verification_model_batch: list[str] = ["target" for _ in range(bsz)]
    forced_target_by_interval_batch: list[bool] = [bool(x) for x in force_target_batch]

    draft_total_tokens = 0
    draft_total_slots = 0
    draft_model_tokens = 0
    draft_model_slots = 0
    inter_draft_tokens = 0
    inter_draft_slots = 0
    cumprod = [1.0 for _ in range(bsz)]
    threshold_cross_position_round: int | None = None
    for step_idx in range(k):
        if method in {"vanila", "bump"}:
            use_inter = method == "bump" and step_idx == 0
            current_model = inter_model if use_inter else draft_model
            current_device = device_inter if use_inter else device_draft
            next_logits, token_count, slot_count = _batched_next_logits(current_model, seqs, current_device)
            draft_total_tokens += token_count
            draft_total_slots += slot_count
            if use_inter:
                inter_draft_tokens += token_count
                inter_draft_slots += slot_count
            else:
                draft_model_tokens += token_count
                draft_model_slots += slot_count
            draft_logits = next_logits if not use_inter else None
            inter_logits = next_logits if use_inter else None
        else:
            draft_logits, token_count, slot_count = _batched_next_logits(draft_model, seqs, device_draft)
            draft_total_tokens += token_count
            draft_total_slots += slot_count
            draft_model_tokens += token_count
            draft_model_slots += slot_count
            inter_logits, inter_draft_tc, inter_draft_sc = _batched_next_logits(inter_model, seqs, device_inter)
            inter_draft_tokens += inter_draft_tc
            inter_draft_slots += inter_draft_sc

        next_tokens: list[int] = []
        for i in range(bsz):
            use_inter_for_token = False
            draft_conf = None
            if method == "morphable":
                draft_probs = torch.softmax(draft_logits[i].float(), dim=-1)
                draft_conf = float(draft_probs.max().item())
                cumprod[i] *= draft_conf
                draft_confidence_batch[i].append(draft_conf)
                draft_confidence_cumprod_batch[i].append(float(cumprod[i]))
            else:
                use_inter_for_token = method == "bump" and step_idx == 0
                if draft_logits is not None:
                    draft_probs = torch.softmax(draft_logits[i].float(), dim=-1)
                    draft_conf = float(draft_probs.max().item())
                    cumprod[i] *= draft_conf
                else:
                    draft_conf = None

            if method == "morphable":
                # morphable thresholding is determined by batch-average cumulative confidence.
                use_inter_for_token = (
                    threshold_cross_position_round is not None and step_idx >= threshold_cross_position_round
                )
            selected_logits = inter_logits[i] if use_inter_for_token else draft_logits[i]
            next_tok = int(selected_logits.argmax(dim=-1).item())
            next_tokens.append(next_tok)
            prob_info = _topk_info(selected_logits, k=5)
            draft_tokens_batch[i].append(next_tok)
            draft_step_stats_batch[i].append(
                {
                    "position": step_idx,
                    "draft_token_id": next_tok,
                    "draft_top1_prob": prob_info["top1_prob"],
                    "draft_top5_token_ids": prob_info["top5_token_ids"],
                    "draft_top5_probs": prob_info["top5_probs"],
                    "draft_source_model": "intermediate" if use_inter_for_token else "draft",
                    "draft_model_top1_confidence": draft_conf,
                    "draft_confidence_cumprod": float(cumprod[i]),
                }
            )
            seqs[i].append(next_tok)
        if method == "morphable":
            avg_cumprod = float(sum(cumprod) / max(len(cumprod), 1))
            draft_confidence_cumprod_avg_per_position.append(avg_cumprod)
            if threshold_cross_position_round is None and avg_cumprod < confidence_threshold:
                threshold_cross_position_round = step_idx

    if method == "morphable":
        threshold_cross_position_batch = [threshold_cross_position_round for _ in range(bsz)]

    full_seqs = [list(prompt_ids_batch[i]) + [int(recovery_token_ids[i])] + draft_tokens_batch[i] for i in range(bsz)]
    positions_per_sample = [list(range(len(prompt_ids_batch[i]) - 1, len(prompt_ids_batch[i]) + k + 1)) for i in range(bsz)]
    logits_inter, inter_total_tokens, inter_total_slots = _batched_logits_at_positions(
        inter_model,
        full_seqs,
        positions_per_sample,
        device_inter,
    )
    logits_target, target_total_tokens, target_total_slots = _batched_logits_at_positions(
        target_model,
        full_seqs,
        positions_per_sample,
        device_target,
    )

    if method == "morphable":
        for i in range(bsz):
            forced_target = forced_target_by_interval_batch[i]
            if forced_target:
                verification_model_batch[i] = "target"
            elif threshold_cross_position_batch[i] is None:
                verification_model_batch[i] = "intermediate"
            else:
                verification_model_batch[i] = "target"
    else:
        verification_model_batch = ["target" for _ in range(bsz)]

    return {
        "draft_tokens_batch": draft_tokens_batch,
        "draft_step_stats_batch": draft_step_stats_batch,
        "draft_confidence_batch": draft_confidence_batch,
        "draft_confidence_cumprod_batch": draft_confidence_cumprod_batch,
        "threshold_cross_position_batch": threshold_cross_position_batch,
        "draft_confidence_cumprod_avg_per_position": draft_confidence_cumprod_avg_per_position,
        "verification_model_batch": verification_model_batch,
        "forced_target_by_interval_batch": forced_target_by_interval_batch,
        "logits_inter": logits_inter,
        "logits_target": logits_target,
        "batch_characteristics": {
            "round_batch_size": bsz,
            "draft_tokens_computed": draft_total_tokens,
            "draft_padded_slots": draft_total_slots,
            "inter_tokens_computed": inter_total_tokens,
            "inter_padded_slots": inter_total_slots,
            "target_tokens_computed": target_total_tokens,
            "target_padded_slots": target_total_slots,
            "draft_model_draft_tokens": draft_model_tokens,
            "draft_model_draft_slots": draft_model_slots,
            "inter_model_draft_tokens": inter_draft_tokens,
            "inter_model_draft_slots": inter_draft_slots,
            "inter_model_verify_tokens": inter_total_tokens,
            "inter_model_verify_slots": inter_total_slots,
            "target_model_verify_tokens": target_total_tokens,
            "target_model_verify_slots": target_total_slots,
        },
    }


def _build_requests(
    *,
    dataset_key: str,
    prompt_records: list[dict[str, Any]],
    args,
    tokenizer,
    target_model,
) -> list[dict[str, Any]]:
    prompt_style = resolve_prompt_style(args.prompt_style, args.target)
    requests: list[dict[str, Any]] = []
    for record in prompt_records:
        sample_id = record["sample_id"]
        max_new_tokens = get_dataset_max_new_tokens(
            dataset_key,
            args.alpaca_max_new_tokens,
            args.gsm8k_max_new_tokens,
            args.humaneval_max_new_tokens,
            args.mt_bench_max_new_tokens,
            args.qa_max_new_tokens,
        )
        prompt_entries = prepare_record_prompt_entries(
            tokenizer,
            target_model,
            args.device_target,
            record,
            prompt_style=prompt_style,
            max_prompt_tokens=args.max_prompt_tokens,
            use_chat_template=args.chat_template,
            conversation_turn_index=args.conversation_turn_index,
            conversation_turn_mode=args.conversation_turn_mode,
            history_max_new_tokens=max_new_tokens,
        )
        for entry in prompt_entries:
            prompt_ids = entry["prompt_ids"]
            prompt_meta = entry["meta"]
            if not prompt_ids:
                continue
            turn_index = int(prompt_meta["turn_index_used"])
            request_id = f"{sample_id}:turn_{turn_index}" if prompt_meta["num_turns_in_record"] > 1 else str(sample_id)
            requests.append(
                {
                    "dataset": dataset_key,
                    "request_sample_id": sample_id,
                    "request_id": request_id,
                    "record": record,
                    "prompt_ids": list(prompt_ids),
                    "prompt_meta": prompt_meta,
                    "turn_index": turn_index,
                    "max_new_tokens": max_new_tokens,
                }
            )
    return requests


def _batch_stats_from_round_chars(round_chars: dict[str, int]) -> dict[str, float | int]:
    draft_slots = max(round_chars["draft_padded_slots"], 1)
    inter_slots = max(round_chars["inter_padded_slots"], 1)
    target_slots = max(round_chars["target_padded_slots"], 1)
    draft_model_draft_slots = max(round_chars["draft_model_draft_slots"], 1)
    inter_model_draft_slots = max(round_chars["inter_model_draft_slots"], 1)
    inter_model_verify_slots = max(round_chars["inter_model_verify_slots"], 1)
    target_model_verify_slots = max(round_chars["target_model_verify_slots"], 1)
    return {
        **round_chars,
        "draft_utilization": float(round_chars["draft_tokens_computed"] / draft_slots),
        "inter_utilization": float(round_chars["inter_tokens_computed"] / inter_slots),
        "target_utilization": float(round_chars["target_tokens_computed"] / target_slots),
        "draft_model_draft_utilization": float(round_chars["draft_model_draft_tokens"] / draft_model_draft_slots),
        "inter_model_draft_utilization": float(round_chars["inter_model_draft_tokens"] / inter_model_draft_slots),
        "inter_model_verify_utilization": float(round_chars["inter_model_verify_tokens"] / inter_model_verify_slots),
        "target_model_verify_utilization": float(round_chars["target_model_verify_tokens"] / target_model_verify_slots),
    }


def run_dataset_profile_batch(
    *,
    dataset_key: str,
    prompt_records: list[dict[str, Any]],
    args,
    data_dir: Path,
    tokenizer,
    draft_model,
    inter_model,
    target_model,
    starting_overall_round: int,
) -> tuple[dict[str, Any], dict[str, Any], int, Path, Path | None, Path | None]:
    dataset_out_dir = Path(args.output_dir) / dataset_key
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    verification_jsonl_path = dataset_out_dir / args.verification_jsonl
    detail_path = dataset_out_dir / "per_position_top5_detail.jsonl" if args.save_per_position_detail else None
    request_metadata_path = dataset_out_dir / "request_metadata.jsonl"
    response_path = dataset_out_dir / "response.jsonl"

    state = make_metric_state()
    state["batch_sizes"] = []
    state["batch_draft_utilization"] = []
    state["batch_inter_utilization"] = []
    state["batch_target_utilization"] = []
    state["batch_draft_model_draft_utilization"] = []
    state["batch_inter_model_draft_utilization"] = []
    state["batch_inter_model_verify_utilization"] = []
    state["batch_target_model_verify_utilization"] = []
    state["total_draft_model_draft_tokens"] = 0
    state["total_inter_model_draft_tokens"] = 0
    state["total_inter_model_verify_tokens"] = 0
    state["total_target_model_verify_tokens"] = 0
    state["verification_model_target_rounds"] = 0
    state["verification_model_intermediate_rounds"] = 0
    state["interval_forced_target_rounds"] = 0
    state["bonus_recovery_included_count"] = 0
    state["bonus_recovery_skipped_count"] = 0
    stop_ids = _get_stop_token_ids(tokenizer)

    overall_round = starting_overall_round
    requests = _build_requests(
        dataset_key=dataset_key,
        prompt_records=prompt_records,
        args=args,
        tokenizer=tokenizer,
        target_model=target_model,
    )
    total_requests = len(requests)

    verification_f = verification_jsonl_path.open("w", encoding="utf-8") if _is_main_process() else None
    detail_f = detail_path.open("w", encoding="utf-8") if (_is_main_process() and detail_path is not None) else None
    request_meta_f = request_metadata_path.open("w", encoding="utf-8") if _is_main_process() else None
    response_f = response_path.open("w", encoding="utf-8") if _is_main_process() else None
    try:
        request_counter = 0
        for chunk_start in range(0, total_requests, args.batch_size):
            batch_requests = requests[chunk_start : chunk_start + args.batch_size]
            if not batch_requests:
                continue

            initial_prompts = [r["prompt_ids"] for r in batch_requests]
            initial_logits, _, _ = _sequential_next_logits(target_model, initial_prompts, args.device_target)
            current_recoveries = [int(initial_logits[i].argmax(dim=-1).item()) for i in range(len(batch_requests))]

            batch_states: list[dict[str, Any]] = []
            for i, req in enumerate(batch_requests):
                initial_context_tokens = len(req["prompt_ids"])
                state["request_count"] += 1
                state["initial_context_tokens"].append(initial_context_tokens)
                batch_states.append(
                    {
                        "request": req,
                        "request_index": request_counter + i,
                        "prompt_ids_for_round": list(req["prompt_ids"]),
                        "current_recovery": current_recoveries[i],
                        "pending_inter_recovery": None,
                        "verification_counter_since_target": 0,
                        "carry_over_prefix_tokens": [],
                        "generated_count": 0,
                        "generated_output_ids": [],
                        "request_round_idx": 0,
                        "max_context_seen": initial_context_tokens,
                        "done": False,
                    }
                )
                if args.print_request_progress:
                    _rank0_print(
                        f"[{dataset_key}] request {request_counter + i + 1}/{total_requests} sample={req['request_id']} initial_ctx={initial_context_tokens} max_new={req['max_new_tokens']} method={args.method}",
                        flush=True,
                    )

            while True:
                active_states = [s for s in batch_states if (not s["done"]) and s["generated_count"] < s["request"]["max_new_tokens"]]
                if not active_states:
                    break

                prompt_batch = [list(s["prompt_ids_for_round"]) for s in active_states]
                recovery_batch = [int(s["current_recovery"]) for s in active_states]
                force_target_batch = [
                    bool(args.method == "morphable" and args.interval > 0 and s["verification_counter_since_target"] == args.interval)
                    for s in active_states
                ]
                round_out = run_one_verify_round_batch(
                    method=args.method,
                    draft_model=draft_model,
                    inter_model=inter_model,
                    target_model=target_model,
                    prompt_ids_batch=prompt_batch,
                    recovery_token_ids=recovery_batch,
                    k=args.k,
                    device_draft=args.device_draft,
                    device_inter=args.device_intermediate,
                    device_target=args.device_target,
                    confidence_threshold=args.confidence_threshold,
                    force_target_batch=force_target_batch,
                )
                round_chars = _batch_stats_from_round_chars(round_out["batch_characteristics"])
                state["batch_sizes"].append(int(round_chars["round_batch_size"]))
                state["batch_draft_utilization"].append(float(round_chars["draft_utilization"]))
                state["batch_inter_utilization"].append(float(round_chars["inter_utilization"]))
                state["batch_target_utilization"].append(float(round_chars["target_utilization"]))
                state["batch_draft_model_draft_utilization"].append(float(round_chars["draft_model_draft_utilization"]))
                state["batch_inter_model_draft_utilization"].append(float(round_chars["inter_model_draft_utilization"]))
                state["batch_inter_model_verify_utilization"].append(float(round_chars["inter_model_verify_utilization"]))
                state["batch_target_model_verify_utilization"].append(float(round_chars["target_model_verify_utilization"]))
                state["total_draft_model_draft_tokens"] += int(round_chars["draft_model_draft_tokens"])
                state["total_inter_model_draft_tokens"] += int(round_chars["inter_model_draft_tokens"])
                state["total_inter_model_verify_tokens"] += int(round_chars["inter_model_verify_tokens"])
                state["total_target_model_verify_tokens"] += int(round_chars["target_model_verify_tokens"])

                draft_tokens_batch = round_out["draft_tokens_batch"]
                draft_step_stats_batch = round_out["draft_step_stats_batch"]
                draft_confidence_batch = round_out["draft_confidence_batch"]
                draft_confidence_cumprod_batch = round_out["draft_confidence_cumprod_batch"]
                draft_confidence_cumprod_avg_per_position = round_out["draft_confidence_cumprod_avg_per_position"]
                threshold_cross_position_batch = round_out["threshold_cross_position_batch"]
                verification_model_batch = round_out["verification_model_batch"]
                forced_target_by_interval_batch = round_out["forced_target_by_interval_batch"]
                logits_inter = round_out["logits_inter"]
                logits_target = round_out["logits_target"]

                carry_target_indices = [
                    i
                    for i, s in enumerate(active_states)
                    if verification_model_batch[i] == "target" and bool(s["carry_over_prefix_tokens"])
                ]
                carry_logits_inter_by_idx: dict[int, torch.Tensor] = {}
                carry_logits_target_by_idx: dict[int, torch.Tensor] = {}
                if carry_target_indices:
                    carry_prompt_batch: list[list[int]] = []
                    carry_recovery_batch: list[int] = []
                    carry_candidate_batch: list[list[int]] = []
                    carry_positions_batch: list[list[int]] = []
                    for i in carry_target_indices:
                        s = active_states[i]
                        carry_prefix_tokens = list(s["carry_over_prefix_tokens"])
                        raw_draft_tokens = draft_tokens_batch[i]
                        eval_draft_tokens = carry_prefix_tokens + raw_draft_tokens
                        base_prompt = list(s["prompt_ids_for_round"])
                        if len(carry_prefix_tokens) <= len(base_prompt):
                            base_prompt = base_prompt[: len(base_prompt) - len(carry_prefix_tokens)]
                        carry_prompt_batch.append(base_prompt)
                        carry_recovery_batch.append(int(s["current_recovery"]))
                        carry_candidate_batch.append(eval_draft_tokens)
                        carry_positions_batch.append(list(range(len(base_prompt) - 1, len(base_prompt) + len(eval_draft_tokens) + 1)))

                    carry_full_seqs = [
                        list(carry_prompt_batch[j]) + [int(carry_recovery_batch[j])] + list(carry_candidate_batch[j])
                        for j in range(len(carry_prompt_batch))
                    ]
                    carry_logits_inter, _, _, _ = _batched_logits_at_positions_padded(
                        inter_model,
                        carry_full_seqs,
                        carry_positions_batch,
                        args.device_intermediate,
                    )
                    carry_logits_target, _, _, _ = _batched_logits_at_positions_padded(
                        target_model,
                        carry_full_seqs,
                        carry_positions_batch,
                        args.device_target,
                    )
                    for local_idx, global_idx in enumerate(carry_target_indices):
                        valid_len = len(carry_positions_batch[local_idx])
                        carry_logits_inter_by_idx[global_idx] = carry_logits_inter[local_idx : local_idx + 1, :valid_len]
                        carry_logits_target_by_idx[global_idx] = carry_logits_target[local_idx : local_idx + 1, :valid_len]

                for i, s in enumerate(active_states):
                    req = s["request"]
                    raw_draft_tokens = draft_tokens_batch[i]
                    verification_model = verification_model_batch[i]
                    forced_target_by_interval = forced_target_by_interval_batch[i]
                    threshold_cross_position = threshold_cross_position_batch[i]
                    carry_prefix_tokens = list(s["carry_over_prefix_tokens"])

                    if verification_model == "target" and carry_prefix_tokens:
                        eval_draft_tokens = carry_prefix_tokens + raw_draft_tokens
                        logits_inter_eval = carry_logits_inter_by_idx[i]
                        logits_target_eval = carry_logits_target_by_idx[i]
                    else:
                        eval_draft_tokens = raw_draft_tokens
                        logits_inter_eval = logits_inter[i : i + 1]
                        logits_target_eval = logits_target[i : i + 1]

                    (
                        inter_topk_list,
                        target_topk_list,
                        accept_target_list,
                        accept_inter_list,
                        draft_tok_list,
                    ) = compute_position_stats(
                        eval_draft_tokens,
                        logits_inter_eval,
                        logits_target_eval,
                        args.topk,
                    )
                    target_prefix_accept = [1 if all(accept_target_list[: j + 1]) else 0 for j in range(len(eval_draft_tokens))]
                    inter_prefix_accept = [1 if all(accept_inter_list[: j + 1]) else 0 for j in range(len(eval_draft_tokens))]
                    for j in range(len(eval_draft_tokens)):
                        state["position_accept_target"][j].append(target_prefix_accept[j])
                        state["position_accept_inter"][j].append(inter_prefix_accept[j])

                    n_accept_target = next((idx for idx, v in enumerate(accept_target_list) if v == 0), len(accept_target_list))
                    n_accept_inter = next((idx for idx, v in enumerate(accept_inter_list) if v == 0), len(accept_inter_list))
                    state["accept_len_target_list"].append(n_accept_target)
                    state["accept_len_inter_list"].append(n_accept_inter)
                    round_accept_rate_target = n_accept_target / len(eval_draft_tokens) if eval_draft_tokens else 0.0
                    round_accept_rate_inter = n_accept_inter / len(eval_draft_tokens) if eval_draft_tokens else 0.0

                    if n_accept_target < len(eval_draft_tokens):
                        target_recovery = int(logits_target_eval[0, n_accept_target + 1].argmax(dim=-1).item())
                    else:
                        target_recovery = int(logits_target_eval[0, len(eval_draft_tokens) + 1].argmax(dim=-1).item())
                    if n_accept_inter < len(eval_draft_tokens):
                        inter_recovery = int(logits_inter_eval[0, n_accept_inter + 1].argmax(dim=-1).item())
                    else:
                        inter_recovery = int(logits_inter_eval[0, len(eval_draft_tokens) + 1].argmax(dim=-1).item())

                    target_accepted_token_stats: list[dict[str, Any]] = []
                    for j in range(n_accept_target):
                        token_stats = _token_and_topk_probs_from_logits(
                            logits_target_eval[0, j + 1],
                            eval_draft_tokens[j],
                            topk=args.topk,
                        )
                        token_stats["position"] = j
                        target_accepted_token_stats.append(token_stats)

                    target_recovery_logits_idx = (n_accept_target + 1) if n_accept_target < len(eval_draft_tokens) else len(eval_draft_tokens) + 1
                    target_recovery_stats = _token_and_topk_probs_from_logits(
                        logits_target_eval[0, target_recovery_logits_idx],
                        target_recovery,
                        topk=args.topk,
                    )
                    target_recovery_stats["position"] = target_recovery_logits_idx - 1

                    precision_stats = compute_intermediate_precision_against_target(
                        draft_tokens=eval_draft_tokens,
                        n_accept_inter=n_accept_inter,
                        n_accept_target=n_accept_target,
                        target_recovery=target_recovery,
                    )
                    state["inter_precision_tp_total"] += int(precision_stats["true_positive_count"])
                    state["inter_precision_fp_total"] += int(precision_stats["false_positive_count"])
                    if precision_stats["precision"] is not None:
                        state["inter_precision_round_values"].append(float(precision_stats["precision"]))
                        state["inter_precision_nonempty_rounds"] += 1

                    context_tokens_before_round = len(s["prompt_ids_for_round"]) + 1
                    s["max_context_seen"] = max(s["max_context_seen"], context_tokens_before_round)
                    dataset_global_round = state["total_rounds"]
                    round_idx_for_row = s["request_round_idx"]
                    round_sample_id = f"{dataset_key}:{req['request_id']}:round_{round_idx_for_row}"

                    verification_row = {
                        "request_index": s["request_index"],
                        "request_sample_id": req["request_sample_id"],
                        "sample_id": round_sample_id,
                        "request_id": req["request_id"],
                        "dataset": dataset_key,
                        "method": args.method,
                        "batch_characteristics": round_chars,
                        "prompt_style": req["prompt_meta"]["prompt_style"],
                        "turn_index_used": req["turn_index"],
                        "num_turns_in_record": req["prompt_meta"]["num_turns_in_record"],
                        "history_mode": req["prompt_meta"]["history_mode"],
                        "history_assistant_turns_generated": req["prompt_meta"]["history_assistant_turns_generated"],
                        "verification_round": round_idx_for_row,
                        "context_tokens_initial": len(req["prompt_ids"]),
                        "context_tokens_before_round": context_tokens_before_round,
                        "dataset_global_round": dataset_global_round,
                        "overall_global_round": overall_round,
                        "k": len(eval_draft_tokens),
                        "verification_model": verification_model,
                        "forced_target_by_interval": bool(forced_target_by_interval),
                        "confidence_threshold": args.confidence_threshold if args.method == "morphable" else None,
                        "threshold_cross_position": threshold_cross_position,
                        "draft_confidence_per_position": draft_confidence_batch[i],
                        "draft_confidence_cumprod_per_position": draft_confidence_cumprod_batch[i],
                        "draft_confidence_cumprod_avg_per_position": draft_confidence_cumprod_avg_per_position,
                        "bonus_method": args.bonus_method if args.method == "morphable" else None,
                        "bonus_threshold": args.bonus_threshold if args.method == "morphable" else None,
                        "target": {
                            "acceptance_per_position_raw": accept_target_list,
                            "acceptance_per_position_prefix": target_prefix_accept,
                            "acceptance_rate": round_accept_rate_target,
                            "acceptance_length": n_accept_target,
                            "recovery_token_id": target_recovery,
                            "accepted_token_stats": target_accepted_token_stats,
                            "recovery_token_stats": target_recovery_stats,
                        },
                        "intermediate": {
                            "acceptance_per_position_raw": accept_inter_list,
                            "acceptance_per_position_prefix": inter_prefix_accept,
                            "acceptance_rate": round_accept_rate_inter,
                            "acceptance_length": n_accept_inter,
                            "recovery_token_id": inter_recovery,
                            "precision_vs_target": precision_stats,
                        },
                    }
                    if n_accept_target == n_accept_inter:
                        state["same_accept_len_count"] += 1
                        if target_recovery == inter_recovery:
                            state["same_accept_len_bonus_same_count"] += 1
                    else:
                        state["diff_accept_len_count"] += 1

                    if detail_f is not None:
                        for j, (dt, itopk, ttopk, at, ai) in enumerate(
                            zip(draft_tok_list, inter_topk_list, target_topk_list, accept_target_list, accept_inter_list)
                        ):
                            carry_len = max(len(eval_draft_tokens) - len(raw_draft_tokens), 0)
                            if j < carry_len:
                                step_stat = {
                                    "draft_source_model": "carry_over_prefix",
                                    "draft_top1_prob": None,
                                    "draft_top5_token_ids": [],
                                    "draft_top5_probs": [],
                                    "draft_model_top1_confidence": None,
                                    "draft_confidence_cumprod": None,
                                }
                            else:
                                step_stat = draft_step_stats_batch[i][j - carry_len]
                            target_pos_stats = _token_and_topk_probs_from_logits(logits_target_eval[0, j + 1], dt, topk=args.topk)
                            inter_pos_stats = _token_and_topk_probs_from_logits(logits_inter_eval[0, j + 1], dt, topk=args.topk)
                            detail_row = {
                                "dataset": dataset_key,
                                "request_index": s["request_index"],
                                "request_sample_id": req["request_sample_id"],
                                "sample_id": round_sample_id,
                                "request_id": req["request_id"],
                                "method": args.method,
                                "batch_characteristics": round_chars,
                                "prompt_style": req["prompt_meta"]["prompt_style"],
                                "turn_index_used": req["turn_index"],
                                "num_turns_in_record": req["prompt_meta"]["num_turns_in_record"],
                                "history_mode": req["prompt_meta"]["history_mode"],
                                "history_assistant_turns_generated": req["prompt_meta"]["history_assistant_turns_generated"],
                                "verification_round": round_idx_for_row,
                                "dataset_global_round": dataset_global_round,
                                "overall_global_round": overall_round,
                                "position": j,
                                "draft_token_id": dt,
                                "draft_source_model": step_stat["draft_source_model"],
                                "draft_top1_prob": step_stat["draft_top1_prob"],
                                "draft_top5_token_ids": step_stat["draft_top5_token_ids"],
                                "draft_top5_probs": step_stat["draft_top5_probs"],
                                "draft_model_top1_confidence": step_stat.get("draft_model_top1_confidence"),
                                "draft_confidence_cumprod": step_stat.get("draft_confidence_cumprod"),
                                "intermediate_topk": itopk,
                                "target_topk": ttopk,
                                "intermediate_draft_token_prob": inter_pos_stats["token_prob"],
                                "intermediate_top1_prob": inter_pos_stats["top1_prob"],
                                "intermediate_top5_token_ids": inter_pos_stats["top5_token_ids"],
                                "intermediate_top5_probs": inter_pos_stats["top5_probs"],
                                "target_draft_token_prob": target_pos_stats["token_prob"],
                                "target_top1_prob": target_pos_stats["top1_prob"],
                                "target_top5_token_ids": target_pos_stats["top5_token_ids"],
                                "target_top5_probs": target_pos_stats["top5_probs"],
                                "accept_by_target": at,
                                "accept_by_intermediate": ai,
                                "verification_model": verification_model,
                                "forced_target_by_interval": bool(forced_target_by_interval),
                                "threshold_cross_position": threshold_cross_position,
                            }
                            detail_f.write(json.dumps(detail_row, ensure_ascii=False) + "\n")
                            state["detail_rows_written"] += 1

                    state["total_rounds"] += 1
                    overall_round += 1
                    s["request_round_idx"] += 1

                    if verification_model == "intermediate":
                        state["verification_model_intermediate_rounds"] += 1
                        s["verification_counter_since_target"] += 1
                        # In morphable intermediate verification, we stage provisional tokens
                        # for later target verification and do not commit output yet.
                        recovery_included = False
                        carry_tokens = [s["current_recovery"]] + raw_draft_tokens[:n_accept_inter]
                        if n_accept_inter < len(raw_draft_tokens):
                            inter_recovery_prob = float(torch.softmax(logits_inter_eval[0, n_accept_inter + 1].float(), dim=-1)[inter_recovery].item())
                            recovery_included = _should_include_bonus_recovery(
                                bonus_method=args.bonus_method,
                                bonus_threshold=args.bonus_threshold,
                                recovery_confidence=inter_recovery_prob,
                            )
                            if recovery_included:
                                carry_tokens.append(inter_recovery)
                        if recovery_included:
                            state["bonus_recovery_included_count"] += 1
                        else:
                            state["bonus_recovery_skipped_count"] += 1
                        verification_row["recovery_included_in_prefix"] = bool(recovery_included)
                        s["carry_over_prefix_tokens"].extend(carry_tokens)

                        s["prompt_ids_for_round"] = s["prompt_ids_for_round"] + carry_tokens
                        s["max_context_seen"] = max(s["max_context_seen"], len(s["prompt_ids_for_round"]))
                        s["current_recovery"] = inter_recovery
                        # Do not terminate on intermediate EOS; target must finalize termination.
                        if verification_f is not None:
                            verification_f.write(json.dumps(verification_row, ensure_ascii=False) + "\n")
                        continue
                    else:
                        state["verification_model_target_rounds"] += 1
                        if forced_target_by_interval:
                            state["interval_forced_target_rounds"] += 1
                        s["verification_counter_since_target"] = 0
                        verification_row["recovery_included_in_prefix"] = None

                    # Target verification commits accepted prefix over the combined candidate
                    # stream (carry-over provisional + current round draft).
                    base_prompt_for_commit = list(s["prompt_ids_for_round"])
                    if carry_prefix_tokens:
                        base_prompt_for_commit = base_prompt_for_commit[: len(base_prompt_for_commit) - len(carry_prefix_tokens)]
                    s["carry_over_prefix_tokens"] = []
                    emitted_tokens_full = [s["current_recovery"]] + eval_draft_tokens[:n_accept_target]
                    next_recovery = target_recovery

                    if verification_f is not None:
                        verification_f.write(json.dumps(verification_row, ensure_ascii=False) + "\n")
                    remaining_budget = req["max_new_tokens"] - s["generated_count"]
                    committed_tokens = emitted_tokens_full[: max(remaining_budget, 0)]
                    committed_tokens, hit_stop_in_committed = _truncate_at_stop_token(committed_tokens, stop_ids)
                    s["prompt_ids_for_round"] = base_prompt_for_commit + committed_tokens
                    s["generated_output_ids"].extend(int(tok) for tok in committed_tokens)
                    s["generated_count"] += len(committed_tokens)
                    s["max_context_seen"] = max(s["max_context_seen"], len(s["prompt_ids_for_round"]))

                    if hit_stop_in_committed:
                        s["done"] = True
                        continue

                    if len(committed_tokens) < len(emitted_tokens_full):
                        s["done"] = True
                        continue

                    s["current_recovery"] = next_recovery
                    if s["generated_count"] >= req["max_new_tokens"]:
                        s["done"] = True
                        continue
                    if s["current_recovery"] in stop_ids:
                        if s["generated_count"] < req["max_new_tokens"]:
                            s["prompt_ids_for_round"] = s["prompt_ids_for_round"] + [s["current_recovery"]]
                            s["generated_output_ids"].append(int(s["current_recovery"]))
                            s["generated_count"] += 1
                            s["max_context_seen"] = max(s["max_context_seen"], len(s["prompt_ids_for_round"]))
                        s["done"] = True
                        continue

                if verification_f is not None:
                    verification_f.flush()
                if detail_f is not None:
                    detail_f.flush()

            for s in batch_states:
                req = s["request"]
                final_context_tokens = len(s["prompt_ids_for_round"])
                state["final_context_tokens"].append(final_context_tokens)
                state["max_context_tokens_seen"].append(s["max_context_seen"])
                state["generated_tokens_per_request"].append(s["generated_count"])
                state["verification_rounds_per_request"].append(s["request_round_idx"])

                current_turn_text = str(req["record"]["turns"][req["turn_index"]])
                generated_text = _decode_generated_ids(tokenizer, s["generated_output_ids"])
                response_row = {
                    "dataset": dataset_key,
                    "request_index": s["request_index"],
                    "request_sample_id": req["request_sample_id"],
                    "sample_id": req["request_id"],
                    "method": args.method,
                    "prompt_style": req["prompt_meta"]["prompt_style"],
                    "turn_index_used": req["turn_index"],
                    "num_turns_in_record": req["prompt_meta"]["num_turns_in_record"],
                    "history_mode": req["prompt_meta"]["history_mode"],
                    "history_assistant_turns_generated": req["prompt_meta"]["history_assistant_turns_generated"],
                    "question": current_turn_text,
                    "generated_token_ids": [int(tok) for tok in s["generated_output_ids"]],
                    "generated_text": generated_text,
                    "generated_tokens": s["generated_count"],
                    "num_verification_rounds": s["request_round_idx"],
                }
                request_meta_row = {
                    "dataset": dataset_key,
                    "request_index": s["request_index"],
                    "request_sample_id": req["request_sample_id"],
                    "sample_id": req["request_id"],
                    "method": args.method,
                    "prompt_style": req["prompt_meta"]["prompt_style"],
                    "turn_index_used": req["turn_index"],
                    "num_turns_in_record": req["prompt_meta"]["num_turns_in_record"],
                    "history_mode": req["prompt_meta"]["history_mode"],
                    "history_assistant_turns_generated": req["prompt_meta"]["history_assistant_turns_generated"],
                    "context_tokens_initial": len(req["prompt_ids"]),
                    "context_tokens_final": final_context_tokens,
                    "max_context_tokens_seen": s["max_context_seen"],
                    "generated_tokens": s["generated_count"],
                    "num_verification_rounds": s["request_round_idx"],
                }
                if response_f is not None:
                    response_f.write(json.dumps(response_row, ensure_ascii=False) + "\n")
                if request_meta_f is not None:
                    request_meta_f.write(json.dumps(request_meta_row, ensure_ascii=False) + "\n")
                if args.print_request_progress:
                    _rank0_print(
                        f"[{dataset_key}] done request {s['request_index'] + 1}/{total_requests} sample={req['request_id']} rounds={s['request_round_idx']} generated={s['generated_count']}",
                        flush=True,
                    )

            if response_f is not None:
                response_f.flush()
            if request_meta_f is not None:
                request_meta_f.flush()
            request_counter += len(batch_requests)
    finally:
        if verification_f is not None:
            verification_f.close()
        if detail_f is not None:
            detail_f.close()
        if request_meta_f is not None:
            request_meta_f.close()
        if response_f is not None:
            response_f.close()

    summary = build_summary(
        args=args,
        data_dir=data_dir,
        dataset_keys=[dataset_key],
        num_prompts=total_requests,
        state=state,
        summary_scope="per_dataset",
    )
    summary["config"]["method"] = args.method
    summary["config"]["batch_size"] = args.batch_size
    summary["config"]["confidence_threshold"] = args.confidence_threshold
    summary["config"]["interval"] = args.interval
    summary["config"]["bonus_method"] = args.bonus_method
    summary["config"]["bonus_threshold"] = args.bonus_threshold
    summary["batch_characteristics"] = {
        "round_batch_size": _stats_num(state.get("batch_sizes", [])),
        "draft_utilization": _stats_num(state.get("batch_draft_utilization", [])),
        "inter_utilization": _stats_num(state.get("batch_inter_utilization", [])),
        "target_utilization": _stats_num(state.get("batch_target_utilization", [])),
        "draft_model_draft_utilization": _stats_num(state.get("batch_draft_model_draft_utilization", [])),
        "inter_model_draft_utilization": _stats_num(state.get("batch_inter_model_draft_utilization", [])),
        "inter_model_verify_utilization": _stats_num(state.get("batch_inter_model_verify_utilization", [])),
        "target_model_verify_utilization": _stats_num(state.get("batch_target_model_verify_utilization", [])),
        "total_draft_model_draft_tokens": int(state.get("total_draft_model_draft_tokens", 0)),
        "total_inter_model_draft_tokens": int(state.get("total_inter_model_draft_tokens", 0)),
        "total_inter_model_verify_tokens": int(state.get("total_inter_model_verify_tokens", 0)),
        "total_target_model_verify_tokens": int(state.get("total_target_model_verify_tokens", 0)),
    }
    total_model_rounds = state["verification_model_target_rounds"] + state["verification_model_intermediate_rounds"]
    summary["morphable_characteristics"] = {
        "verification_model_target_rounds": state["verification_model_target_rounds"],
        "verification_model_intermediate_rounds": state["verification_model_intermediate_rounds"],
        "verification_model_target_rate": (state["verification_model_target_rounds"] / total_model_rounds) if total_model_rounds else 0.0,
        "verification_model_intermediate_rate": (state["verification_model_intermediate_rounds"] / total_model_rounds) if total_model_rounds else 0.0,
        "interval_forced_target_rounds": state["interval_forced_target_rounds"],
        "bonus_recovery_included_count": state["bonus_recovery_included_count"],
        "bonus_recovery_skipped_count": state["bonus_recovery_skipped_count"],
    }
    summary_path = save_summary(summary, dataset_out_dir) if _is_main_process() else (dataset_out_dir / "intermediate_vs_target_summary.json")
    return summary, state, overall_round, summary_path, detail_path, response_path


def _stats_num(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(sum(values) / len(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch speculative decoding profile with vanila/bump/morphable methods.")
    parser.add_argument("--draft", type=str, default=_model_path("draft", DEFAULT_DRAFT), help="Draft model HF id")
    parser.add_argument("--intermediate", type=str, default=_model_path("intermediate", DEFAULT_INTERMEDIATE), help="Intermediate verifier HF id")
    parser.add_argument("--target", type=str, default=_model_path("target", DEFAULT_TARGET), help="Target model HF id")
    parser.add_argument("--output-dir", type=str, default="profile/results_batch", help="Directory to save stats")
    parser.add_argument("--datasets", type=str, default="alpaca,gsm8k,humaneval,mt_bench,qa", help="Comma-separated dataset keys")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SSD_DATASET_DIR", ""), help="Root directory containing dataset jsonl files/subdirs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset.")
    parser.add_argument("--alpaca-max-new-tokens", type=int, default=1024)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=1024)
    parser.add_argument("--humaneval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--mt-bench-max-new-tokens", type=int, default=1024)
    parser.add_argument("--qa-max-new-tokens", type=int, default=1024)
    parser.add_argument("--k", type=int, default=5, help="Number of draft tokens per round")
    parser.add_argument("--method", type=str, default="vanila", choices=["vanila", "vanilla", "bump", "morphable"], help="Speculative method.")
    parser.add_argument("--confidence-threshold", type=float, default=0.8, help="Morphable cumulative draft confidence threshold")
    parser.add_argument("--interval", type=int, default=4, help="Force target verification when intermediate counter reaches interval")
    parser.add_argument("--bonus-method", type=str, default="adaptive", choices=["proactive", "conservative", "adaptive"], help="Carry-over recovery inclusion policy")
    parser.add_argument("--bonus-threshold", type=float, default=0.8, help="Adaptive bonus recovery inclusion confidence threshold")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro batch size for model forwards")
    parser.add_argument("--chat-template", action="store_true", default=True, help="Apply tokenizer chat template when relevant")
    parser.add_argument("--no-chat-template", action="store_false", dest="chat_template")
    parser.add_argument("--prompt-style", type=str, default="auto", choices=["auto", "vicuna", "llama3_instruct", "generic"])
    parser.add_argument("--conversation-turn-index", type=int, default=0)
    parser.add_argument("--conversation-turn-mode", type=str, default="all", choices=["all", "selected"])
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--query-window-size", type=int, default=8, help="Compatibility field for summary config")
    parser.add_argument("--device-draft", type=str, default="auto")
    parser.add_argument("--device-intermediate", type=str, default="auto")
    parser.add_argument("--device-target", type=str, default="auto")
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Global random seed used for reproducible runs")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic backend settings (default: True)",
    )
    parser.add_argument("--save-per-position-detail", action="store_true", default=True)
    parser.add_argument("--no-save-per-position-detail", action="store_false", dest="save_per_position_detail")
    parser.add_argument("--print-request-progress", action="store_true", default=True)
    parser.add_argument("--no-print-request-progress", action="store_false", dest="print_request_progress")
    parser.add_argument("--verification-jsonl", type=str, default="verification_metrics.jsonl")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if args.interval < 0:
        raise ValueError("--interval must be >= 0")
    if not (0.0 <= args.confidence_threshold <= 1.0):
        raise ValueError("--confidence-threshold must be in [0, 1]")
    if not (0.0 <= args.bonus_threshold <= 1.0):
        raise ValueError("--bonus-threshold must be in [0, 1]")
    args.method = _normalize_method(args.method)
    args.bonus_method = _normalize_bonus_method(args.bonus_method)

    configure_reproducibility(seed=args.seed, deterministic=args.deterministic)
    torch.use_deterministic_algorithms(True)

    local_rank = _maybe_init_target_tp(args.target_tp_size)
    args.device_draft = _resolve_device_arg(args.device_draft, fallback_cuda_index=local_rank)
    args.device_intermediate = _resolve_device_arg(args.device_intermediate, fallback_cuda_index=local_rank)
    args.device_target = _resolve_device_arg(args.device_target, fallback_cuda_index=local_rank)

    args.output_dir = str(build_model_run_dir(args.output_dir, args.draft, args.intermediate, args.target) / f"method__{args.method}")
    os.makedirs(args.output_dir, exist_ok=True)
    _rank0_print(f"Output directory: {args.output_dir}")

    if not args.data_dir:
        raise ValueError("--data-dir is required (or set SSD_DATASET_DIR)")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dataset_keys = parse_csv_list(args.datasets)
    dataset_prompts: dict[str, list[dict[str, Any]]] = {}
    total_prompts = 0
    for key in dataset_keys:
        prompts = get_dataset_records(key, data_dir, args.max_samples)
        if prompts:
            dataset_prompts[key] = prompts
            total_prompts += sum(
                count_profile_requests_for_record(record["turns"], args.conversation_turn_mode, args.conversation_turn_index)
                for record in prompts
            )
    if not dataset_prompts:
        _rank0_print("No prompts loaded. Exiting.")
        return 1

    tokenizer = get_tokenizer(args.target)

    def _load_causal_lm(path: str, device: str, *, tp_size: int = 1):
        load_kwargs = {"torch_dtype": torch.bfloat16}
        # load_kwargs = {"torch_dtype": torch.float32}
        if tp_size > 1:
            try:
                return AutoModelForCausalLM.from_pretrained(path, tp_plan="auto", tp_size=tp_size, **load_kwargs)
            except TypeError:
                return AutoModelForCausalLM.from_pretrained(path, tp_plan="auto", **load_kwargs)
        return AutoModelForCausalLM.from_pretrained(path, **load_kwargs).to(device)

    _rank0_print("Loading draft model...")
    draft_model = _load_causal_lm(args.draft, args.device_draft)
    draft_model.set_attn_implementation("eager")
    draft_model.eval()
    
    _rank0_print("Loading intermediate model...")
    inter_model = _load_causal_lm(args.intermediate, args.device_intermediate)
    inter_model.set_attn_implementation("eager")
    inter_model.eval()
    _rank0_print("Loading target model...")
    target_model = _load_causal_lm(args.target, args.device_target, tp_size=args.target_tp_size)
    target_model.set_attn_implementation("eager")
    target_model.eval()

    overall_state = make_metric_state()
    overall_state["batch_sizes"] = []
    overall_state["batch_draft_utilization"] = []
    overall_state["batch_inter_utilization"] = []
    overall_state["batch_target_utilization"] = []
    overall_state["batch_draft_model_draft_utilization"] = []
    overall_state["batch_inter_model_draft_utilization"] = []
    overall_state["batch_inter_model_verify_utilization"] = []
    overall_state["batch_target_model_verify_utilization"] = []
    overall_state["total_draft_model_draft_tokens"] = 0
    overall_state["total_inter_model_draft_tokens"] = 0
    overall_state["total_inter_model_verify_tokens"] = 0
    overall_state["total_target_model_verify_tokens"] = 0
    overall_state["verification_model_target_rounds"] = 0
    overall_state["verification_model_intermediate_rounds"] = 0
    overall_state["interval_forced_target_rounds"] = 0
    overall_state["bonus_recovery_included_count"] = 0
    overall_state["bonus_recovery_skipped_count"] = 0
    overall_round = 0

    for dataset_key in dataset_keys:
        prompts = dataset_prompts.get(dataset_key, [])
        if not prompts:
            _rank0_print(f"[skip] {dataset_key}: no prompts")
            continue
        _rank0_print(f"[run] dataset={dataset_key} method={args.method} batch_size={args.batch_size}")
        summary, dataset_state, overall_round, summary_path, detail_path, response_path = run_dataset_profile_batch(
            dataset_key=dataset_key,
            prompt_records=prompts,
            args=args,
            data_dir=data_dir,
            tokenizer=tokenizer,
            draft_model=draft_model,
            inter_model=inter_model,
            target_model=target_model,
            starting_overall_round=overall_round,
        )
        merge_metric_states(overall_state, dataset_state)
        overall_state["batch_sizes"].extend(dataset_state.get("batch_sizes", []))
        overall_state["batch_draft_utilization"].extend(dataset_state.get("batch_draft_utilization", []))
        overall_state["batch_inter_utilization"].extend(dataset_state.get("batch_inter_utilization", []))
        overall_state["batch_target_utilization"].extend(dataset_state.get("batch_target_utilization", []))
        overall_state["batch_draft_model_draft_utilization"].extend(dataset_state.get("batch_draft_model_draft_utilization", []))
        overall_state["batch_inter_model_draft_utilization"].extend(dataset_state.get("batch_inter_model_draft_utilization", []))
        overall_state["batch_inter_model_verify_utilization"].extend(dataset_state.get("batch_inter_model_verify_utilization", []))
        overall_state["batch_target_model_verify_utilization"].extend(dataset_state.get("batch_target_model_verify_utilization", []))
        overall_state["total_draft_model_draft_tokens"] += int(dataset_state.get("total_draft_model_draft_tokens", 0))
        overall_state["total_inter_model_draft_tokens"] += int(dataset_state.get("total_inter_model_draft_tokens", 0))
        overall_state["total_inter_model_verify_tokens"] += int(dataset_state.get("total_inter_model_verify_tokens", 0))
        overall_state["total_target_model_verify_tokens"] += int(dataset_state.get("total_target_model_verify_tokens", 0))
        overall_state["verification_model_target_rounds"] += int(dataset_state.get("verification_model_target_rounds", 0))
        overall_state["verification_model_intermediate_rounds"] += int(dataset_state.get("verification_model_intermediate_rounds", 0))
        overall_state["interval_forced_target_rounds"] += int(dataset_state.get("interval_forced_target_rounds", 0))
        overall_state["bonus_recovery_included_count"] += int(dataset_state.get("bonus_recovery_included_count", 0))
        overall_state["bonus_recovery_skipped_count"] += int(dataset_state.get("bonus_recovery_skipped_count", 0))
        _rank0_print(f"Wrote dataset summary to {summary_path}")
        if detail_path is not None:
            _rank0_print(f"Wrote detail rows to {detail_path}")
        if response_path is not None:
            _rank0_print(f"Wrote responses to {response_path}")
        _rank0_print(
            f"[{dataset_key}] Avg acceptance length (target/intermediate): {summary['avg_acceptance_length_target']:.4f} / {summary['avg_acceptance_length_intermediate']:.4f}"
        )

    overall_summary = build_summary(
        args=args,
        data_dir=data_dir,
        dataset_keys=list(dataset_prompts.keys()),
        num_prompts=total_prompts,
        state=overall_state,
        summary_scope="aggregate",
    )
    overall_summary["config"]["method"] = args.method
    overall_summary["config"]["batch_size"] = args.batch_size
    overall_summary["config"]["confidence_threshold"] = args.confidence_threshold
    overall_summary["config"]["interval"] = args.interval
    overall_summary["config"]["bonus_method"] = args.bonus_method
    overall_summary["config"]["bonus_threshold"] = args.bonus_threshold
    overall_summary["batch_characteristics"] = {
        "round_batch_size": _stats_num(overall_state.get("batch_sizes", [])),
        "draft_utilization": _stats_num(overall_state.get("batch_draft_utilization", [])),
        "inter_utilization": _stats_num(overall_state.get("batch_inter_utilization", [])),
        "target_utilization": _stats_num(overall_state.get("batch_target_utilization", [])),
        "draft_model_draft_utilization": _stats_num(overall_state.get("batch_draft_model_draft_utilization", [])),
        "inter_model_draft_utilization": _stats_num(overall_state.get("batch_inter_model_draft_utilization", [])),
        "inter_model_verify_utilization": _stats_num(overall_state.get("batch_inter_model_verify_utilization", [])),
        "target_model_verify_utilization": _stats_num(overall_state.get("batch_target_model_verify_utilization", [])),
        "total_draft_model_draft_tokens": int(overall_state.get("total_draft_model_draft_tokens", 0)),
        "total_inter_model_draft_tokens": int(overall_state.get("total_inter_model_draft_tokens", 0)),
        "total_inter_model_verify_tokens": int(overall_state.get("total_inter_model_verify_tokens", 0)),
        "total_target_model_verify_tokens": int(overall_state.get("total_target_model_verify_tokens", 0)),
    }
    total_model_rounds = overall_state["verification_model_target_rounds"] + overall_state["verification_model_intermediate_rounds"]
    overall_summary["morphable_characteristics"] = {
        "verification_model_target_rounds": overall_state["verification_model_target_rounds"],
        "verification_model_intermediate_rounds": overall_state["verification_model_intermediate_rounds"],
        "verification_model_target_rate": (overall_state["verification_model_target_rounds"] / total_model_rounds) if total_model_rounds else 0.0,
        "verification_model_intermediate_rate": (overall_state["verification_model_intermediate_rounds"] / total_model_rounds) if total_model_rounds else 0.0,
        "interval_forced_target_rounds": overall_state["interval_forced_target_rounds"],
        "bonus_recovery_included_count": overall_state["bonus_recovery_included_count"],
        "bonus_recovery_skipped_count": overall_state["bonus_recovery_skipped_count"],
    }
    overall_summary_path = save_summary(overall_summary, Path(args.output_dir)) if _is_main_process() else (Path(args.output_dir) / "intermediate_vs_target_summary.json")
    _rank0_print(f"Wrote aggregate summary to {overall_summary_path}")
    _maybe_barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
