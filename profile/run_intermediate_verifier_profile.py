#!/usr/bin/env python3
"""
Profile script: Draft -> serial Intermediate verifier -> Target verifier.
Compares position-wise:
- Intermediate verifier vs Target: top-k tokens per position, acceptance rate, avg acceptance length.
- Accept decision is always made by Target top-1; we record what Intermediate would have done.
Additionally records per-round intermediate-vs-target precision for intermediate-accepted tokens:
- Intermediate accepted tokens are the draft prefix up to intermediate acceptance length.
- Target emitted token sequence for precision comparison is the target-accepted draft prefix,
  followed by target recovery token when a reject occurs.
- For each intermediate-accepted token, if it matches the target-emitted token at the same
  position it is counted as a true positive; otherwise false positive.
  Precision = TP / (TP + FP), or null when TP + FP == 0.

Additional draft-side signals:
- Per drafted position: draft top-1 probability and draft top-5 probabilities in
  per_position_top5_detail.jsonl.
- Per verification round: draft-start token ids and last-layer q_proj query vectors
  saved to draft_query_vectors.pt.
- Per drafted position: summed attention map over all draft layers and heads, followed by
  a softmax normalization, saved to draft_attention_maps.pt.
- Per target verification round: accepted-token and recovery-token probabilities
  saved to verification_metrics.jsonl and per_position_top5_detail.jsonl.

Results are saved per dataset under:
  <output_dir>/<dataset_key>/
    - verification_metrics.jsonl
    - intermediate_vs_target_summary.json
    - per_position_top5_detail.jsonl
    - request_metadata.jsonl
    - response.jsonl
    - draft_query_vectors.pt
    - draft_attention_maps.pt
An aggregate summary across all requested datasets is also written to:
  <output_dir>/intermediate_vs_target_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("Install transformers: pip install transformers")

try:
    from fastchat.model import get_conversation_template
except ImportError:
    get_conversation_template = None


def _model_path(name: str, default_hf: str) -> str:
    env = os.environ.get(f"SSD_PROFILE_{name.upper()}_MODEL")
    return env if env else default_hf


DEFAULT_DRAFT = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_INTERMEDIATE = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_TARGET = "meta-llama/Llama-3.3-70B-Instruct"

# DEFAULT_DRAFT = "double7/vicuna-68m"
# DEFAULT_INTERMEDIATE = "lmsys/vicuna-7b-v1.3"
# DEFAULT_TARGET = "lmsys/vicuna-13b-v1.3"


def sanitize_path_component(name: str) -> str:
    name = str(name).strip().replace("\\", "/").rstrip("/")
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "unknown_model"


def build_model_run_dir(output_dir: str | Path, draft: str, intermediate: str, target: str) -> Path:
    root = Path(output_dir).expanduser()
    run_name = (
        f"draft__{sanitize_path_component(draft)}"
        f"__intermediate__{sanitize_path_component(intermediate)}"
        f"__target__{sanitize_path_component(target)}"
    )
    return root / run_name


DATASET_KEYS_SUPPORTED = ["alpaca", "gsm8k", "humaneval", "mt_bench", "qa"]


def extract_first_present(example: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in example and example[key] is not None:
            return str(example[key])
    return default


def _build_single_turn_text(dataset_key: str, example: dict[str, Any]) -> str:
    if dataset_key == "alpaca":
        instruction = extract_first_present(example, ["instruction", "text", "prompt", "question"])
        input_text = extract_first_present(example, ["input"], default="")
        if input_text:
            return f"{instruction}\n\nInput:\n{input_text}"
        return instruction
    if dataset_key == "gsm8k":
        question = extract_first_present(example, ["question", "text", "problem", "prompt"])
        return f"Solve the following math word problem.\n\nQuestion:\n{question}"
    if dataset_key == "humaneval":
        return extract_first_present(example, ["prompt", "text", "question"])
    if dataset_key == "qa":
        question = extract_first_present(example, ["question", "prompt", "text"])
        context = extract_first_present(example, ["context", "passage"], default="")
        if context:
            return f"Context:\n{context}\n\nQuestion:\n{question}"
        return question
    turns = example.get("turns")
    if isinstance(turns, list) and turns:
        return str(turns[0])
    return extract_first_present(example, ["question", "prompt", "text"])


def extract_turns(dataset_key: str, example: dict[str, Any]) -> list[str]:
    turns = example.get("turns")
    if isinstance(turns, list):
        cleaned = [str(x) for x in turns if x is not None and str(x).strip()]
        if cleaned:
            return cleaned
    single_turn = _build_single_turn_text(dataset_key, example).strip()
    return [single_turn] if single_turn else []


def _read_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _resolve_dataset_jsonl(data_dir: Path, dataset_key: str) -> Path:
    aliases = {
        "alpaca": ["alpaca"],
        "gsm8k": ["gsm8k", "gsm"],
        "humaneval": ["humaneval"],
        "mt_bench": ["mt_bench", "mtbench"],
        "qa": ["qa"],
    }
    keys = aliases.get(dataset_key, [dataset_key])
    candidates: list[Path] = []
    for k in keys:
        candidates.extend(data_dir.glob(f"{k}.jsonl"))
        candidates.extend(data_dir.glob(f"{k}_*.jsonl"))
        candidates.extend(data_dir.glob(f"{k}/{k}.jsonl"))
        candidates.extend(data_dir.glob(f"{k}/{k}_*.jsonl"))
        candidates.extend(data_dir.glob(f"{k}/*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Could not find jsonl for dataset '{dataset_key}' under {data_dir}")
    return sorted(candidates)[0]


def get_dataset_records(dataset_key: str, data_dir: Path, max_samples: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if dataset_key not in DATASET_KEYS_SUPPORTED:
        raise ValueError(f"Unsupported dataset key: {dataset_key}. Supported={DATASET_KEYS_SUPPORTED}")
    dataset_path = _resolve_dataset_jsonl(data_dir, dataset_key)
    rows = _read_jsonl(dataset_path, max_samples=max_samples)
    for i, row in enumerate(rows):
        turns = extract_turns(dataset_key, row)
        if not turns:
            continue
        sid = extract_first_present(row, ["question_id", "id", "name", "task_id"], str(i))
        out.append({"dataset_key": dataset_key, "sample_id": sid, "row": row, "turns": turns})
    _rank0_print(f"[dataset] {dataset_key}: path={dataset_path}, samples={len(out)}")
    return out


def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)



def _distributed_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_rank() -> int:
    if _distributed_is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _get_world_size() -> int:
    if _distributed_is_initialized():
        return int(torch.distributed.get_world_size())
    return 1


def _is_main_process() -> bool:
    return _get_rank() == 0


def _rank0_print(*args, **kwargs) -> None:
    if _is_main_process():
        print(*args, **kwargs)


def _maybe_init_target_tp(target_tp_size: int) -> int:
    if target_tp_size <= 1:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("--target-tp-size > 1 requires CUDA.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not _distributed_is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world_size = _get_world_size()
    if world_size != int(target_tp_size):
        raise RuntimeError(
            f"--target-tp-size={target_tp_size} but WORLD_SIZE={world_size}. Launch with torchrun --nproc_per_node {target_tp_size}."
        )
    return local_rank


def _maybe_barrier() -> None:
    if _distributed_is_initialized():
        torch.distributed.barrier()


def _resolve_device_arg(device: str, *, fallback_cuda_index: int = 0) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return f"cuda:{fallback_cuda_index}"
        return "cpu"
    return device


def configure_reproducibility(*, seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _get_model_input_device(model, fallback_device: str) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    return torch.device(fallback_device)
EAGLE_LLAMA3_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information."
)


def infer_prompt_style(model_name_or_path: str) -> str:
    low = model_name_or_path.lower()
    if "vicuna" in low:
        return "vicuna"
    if "llama" in low and "instruct" in low and re.search(r"llama[-_ ]?3(\.1|\.2|\.3)?", low):
        return "llama3_instruct"
    return "generic"


def resolve_prompt_style(requested_style: str, model_name_or_path: str) -> str:
    if requested_style != "auto":
        return requested_style
    return infer_prompt_style(model_name_or_path)


def resolve_turn_index(turns: list[str], turn_index: int) -> int:
    if not turns:
        return 0
    if turn_index < 0:
        turn_index = len(turns) + turn_index
    return min(max(turn_index, 0), len(turns) - 1)


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def make_metric_state() -> dict[str, Any]:
    return {
        "position_accept_target": defaultdict(list),
        "position_accept_inter": defaultdict(list),
        "accept_len_target_list": [],
        "accept_len_inter_list": [],
        "same_accept_len_count": 0,
        "same_accept_len_bonus_same_count": 0,
        "diff_accept_len_count": 0,
        "diff_accept_len_inter_bonus_in_next_draft_count": 0,
        "diff_accept_len_inter_bonus_accept_next_target_count": 0,
        "diff_accept_len_with_next_round_count": 0,
        "total_rounds": 0,
        "detail_rows_written": 0,
        "request_count": 0,
        "initial_context_tokens": [],
        "final_context_tokens": [],
        "max_context_tokens_seen": [],
        "generated_tokens_per_request": [],
        "verification_rounds_per_request": [],
        "inter_precision_tp_total": 0,
        "inter_precision_fp_total": 0,
        "inter_precision_round_values": [],
        "inter_precision_nonempty_rounds": 0,
    }


def merge_metric_states(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for pos, values in src["position_accept_target"].items():
        dst["position_accept_target"][pos].extend(values)
    for pos, values in src["position_accept_inter"].items():
        dst["position_accept_inter"][pos].extend(values)
    dst["accept_len_target_list"].extend(src["accept_len_target_list"])
    dst["accept_len_inter_list"].extend(src["accept_len_inter_list"])
    dst["same_accept_len_count"] += src["same_accept_len_count"]
    dst["same_accept_len_bonus_same_count"] += src["same_accept_len_bonus_same_count"]
    dst["diff_accept_len_count"] += src["diff_accept_len_count"]
    dst["diff_accept_len_inter_bonus_in_next_draft_count"] += src["diff_accept_len_inter_bonus_in_next_draft_count"]
    dst["diff_accept_len_inter_bonus_accept_next_target_count"] += src["diff_accept_len_inter_bonus_accept_next_target_count"]
    dst["diff_accept_len_with_next_round_count"] += src["diff_accept_len_with_next_round_count"]
    dst["total_rounds"] += src["total_rounds"]
    dst["detail_rows_written"] += src["detail_rows_written"]
    dst["request_count"] += src["request_count"]
    dst["initial_context_tokens"].extend(src["initial_context_tokens"])
    dst["final_context_tokens"].extend(src["final_context_tokens"])
    dst["max_context_tokens_seen"].extend(src["max_context_tokens_seen"])
    dst["generated_tokens_per_request"].extend(src["generated_tokens_per_request"])
    dst["verification_rounds_per_request"].extend(src["verification_rounds_per_request"])
    dst["inter_precision_tp_total"] += src["inter_precision_tp_total"]
    dst["inter_precision_fp_total"] += src["inter_precision_fp_total"]
    dst["inter_precision_round_values"].extend(src["inter_precision_round_values"])
    dst["inter_precision_nonempty_rounds"] += src["inter_precision_nonempty_rounds"]


def build_summary(*, args, data_dir: Path, dataset_keys: list[str], num_prompts: int, state: dict[str, Any], summary_scope: str) -> dict[str, Any]:
    position_accept_target = state["position_accept_target"]
    position_accept_inter = state["position_accept_inter"]
    accept_len_target_list = state["accept_len_target_list"]
    accept_len_inter_list = state["accept_len_inter_list"]
    total_rounds = state["total_rounds"]
    max_pos = max(position_accept_target.keys()) if position_accept_target else -1
    position_avg_accept_target: dict[int, float] = {}
    position_avg_accept_inter: dict[int, float] = {}
    for j in range(max_pos + 1):
        if j in position_accept_target and position_accept_target[j]:
            position_avg_accept_target[j] = sum(position_accept_target[j]) / len(position_accept_target[j])
        if j in position_accept_inter and position_accept_inter[j]:
            position_avg_accept_inter[j] = sum(position_accept_inter[j]) / len(position_accept_inter[j])
    avg_accept_len_target = sum(accept_len_target_list) / len(accept_len_target_list) if accept_len_target_list else 0
    avg_accept_len_inter = sum(accept_len_inter_list) / len(accept_len_inter_list) if accept_len_inter_list else 0
    prob_same_accept_length = state["same_accept_len_count"] / total_rounds if total_rounds else 0
    prob_bonus_same_given_same_length = (
        state["same_accept_len_bonus_same_count"] / state["same_accept_len_count"] if state["same_accept_len_count"] else 0
    )
    prob_inter_bonus_in_next_draft = (
        state["diff_accept_len_inter_bonus_in_next_draft_count"] / state["diff_accept_len_with_next_round_count"]
        if state["diff_accept_len_with_next_round_count"]
        else 0
    )
    prob_inter_bonus_accept_next_target = (
        state["diff_accept_len_inter_bonus_accept_next_target_count"] / state["diff_accept_len_with_next_round_count"]
        if state["diff_accept_len_with_next_round_count"]
        else 0
    )

    def _stats(values: list[int | float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    inter_tp = state["inter_precision_tp_total"]
    inter_fp = state["inter_precision_fp_total"]
    inter_precision = inter_tp / (inter_tp + inter_fp) if (inter_tp + inter_fp) > 0 else 0.0

    return {
        "config": {
            "summary_scope": summary_scope,
            "draft": args.draft,
            "intermediate": args.intermediate,
            "target": args.target,
            "k": args.k,
            "datasets": dataset_keys,
            "data_dir": str(data_dir),
            "max_samples_per_dataset": args.max_samples,
            "alpaca_max_new_tokens": args.alpaca_max_new_tokens,
            "gsm8k_max_new_tokens": args.gsm8k_max_new_tokens,
            "humaneval_max_new_tokens": args.humaneval_max_new_tokens,
            "mt_bench_max_new_tokens": args.mt_bench_max_new_tokens,
            "qa_max_new_tokens": args.qa_max_new_tokens,
            "num_prompts": num_prompts,
            "total_verify_rounds": total_rounds,
            "detail_rows_written": state["detail_rows_written"],
            "request_count": state["request_count"],
            "prompt_style_requested": args.prompt_style,
            "prompt_style_resolved": resolve_prompt_style(args.prompt_style, args.target),
            "conversation_turn_index": args.conversation_turn_index,
            "conversation_turn_mode": args.conversation_turn_mode,
            "query_window_size": args.query_window_size,
        },
        "metric_definitions": {
            "position_accept_rate": "At position j: (rounds where positions 0,1,...,j were ALL accepted) / total_rounds. If any earlier position rejected, round counts as reject at j. So rate j = P(accept 0 and ... and j).",
            "avg_acceptance_length": "Per round: first index where draft != target top-1 (or K if all accepted). Mean over rounds. With position_accept_rate above, E[length] = rate[0]+rate[1]+...+rate[K-1].",
            "intermediate_precision_vs_target": "For each round, take the intermediate-accepted draft prefix. Compare token-by-token against the target-emitted sequence for that round, defined as target-accepted draft prefix plus the target recovery token when target rejects. Match at same position => true positive, mismatch or missing target token => false positive. Precision = TP / (TP + FP).",
            "draft_query_vector": "Last-layer q_proj output before RoPE at the draft-start conditioning step. Saved in draft_query_vectors.pt together with the recent token-id window.",
        },
        "position_accept_rate_target": position_avg_accept_target,
        "position_accept_rate_intermediate": position_avg_accept_inter,
        "avg_acceptance_length_target": avg_accept_len_target,
        "avg_acceptance_length_intermediate": avg_accept_len_inter,
        "per_position_counts": {str(j): len(position_accept_target[j]) for j in range(max_pos + 1) if j in position_accept_target},
        "same_accept_length": {
            "count": state["same_accept_len_count"],
            "prob_same_accept_length": prob_same_accept_length,
            "bonus_same_count": state["same_accept_len_bonus_same_count"],
            "prob_bonus_same_given_same_length": prob_bonus_same_given_same_length,
        },
        "different_accept_length": {
            "count": state["diff_accept_len_count"],
            "count_with_next_round": state["diff_accept_len_with_next_round_count"],
            "inter_bonus_in_next_draft_count": state["diff_accept_len_inter_bonus_in_next_draft_count"],
            "prob_inter_bonus_in_next_draft": prob_inter_bonus_in_next_draft,
            "inter_bonus_accept_next_target_count": state["diff_accept_len_inter_bonus_accept_next_target_count"],
            "prob_inter_bonus_accept_next_target": prob_inter_bonus_accept_next_target,
        },
        "intermediate_precision_vs_target": {
            "true_positive_count": inter_tp,
            "false_positive_count": inter_fp,
            "precision": inter_precision,
            "rounds_with_nonempty_intermediate_accepts": state["inter_precision_nonempty_rounds"],
            "per_round_precision_stats": _stats(state["inter_precision_round_values"]),
        },
        "context_token_stats": {
            "initial_context_tokens": _stats(state["initial_context_tokens"]),
            "final_context_tokens": _stats(state["final_context_tokens"]),
            "max_context_tokens_seen": _stats(state["max_context_tokens_seen"]),
            "generated_tokens_per_request": _stats(state["generated_tokens_per_request"]),
            "verification_rounds_per_request": _stats(state["verification_rounds_per_request"]),
        },
    }


def save_summary(summary: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intermediate_vs_target_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def get_dataset_max_new_tokens(
    dataset_key: str,
    alpaca_max_new_tokens: int,
    gsm8k_max_new_tokens: int,
    humaneval_max_new_tokens: int,
    mt_bench_max_new_tokens: int,
    qa_max_new_tokens: int,
) -> int:
    if dataset_key == "alpaca":
        return alpaca_max_new_tokens
    if dataset_key == "gsm8k":
        return gsm8k_max_new_tokens
    if dataset_key == "humaneval":
        return humaneval_max_new_tokens
    if dataset_key == "mt_bench":
        return mt_bench_max_new_tokens
    if dataset_key == "qa":
        return qa_max_new_tokens
    return qa_max_new_tokens


def get_logits_at_positions(model, input_ids: torch.Tensor, positions: list[int], device: str):
    inp = input_ids.to(_get_model_input_device(model, device))
    with torch.inference_mode():
        out = model(inp)
    logits = out.logits
    return torch.stack([logits[0, p] for p in positions], dim=0).unsqueeze(0)


def _topk_probs_from_logits(logits_1d: torch.Tensor, topk: int = 5) -> dict[str, Any]:
    probs = torch.softmax(logits_1d.float(), dim=-1)
    vals, idx = torch.topk(probs, k=min(topk, probs.numel()))
    return {
        "top1_token_id": int(idx[0].item()),
        "top1_prob": float(vals[0].item()),
        "top5_token_ids": [int(x) for x in idx.cpu().tolist()],
        "top5_probs": [float(x) for x in vals.cpu().tolist()],
    }


def _token_and_topk_probs_from_logits(logits_1d: torch.Tensor, token_id: int, topk: int = 5) -> dict[str, Any]:
    probs = torch.softmax(logits_1d.float(), dim=-1)
    vals, idx = torch.topk(probs, k=min(topk, probs.numel()))
    token_id = int(token_id)
    return {
        "token_id": token_id,
        "token_prob": float(probs[token_id].item()),
        "top1_token_id": int(idx[0].item()),
        "top1_prob": float(vals[0].item()),
        "top5_token_ids": [int(x) for x in idx.cpu().tolist()],
        "top5_probs": [float(x) for x in vals.cpu().tolist()],
    }


def _aggregate_last_query_attention(attentions) -> dict[str, torch.Tensor]:
    if attentions is None:
        raise RuntimeError("output_attentions=True did not return attentions")
    per_layer_sums: list[torch.Tensor] = []
    for layer_attn in attentions:
        last_q = layer_attn[0, :, -1, :].float()
        per_layer_sums.append(last_q.sum(dim=0))
    attn_sum = torch.stack(per_layer_sums, dim=0).sum(dim=0)
    attn_sum_softmax = torch.softmax(attn_sum, dim=0)
    attn_sum_renorm = attn_sum / attn_sum.sum().clamp_min(1e-12)
    return {
        "attention_sum": attn_sum.detach().to(torch.float16).cpu(),
        "attention_sum_softmax": attn_sum_softmax.detach().to(torch.float16).cpu(),
        "attention_sum_renorm": attn_sum_renorm.detach().to(torch.float16).cpu(),
    }


def _get_last_layer_qproj(model):
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers") or len(base.layers) == 0:
        raise TypeError(
            "Draft query-vector extraction expects a LLaMA/Vicuna style HF causal LM with model.layers[*].self_attn.q_proj"
        )
    return base.layers[-1].self_attn.q_proj


def extract_draft_start_context_from_output(
    draft_model,
    model_output,
    seq: list[int],
    window_size: int,
) -> dict[str, Any]:
    hidden_states = getattr(model_output, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("output_hidden_states=True did not return enough hidden states for draft query extraction")

    if not seq:
        raise ValueError("seq must be non-empty")

    window_size = max(1, min(window_size, len(seq)))
    window_start = len(seq) - window_size

    hs_input_last_layer = hidden_states[-2][0, window_start:]  # [window, hidden]
    q_proj = _get_last_layer_qproj(draft_model)
    q_window = q_proj(hs_input_last_layer).detach().to(torch.float16).cpu()

    return {
        "draft_start_position": len(seq) - 1,
        "draft_start_prev_token_id": int(seq[-1]),
        "draft_start_recent_token_ids": torch.tensor(seq[window_start:], dtype=torch.int32),
        "draft_start_position_ids_window": torch.tensor(list(range(window_start, len(seq))), dtype=torch.int32),
        "draft_start_query_vector": q_window[-1].clone(),
        "draft_start_query_vectors_window": q_window.clone(),
    }


def run_one_verify_round(
    draft_model,
    inter_model,
    target_model,
    prompt_ids: list[int],
    recovery_token_id: int,
    k: int,
    device_draft: str,
    device_inter: str,
    device_target: str,
    query_window_size: int,
    save_draft_query_vectors: bool,
    save_draft_attention_maps: bool,
):
    prompt_len = len(prompt_ids)
    seq = list(prompt_ids) + [recovery_token_id]
    draft_tokens: list[int] = []
    draft_step_stats: list[dict[str, Any]] = []
    draft_start_context: dict[str, Any] | None = None
    draft_attention_records: list[dict[str, Any]] = []
    draft_input_device = _get_model_input_device(draft_model, device_draft)

    for step_idx in range(k):
        inp = torch.tensor([seq], dtype=torch.long, device=draft_input_device)

        need_hidden = save_draft_query_vectors and step_idx == 0
        need_attn = bool(save_draft_attention_maps)

        with torch.inference_mode():
            out = draft_model(
                inp,
                output_hidden_states=need_hidden,
                output_attentions=need_attn,
                use_cache=False,
                return_dict=True,
            )

            if need_hidden:
                draft_start_context = extract_draft_start_context_from_output(
                    draft_model=draft_model,
                    model_output=out,
                    seq=seq,
                    window_size=query_window_size,
                )

            if need_attn:
                attn_info = _aggregate_last_query_attention(out.attentions)

        next_logits = out.logits[0, -1]
        prob_info = _topk_probs_from_logits(next_logits, topk=5)
        next_tok = int(prob_info["top1_token_id"])

        if need_attn:
            draft_attention_records.append(
                {
                    "position": step_idx,
                    "draft_token_id": next_tok,
                    "context_token_ids": torch.tensor(seq, dtype=torch.int32).cpu(),
                    "context_length": len(seq),
                    **attn_info,
                }
            )

        draft_tokens.append(next_tok)
        draft_step_stats.append(
            {
                "position": step_idx,
                "draft_token_id": next_tok,
                "draft_top1_prob": prob_info["top1_prob"],
                "draft_top5_token_ids": prob_info["top5_token_ids"],
                "draft_top5_probs": prob_info["top5_probs"],
            }
        )
        seq.append(next_tok)

    positions_needed = list(range(prompt_len - 1, prompt_len + k + 1))
    full_seq = list(prompt_ids) + [recovery_token_id] + draft_tokens
    inp_full = torch.tensor([full_seq], dtype=torch.long)
    logits_inter = get_logits_at_positions(inter_model, inp_full, positions_needed, device_inter)
    logits_target = get_logits_at_positions(target_model, inp_full, positions_needed, device_target)
    return (
        draft_tokens,
        logits_inter,
        logits_target,
        draft_step_stats,
        draft_start_context,
        draft_attention_records,
    )


def compute_position_stats(draft_tokens: list[int], logits_inter: torch.Tensor, logits_target: torch.Tensor, topk: int):
    inter_topk_list = []
    target_topk_list = []
    accept_target_list = []
    accept_inter_list = []
    draft_tok_list = []
    for j, draft_tok in enumerate(draft_tokens):
        inter_logits_j = logits_inter[0, j + 1]
        target_logits_j = logits_target[0, j + 1]
        inter_topk = inter_logits_j.topk(min(topk, inter_logits_j.size(0))).indices.cpu().tolist()
        target_topk = target_logits_j.topk(min(topk, target_logits_j.size(0))).indices.cpu().tolist()
        # inter_top1 = inter_topk[0]
        # target_top1 = target_topk[0]
        inter_top1 = int(inter_logits_j.argmax(dim=-1).item())
        target_top1 = int(target_logits_j.argmax(dim=-1).item())

        argmax_target = int(target_logits_j.argmax(dim=-1).item())
        topk_target = int(target_logits_j.topk(1).indices[0].item())
        if argmax_target != topk_target:
            print("TARGET TOP1 MISMATCH", j, argmax_target, topk_target)

            
        accept_target_list.append(1 if draft_tok == target_top1 else 0)
        accept_inter_list.append(1 if draft_tok == inter_top1 else 0)
        inter_topk_list.append(inter_topk)
        target_topk_list.append(target_topk)
        draft_tok_list.append(draft_tok)
    return inter_topk_list, target_topk_list, accept_target_list, accept_inter_list, draft_tok_list


def acceptance_length(accept_list: list[int]) -> int:
    for i, accepted in enumerate(accept_list):
        if accepted == 0:
            return i
    return len(accept_list)


def get_recovery_token_from_logits(logits: torch.Tensor, accept_len: int, k: int) -> int:
    del k
    return int(logits[0, accept_len + 1].argmax(dim=-1).item())


def compute_intermediate_precision_against_target(
    draft_tokens: list[int],
    n_accept_inter: int,
    n_accept_target: int,
    target_recovery: int | None,
) -> dict[str, Any]:
    inter_accepted_ids = list(draft_tokens[:n_accept_inter])
    target_emitted_ids = list(draft_tokens[:n_accept_target])
    if target_recovery is not None:
        target_emitted_ids.append(int(target_recovery))

    true_positive = 0
    false_positive = 0
    comparisons: list[dict[str, Any]] = []
    for idx, inter_tok in enumerate(inter_accepted_ids):
        if idx < len(target_emitted_ids):
            target_tok = int(target_emitted_ids[idx])
            is_tp = int(inter_tok == target_tok)
        else:
            target_tok = None
            is_tp = 0
        if is_tp:
            true_positive += 1
        else:
            false_positive += 1
        comparisons.append(
            {
                "position": idx,
                "intermediate_token_id": int(inter_tok),
                "target_token_id": target_tok,
                "is_true_positive": bool(is_tp),
            }
        )

    denom = true_positive + false_positive
    precision = (true_positive / denom) if denom > 0 else None
    return {
        "intermediate_accepted_token_ids": [int(x) for x in inter_accepted_ids],
        "target_emitted_token_ids_for_precision": [int(x) for x in target_emitted_ids],
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "precision": precision,
        "token_comparisons": comparisons,
    }


def _normalize_special_tokens_map_values(tokenizer) -> list[str]:
    values: list[str] = []
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            values.extend(str(tok) for tok in special_token)
        else:
            values.append(str(special_token))
    return values


def _get_stop_token_ids(tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, list):
        stop_ids.update(int(x) for x in eos_token_id if x is not None)
    elif eos_token_id is not None:
        stop_ids.add(int(eos_token_id))
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id:
        stop_ids.add(int(eot_id))
    return stop_ids


def _truncate_at_stop_token(token_ids: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    for i, tok in enumerate(token_ids):
        if int(tok) in stop_ids:
            return list(token_ids[: i + 1]), True
    return list(token_ids), False


def _greedy_generate_ids(model, tokenizer, input_ids: list[int], device: str, max_new_tokens: int) -> list[int]:
    seq = list(input_ids)
    generated: list[int] = []
    stop_ids = _get_stop_token_ids(tokenizer)
    input_device = _get_model_input_device(model, device)
    for _ in range(max_new_tokens):
        inp = torch.tensor([seq], dtype=torch.long, device=input_device)
        with torch.inference_mode():
            out = model(inp)
        next_tok = int(out.logits[0, -1].argmax(dim=-1).item())
        generated.append(next_tok)
        seq.append(next_tok)
        if next_tok in stop_ids:
            break
    return generated


def _decode_vicuna_output(tokenizer, output_ids: list[int], conv) -> str:
    trimmed = list(output_ids)
    stop_token_ids = getattr(conv, "stop_token_ids", None)
    if stop_token_ids:
        for i, tok in enumerate(trimmed):
            if tok in stop_token_ids:
                trimmed = trimmed[:i]
                break
    output = tokenizer.decode(trimmed, spaces_between_special_tokens=False)
    stop_str = getattr(conv, "stop_str", None)
    if isinstance(stop_str, str) and stop_str:
        idx = output.find(stop_str)
        if idx > 0:
            output = output[:idx]
    for special_tok in _normalize_special_tokens_map_values(tokenizer):
        output = output.replace(special_tok, "")
    output = output.strip()
    if getattr(conv, "name", "") == "xgen" and output.startswith("Assistant:"):
        output = output.replace("Assistant:", "", 1).strip()
    return output


def _decode_llama3_output(tokenizer, output_ids: list[int]) -> str:
    trimmed = list(output_ids)
    stop_token_ids = list(_get_stop_token_ids(tokenizer))
    if stop_token_ids:
        stop_set = set(stop_token_ids)
        for i, tok in enumerate(trimmed):
            if tok in stop_set:
                trimmed = trimmed[:i]
                break
    output = tokenizer.decode(trimmed, spaces_between_special_tokens=False)
    for special_tok in _normalize_special_tokens_map_values(tokenizer):
        output = output.replace(special_tok, "")
    return output.strip()


def _decode_generated_ids(tokenizer, output_ids: list[int]) -> str:
    trimmed = list(output_ids)
    stop_token_ids = list(_get_stop_token_ids(tokenizer))
    if stop_token_ids:
        stop_set = set(stop_token_ids)
        for i, tok in enumerate(trimmed):
            if tok in stop_set:
                trimmed = trimmed[:i]
                break
    output = tokenizer.decode(trimmed, spaces_between_special_tokens=False)
    for special_tok in _normalize_special_tokens_map_values(tokenizer):
        output = output.replace(special_tok, "")
    return output.strip()


def _truncate_prompt_tokens(tokens: list[int], max_prompt_tokens: int | None) -> list[int]:
    if max_prompt_tokens is not None and len(tokens) > max_prompt_tokens:
        return list(tokens[:max_prompt_tokens])
    return list(tokens)


def _requested_turn_indices(turns: list[str], conversation_turn_mode: str, conversation_turn_index: int) -> list[int]:
    if not turns:
        return []
    if conversation_turn_mode == "all":
        return list(range(len(turns)))
    return [resolve_turn_index(turns, conversation_turn_index)]


def count_profile_requests_for_record(turns: list[str], conversation_turn_mode: str, conversation_turn_index: int) -> int:
    return len(_requested_turn_indices(turns, conversation_turn_mode, conversation_turn_index))


def _has_tokenizer_chat_template(tokenizer) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    return bool(chat_template) and hasattr(tokenizer, "apply_chat_template")


def _render_messages_vllm_style(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([prompt], add_special_tokens=False).input_ids[0]


def prepare_record_prompt_entries(
    tokenizer,
    target_model,
    target_device: str,
    record: dict[str, Any],
    *,
    prompt_style: str,
    max_prompt_tokens: int | None,
    use_chat_template: bool,
    conversation_turn_index: int,
    conversation_turn_mode: str,
    history_max_new_tokens: int,
) -> list[dict[str, Any]]:
    turns = record["turns"]
    num_turns = len(turns)
    requested_turn_indices = _requested_turn_indices(turns, conversation_turn_mode, conversation_turn_index)
    if not requested_turn_indices:
        return []
    requested_turn_set = set(requested_turn_indices)
    max_requested_turn = max(requested_turn_indices)
    entries: list[dict[str, Any]] = []

    prefer_vllm_style = use_chat_template and _has_tokenizer_chat_template(tokenizer)

    if prompt_style in {"llama3_instruct", "vicuna", "generic"} and prefer_vllm_style:
        messages: list[dict[str, str]] = []
        if prompt_style == "llama3_instruct":
            messages.append({"role": "system", "content": EAGLE_LLAMA3_SYSTEM_PROMPT})
            decode_output = _decode_llama3_output
            history_mode = "vllm_llama_chat_template"
        else:
            decode_output = _decode_llama3_output
            history_mode = "vllm_chat_template_messages"
        generated_history_turns = 0
        for j in range(max_requested_turn + 1):
            qs = str(turns[j])
            messages.append({"role": "user", "content": qs})
            input_ids = _render_messages_vllm_style(tokenizer, messages)
            if j in requested_turn_set:
                entries.append(
                    {
                        "prompt_ids": _truncate_prompt_tokens(input_ids, max_prompt_tokens),
                        "meta": {
                            "prompt_style": prompt_style,
                            "turn_index_used": j,
                            "num_turns_in_record": num_turns,
                            "history_mode": history_mode,
                            "history_assistant_turns_generated": generated_history_turns,
                        },
                    }
                )
            if j < max_requested_turn:
                generated = _greedy_generate_ids(target_model, tokenizer, input_ids, target_device, history_max_new_tokens)
                output = decode_output(tokenizer, generated)
                messages.append({"role": "assistant", "content": output})
                generated_history_turns += 1
        return entries

    if prompt_style == "vicuna":
        if get_conversation_template is None:
            raise ImportError(
                "Vicuna prompt_style requires FastChat fallback when tokenizer has no chat_template. Install fastchat or use a tokenizer with chat_template."
            )
        conv = get_conversation_template("vicuna")
        generated_history_turns = 0
        for j in range(max_requested_turn + 1):
            qs = str(turns[j])
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids[0]
            if j in requested_turn_set:
                entries.append(
                    {
                        "prompt_ids": _truncate_prompt_tokens(input_ids, max_prompt_tokens),
                        "meta": {
                            "prompt_style": prompt_style,
                            "turn_index_used": j,
                            "num_turns_in_record": num_turns,
                            "history_mode": "fastchat_vicuna_fallback",
                            "history_assistant_turns_generated": generated_history_turns,
                        },
                    }
                )
            if j < max_requested_turn:
                generated = _greedy_generate_ids(target_model, tokenizer, input_ids, target_device, history_max_new_tokens)
                output = _decode_vicuna_output(tokenizer, generated, conv)
                conv.messages[-1][-1] = output
                generated_history_turns += 1
        return entries

    for j in requested_turn_indices:
        prompt = str(turns[j])
        if use_chat_template and _has_tokenizer_chat_template(tokenizer):
            messages = [{"role": "user", "content": prompt}]
            tokens = _render_messages_vllm_style(tokenizer, messages)
            hist_meta = {"history_mode": "vllm_chat_template_messages", "history_assistant_turns_generated": 0}
        elif use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            tokens = tokenizer([rendered], add_special_tokens=False).input_ids[0]
            hist_meta = {"history_mode": "generic_chat_template", "history_assistant_turns_generated": 0}
        else:
            tokens = tokenizer.encode(prompt, add_special_tokens=True)
            hist_meta = {"history_mode": "generic_plain", "history_assistant_turns_generated": 0}
        entries.append(
            {
                "prompt_ids": _truncate_prompt_tokens(tokens, max_prompt_tokens),
                "meta": {
                    "prompt_style": prompt_style,
                    "turn_index_used": j,
                    "num_turns_in_record": num_turns,
                    **hist_meta,
                },
            }
        )
    return entries


def run_dataset_profile(
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
) -> tuple[dict[str, Any], dict[str, Any], int, Path, Path | None, Path | None, Path | None, Path | None]:
    dataset_out_dir = Path(args.output_dir) / dataset_key
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    verification_jsonl_path = dataset_out_dir / args.verification_jsonl
    detail_path = dataset_out_dir / "per_position_top5_detail.jsonl" if args.save_per_position_detail else None
    request_metadata_path = dataset_out_dir / "request_metadata.jsonl"
    response_path = dataset_out_dir / "response.jsonl"
    draft_query_vectors_path = dataset_out_dir / "draft_query_vectors.pt" if args.save_draft_query_vectors else None
    draft_attention_maps_path = dataset_out_dir / "draft_attention_maps.pt" if args.save_draft_attention_maps else None
    draft_query_vector_records: list[dict[str, Any]] = []
    draft_attention_records_all: list[dict[str, Any]] = []

    state = make_metric_state()
    overall_round = starting_overall_round
    total_requests = sum(
        count_profile_requests_for_record(record["turns"], args.conversation_turn_mode, args.conversation_turn_index)
        for record in prompt_records
    )

    verification_f = verification_jsonl_path.open("w", encoding="utf-8") if _is_main_process() else None
    detail_f = detail_path.open("w", encoding="utf-8") if (_is_main_process() and detail_path is not None) else None
    request_meta_f = request_metadata_path.open("w", encoding="utf-8") if _is_main_process() else None
    response_f = response_path.open("w", encoding="utf-8") if _is_main_process() else None
    try:
            prompt_style = resolve_prompt_style(args.prompt_style, args.target)
            request_counter = 0
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
                    pending_inter_recovery: int | None = None
                    turn_index = int(prompt_meta["turn_index_used"])
                    request_sample_id = sample_id
                    request_id = f"{sample_id}:turn_{turn_index}" if prompt_meta["num_turns_in_record"] > 1 else str(sample_id)
                    initial_context_tokens = len(prompt_ids)
                    state["request_count"] += 1
                    state["initial_context_tokens"].append(initial_context_tokens)
                    if args.print_request_progress:
                        _rank0_print(
                            f"[{dataset_key}] request {request_counter + 1}/{total_requests} sample={request_id} initial_ctx={initial_context_tokens} max_new={max_new_tokens} style={prompt_meta['prompt_style']}",
                            flush=True,
                        )
                    inp = torch.tensor([prompt_ids], dtype=torch.long, device=_get_model_input_device(target_model, args.device_target))
                    with torch.inference_mode():
                        out = target_model(inp)
                    current_recovery = int(out.logits[0, -1].argmax(dim=-1).item())
                    generated_count = 0
                    generated_output_ids: list[int] = []
                    prompt_ids_for_round = list(prompt_ids)
                    request_round_idx = 0
                    max_context_seen = initial_context_tokens
                    stop_ids = _get_stop_token_ids(tokenizer)
                    while generated_count < max_new_tokens:
                        context_tokens_before_round = len(prompt_ids_for_round) + 1
                        max_context_seen = max(max_context_seen, context_tokens_before_round)
                        (
                            draft_tokens,
                            logits_inter,
                            logits_target,
                            draft_step_stats,
                            draft_start_context,
                            draft_attention_records,
                        ) = run_one_verify_round(
                            draft_model,
                            inter_model,
                            target_model,
                            prompt_ids_for_round,
                            current_recovery,
                            args.k,
                            args.device_draft,
                            args.device_intermediate,
                            args.device_target,
                            args.query_window_size,
                            args.save_draft_query_vectors,
                            args.save_draft_attention_maps,
                        )
                        (
                            inter_topk_list,
                            target_topk_list,
                            accept_target_list,
                            accept_inter_list,
                            draft_tok_list,
                        ) = compute_position_stats(draft_tokens, logits_inter, logits_target, args.topk)
                        target_prefix_accept = [1 if all(accept_target_list[: j + 1]) else 0 for j in range(len(draft_tokens))]
                        inter_prefix_accept = [1 if all(accept_inter_list[: j + 1]) else 0 for j in range(len(draft_tokens))]
                        for j in range(len(draft_tokens)):
                            state["position_accept_target"][j].append(target_prefix_accept[j])
                            state["position_accept_inter"][j].append(inter_prefix_accept[j])
                        n_accept_target = acceptance_length(accept_target_list)
                        n_accept_inter = acceptance_length(accept_inter_list)
                        state["accept_len_target_list"].append(n_accept_target)
                        state["accept_len_inter_list"].append(n_accept_inter)
                        round_accept_rate_target = n_accept_target / len(draft_tokens) if draft_tokens else 0.0
                        round_accept_rate_inter = n_accept_inter / len(draft_tokens) if draft_tokens else 0.0

                        target_recovery = int(logits_target[0, n_accept_target + 1].argmax(dim=-1).item())
                        inter_recovery = get_recovery_token_from_logits(logits_inter, n_accept_inter, args.k)

                        target_accepted_token_stats: list[dict[str, Any]] = []
                        for j in range(n_accept_target):
                            token_stats = _token_and_topk_probs_from_logits(
                                logits_target[0, j + 1],
                                draft_tokens[j],
                                topk=args.topk,
                            )
                            token_stats["position"] = j
                            target_accepted_token_stats.append(token_stats)

                        target_recovery_logits_idx = n_accept_target + 1
                        target_recovery_stats = _token_and_topk_probs_from_logits(
                            logits_target[0, target_recovery_logits_idx],
                            target_recovery,
                            topk=args.topk,
                        )
                        target_recovery_stats["position"] = target_recovery_logits_idx - 1

                        precision_stats = compute_intermediate_precision_against_target(
                            draft_tokens=draft_tokens,
                            n_accept_inter=n_accept_inter,
                            n_accept_target=n_accept_target,
                            target_recovery=target_recovery,
                        )
                        state["inter_precision_tp_total"] += int(precision_stats["true_positive_count"])
                        state["inter_precision_fp_total"] += int(precision_stats["false_positive_count"])
                        if precision_stats["precision"] is not None:
                            state["inter_precision_round_values"].append(float(precision_stats["precision"]))
                            state["inter_precision_nonempty_rounds"] += 1

                        dataset_global_round = state["total_rounds"]
                        round_idx_for_row = request_round_idx
                        round_sample_id = f"{dataset_key}:{request_id}:round_{round_idx_for_row}"

                        if draft_query_vectors_path is not None and draft_start_context is not None:
                            draft_query_vector_records.append(
                                {
                                    "dataset": dataset_key,
                                    "request_index": request_counter,
                                    "request_sample_id": request_sample_id,
                                    "sample_id": round_sample_id,
                                    "request_id": request_id,
                                    "prompt_style": prompt_meta["prompt_style"],
                                    "turn_index_used": turn_index,
                                    "num_turns_in_record": prompt_meta["num_turns_in_record"],
                                    "history_mode": prompt_meta["history_mode"],
                                    "history_assistant_turns_generated": prompt_meta["history_assistant_turns_generated"],
                                    "verification_round": round_idx_for_row,
                                    "dataset_global_round": dataset_global_round,
                                    "overall_global_round": overall_round,
                                    "context_tokens_initial": initial_context_tokens,
                                    "context_tokens_before_round": context_tokens_before_round,
                                    "query_window_size": args.query_window_size,
                                    **draft_start_context,
                                }
                            )

                        if draft_attention_maps_path is not None:
                            for attn_rec in draft_attention_records:
                                draft_attention_records_all.append(
                                    {
                                        "dataset": dataset_key,
                                        "request_index": request_counter,
                                        "request_sample_id": request_sample_id,
                                        "sample_id": round_sample_id,
                                        "request_id": request_id,
                                        "prompt_style": prompt_meta["prompt_style"],
                                        "turn_index_used": turn_index,
                                        "verification_round": round_idx_for_row,
                                        "dataset_global_round": dataset_global_round,
                                        "overall_global_round": overall_round,
                                        **attn_rec,
                                    }
                                )

                        verification_row = {
                            "request_index": request_counter,
                            "request_sample_id": request_sample_id,
                            "sample_id": round_sample_id,
                            "prompt_style": prompt_meta["prompt_style"],
                            "turn_index_used": turn_index,
                            "num_turns_in_record": prompt_meta["num_turns_in_record"],
                            "history_mode": prompt_meta["history_mode"],
                            "history_assistant_turns_generated": prompt_meta["history_assistant_turns_generated"],
                            "dataset": dataset_key,
                            "verification_round": round_idx_for_row,
                            "context_tokens_initial": initial_context_tokens,
                            "context_tokens_before_round": context_tokens_before_round,
                            "dataset_global_round": dataset_global_round,
                            "overall_global_round": overall_round,
                            "k": len(draft_tokens),
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
                        if verification_f is not None:
                            verification_f.write(json.dumps(verification_row, ensure_ascii=False) + "\n")
                            verification_f.flush()
                        request_round_idx += 1

                        if args.print_round_progress and (
                            round_idx_for_row < args.print_first_n_rounds
                            or (args.progress_every_rounds > 0 and request_round_idx % args.progress_every_rounds == 0)
                        ):
                            prec = precision_stats["precision"]
                            prec_str = "null" if prec is None else f"{prec:.4f}"
                            _rank0_print(
                                f"[{dataset_key}] round sample={request_id} turn={turn_index} round={round_idx_for_row} ctx={context_tokens_before_round} accept_t={n_accept_target}/{len(draft_tokens)} accept_i={n_accept_inter}/{len(draft_tokens)} inter_prec={prec_str} tp={precision_stats['true_positive_count']} fp={precision_stats['false_positive_count']}",
                                flush=True,
                            )

                        if pending_inter_recovery is not None:
                            state["diff_accept_len_with_next_round_count"] += 1
                            in_next_draft = pending_inter_recovery in draft_tokens
                            if in_next_draft:
                                state["diff_accept_len_inter_bonus_in_next_draft_count"] += 1
                                first_j = next(j for j in range(len(draft_tokens)) if draft_tokens[j] == pending_inter_recovery)
                                if first_j < n_accept_target:
                                    state["diff_accept_len_inter_bonus_accept_next_target_count"] += 1
                            pending_inter_recovery = None
                        if n_accept_target == n_accept_inter:
                            state["same_accept_len_count"] += 1
                            if target_recovery == inter_recovery:
                                state["same_accept_len_bonus_same_count"] += 1
                        else:
                            state["diff_accept_len_count"] += 1
                            pending_inter_recovery = inter_recovery

                        if detail_f is not None:
                            for j, (dt, itopk, ttopk, at, ai) in enumerate(
                                zip(draft_tok_list, inter_topk_list, target_topk_list, accept_target_list, accept_inter_list)
                            ):
                                step_stat = draft_step_stats[j]
                                target_pos_stats = _token_and_topk_probs_from_logits(logits_target[0, j + 1], dt, topk=args.topk)
                                inter_pos_stats = _token_and_topk_probs_from_logits(logits_inter[0, j + 1], dt, topk=args.topk)
                                detail_row = {
                                    "dataset": dataset_key,
                                    "request_index": request_counter,
                                    "request_sample_id": request_sample_id,
                                    "sample_id": round_sample_id,
                                    "prompt_style": prompt_meta["prompt_style"],
                                    "turn_index_used": turn_index,
                                    "num_turns_in_record": prompt_meta["num_turns_in_record"],
                                    "history_mode": prompt_meta["history_mode"],
                                    "history_assistant_turns_generated": prompt_meta["history_assistant_turns_generated"],
                                    "verification_round": round_idx_for_row,
                                    "dataset_global_round": dataset_global_round,
                                    "overall_global_round": overall_round,
                                    "position": j,
                                    "draft_token_id": dt,
                                    "draft_top1_prob": step_stat["draft_top1_prob"],
                                    "draft_top5_token_ids": step_stat["draft_top5_token_ids"],
                                    "draft_top5_probs": step_stat["draft_top5_probs"],
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
                                }
                                detail_f.write(json.dumps(detail_row, ensure_ascii=False) + "\n")
                                state["detail_rows_written"] += 1

                        state["total_rounds"] += 1
                        overall_round += 1

                        emitted_tokens_full = [current_recovery] + draft_tokens[:n_accept_target]
                        remaining_budget = max_new_tokens - generated_count
                        committed_tokens = emitted_tokens_full[: max(remaining_budget, 0)]
                        committed_tokens, hit_stop_in_committed = _truncate_at_stop_token(committed_tokens, stop_ids)
                        prompt_ids_for_round = prompt_ids_for_round + committed_tokens
                        generated_output_ids.extend(int(tok) for tok in committed_tokens)
                        generated_count += len(committed_tokens)
                        max_context_seen = max(max_context_seen, len(prompt_ids_for_round))

                        if hit_stop_in_committed:
                            break

                        if len(committed_tokens) < len(emitted_tokens_full):
                            break

                        current_recovery = target_recovery

                        if generated_count >= max_new_tokens:
                            break
                        if current_recovery in stop_ids:
                            if generated_count < max_new_tokens:
                                prompt_ids_for_round = prompt_ids_for_round + [current_recovery]
                                generated_output_ids.append(int(current_recovery))
                                generated_count += 1
                                max_context_seen = max(max_context_seen, len(prompt_ids_for_round))
                            break

                    final_context_tokens = len(prompt_ids_for_round)
                    max_context_seen = max(max_context_seen, final_context_tokens)
                    state["final_context_tokens"].append(final_context_tokens)
                    state["max_context_tokens_seen"].append(max_context_seen)
                    state["generated_tokens_per_request"].append(generated_count)
                    state["verification_rounds_per_request"].append(request_round_idx)
                    current_turn_text = str(record["turns"][turn_index])
                    generated_text = _decode_generated_ids(tokenizer, generated_output_ids)
                    response_row = {
                        "dataset": dataset_key,
                        "request_index": request_counter,
                        "request_sample_id": request_sample_id,
                        "sample_id": request_id,
                        "prompt_style": prompt_meta["prompt_style"],
                        "turn_index_used": turn_index,
                        "num_turns_in_record": prompt_meta["num_turns_in_record"],
                        "history_mode": prompt_meta["history_mode"],
                        "history_assistant_turns_generated": prompt_meta["history_assistant_turns_generated"],
                        "question": current_turn_text,
                        "generated_token_ids": [int(tok) for tok in generated_output_ids],
                        "generated_text": generated_text,
                        "generated_tokens": generated_count,
                        "num_verification_rounds": request_round_idx,
                    }
                    request_meta_row = {
                        "dataset": dataset_key,
                        "request_index": request_counter,
                        "request_sample_id": request_sample_id,
                        "sample_id": request_id,
                        "prompt_style": prompt_meta["prompt_style"],
                        "turn_index_used": turn_index,
                        "num_turns_in_record": prompt_meta["num_turns_in_record"],
                        "history_mode": prompt_meta["history_mode"],
                        "history_assistant_turns_generated": prompt_meta["history_assistant_turns_generated"],
                        "context_tokens_initial": initial_context_tokens,
                        "context_tokens_final": final_context_tokens,
                        "max_context_tokens_seen": max_context_seen,
                        "generated_tokens": generated_count,
                        "num_verification_rounds": request_round_idx,
                    }
                    if response_f is not None:
                        response_f.write(json.dumps(response_row, ensure_ascii=False) + "\n")
                        response_f.flush()
                    if request_meta_f is not None:
                        request_meta_f.write(json.dumps(request_meta_row, ensure_ascii=False) + "\n")
                        request_meta_f.flush()
                    if args.print_request_progress:
                        _rank0_print(
                            f"[{dataset_key}] done request {request_counter + 1}/{total_requests} sample={request_id} rounds={request_round_idx} generated={generated_count} initial_ctx={initial_context_tokens} final_ctx={final_context_tokens} max_ctx={max_context_seen}",
                            flush=True,
                        )
                    request_counter += 1
    finally:
        if verification_f is not None:
            verification_f.close()
        if request_meta_f is not None:
            request_meta_f.close()
        if response_f is not None:
            response_f.close()
        if detail_f is not None:
            detail_f.close()

    if draft_query_vectors_path is not None and _is_main_process():
        torch.save(
            {
                "format_version": 1,
                "dataset": dataset_key,
                "draft_model": args.draft,
                "query_vector_definition": "Last-layer q_proj output before RoPE at the draft-start conditioning step",
                "query_window_size": args.query_window_size,
                "records": draft_query_vector_records,
            },
            draft_query_vectors_path,
        )

    if draft_attention_maps_path is not None and _is_main_process():
        torch.save(
            {
                "format_version": 1,
                "dataset": dataset_key,
                "draft_model": args.draft,
                "attention_definition": "For each drafted position, sum the last-query attention map across all draft layers and heads, then apply softmax over source positions.",
                "records": draft_attention_records_all,
            },
            draft_attention_maps_path,
        )

    summary = build_summary(
        args=args,
        data_dir=data_dir,
        dataset_keys=[dataset_key],
        num_prompts=total_requests,
        state=state,
        summary_scope="per_dataset",
    )
    summary_path = save_summary(summary, dataset_out_dir) if _is_main_process() else (dataset_out_dir / "intermediate_vs_target_summary.json")
    return summary, state, overall_round, summary_path, detail_path, draft_query_vectors_path, draft_attention_maps_path, response_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile intermediate vs target verifier on local data directory datasets.")
    parser.add_argument("--draft", type=str, default=_model_path("draft", DEFAULT_DRAFT), help="Draft model HF id")
    parser.add_argument("--intermediate", type=str, default=_model_path("intermediate", DEFAULT_INTERMEDIATE), help="Intermediate verifier HF id")
    parser.add_argument("--target", type=str, default=_model_path("target", DEFAULT_TARGET), help="Target model HF id")
    parser.add_argument("--output-dir", type=str, default="profile/results", help="Directory to save stats and per-position data")
    parser.add_argument("--datasets", type=str, default="alpaca,gsm8k,humaneval,mt_bench,qa", help="Comma-separated dataset keys")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SSD_DATASET_DIR", ""), help="Root directory containing dataset jsonl files/subdirs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset.")
    parser.add_argument("--alpaca-max-new-tokens", type=int, default=1024)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=1024)
    parser.add_argument("--humaneval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--mt-bench-max-new-tokens", type=int, default=1024)
    parser.add_argument("--qa-max-new-tokens", type=int, default=1024)
    parser.add_argument("--k", type=int, default=5, help="Number of draft tokens per round")
    parser.add_argument("--chat-template", action="store_true", default=True, help="Apply tokenizer chat template when relevant (default: True)")
    parser.add_argument("--no-chat-template", action="store_false", dest="chat_template")
    parser.add_argument("--prompt-style", type=str, default="auto", choices=["auto", "vicuna", "llama3_instruct", "generic"], help="Prompt/tokenization path.")
    parser.add_argument("--conversation-turn-index", type=int, default=0, help="For multi-turn question files, profile this user turn. Use -1 for the last turn.")
    parser.add_argument("--conversation-turn-mode", type=str, default="all", choices=["all", "selected"], help="Profile all user turns in each record, or only the selected turn index. Default: all.")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Truncate prompt to this length")
    parser.add_argument("--topk", type=int, default=5, help="Top-k tokens to record per position for intermediate/target")
    parser.add_argument("--query-window-size", type=int, default=8, help="Number of recent tokens and query vectors to save at draft start")
    parser.add_argument("--device-draft", type=str, default="auto", help="Draft model device. Use auto to place on the local GPU or CPU.")
    parser.add_argument("--device-intermediate", type=str, default="auto", help="Intermediate verifier device. Use auto to place on the local GPU or CPU.")
    parser.add_argument("--device-target", type=str, default="auto", help="Target model device. Use auto to place on the local GPU or CPU.")
    parser.add_argument("--target-tp-size", type=int, default=1, help="Tensor-parallel degree for the target model. Use 4 with torchrun --nproc_per_node 4 for Llama 70B.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed used for reproducible runs")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic backend settings (default: True)",
    )
    parser.add_argument("--save-per-position-detail", action="store_true", default=True, help="Save per-sample per-position details, including draft top-1 and top-5 probabilities")
    parser.add_argument("--no-save-per-position-detail", action="store_false", dest="save_per_position_detail")
    parser.add_argument("--save-draft-query-vectors", action="store_true", default=True, help="Save draft-start token ids and query vectors to draft_query_vectors.pt")
    parser.add_argument("--no-save-draft-query-vectors", action="store_false", dest="save_draft_query_vectors")
    parser.add_argument("--save-draft-attention-maps", action="store_true", default=True, help="Save per-draft-position summed attention maps to draft_attention_maps.pt")
    parser.add_argument("--no-save-draft-attention-maps", action="store_false", dest="save_draft_attention_maps")
    parser.add_argument("--print-request-progress", action="store_true", default=True, help="Print per-request start/end progress logs")
    parser.add_argument("--no-print-request-progress", action="store_false", dest="print_request_progress")
    parser.add_argument("--print-round-progress", action="store_true", default=True, help="Print periodic per-round progress logs")
    parser.add_argument("--no-print-round-progress", action="store_false", dest="print_round_progress")
    parser.add_argument("--progress-every-rounds", type=int, default=10, help="Print per-round progress every N verification rounds after the first few")
    parser.add_argument("--print-first-n-rounds", type=int, default=3, help="Always print the first N verification rounds for each request")
    parser.add_argument("--verification-jsonl", type=str, default="verification_metrics.jsonl", help="Per-verification metrics JSONL filename under each output directory.")
    args = parser.parse_args()

    configure_reproducibility(seed=args.seed, deterministic=args.deterministic)

    local_rank = _maybe_init_target_tp(args.target_tp_size)
    args.device_draft = _resolve_device_arg(args.device_draft, fallback_cuda_index=local_rank)
    args.device_intermediate = _resolve_device_arg(args.device_intermediate, fallback_cuda_index=local_rank)
    args.device_target = _resolve_device_arg(args.device_target, fallback_cuda_index=local_rank)

    args.output_dir = str(build_model_run_dir(args.output_dir, args.draft, args.intermediate, args.target))
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
    _rank0_print(f"Loaded {total_prompts} profiled requests from {list(dataset_prompts.keys())}")

    tokenizer = get_tokenizer(args.target)
    resolved_prompt_style = resolve_prompt_style(args.prompt_style, args.target)
    _rank0_print(f"Prompt style: {resolved_prompt_style} (requested={args.prompt_style}, target={args.target})")

    def _load_causal_lm(path: str, device: str, *, tp_size: int = 1):
        load_kwargs = {"torch_dtype": torch.bfloat16}
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
    overall_round = 0
    for dataset_key in dataset_keys:
        prompts = dataset_prompts.get(dataset_key, [])
        if not prompts:
            _rank0_print(f"[skip] {dataset_key}: no prompts")
            continue
        _rank0_print(f"[run] dataset={dataset_key} -> {Path(args.output_dir) / dataset_key}")
        summary, dataset_state, overall_round, summary_path, detail_path, draft_query_vectors_path, draft_attention_maps_path, response_path = run_dataset_profile(
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
        _rank0_print(f"Wrote dataset summary to {summary_path}")
        _rank0_print(
            f"[{dataset_key}] Avg acceptance length (target/intermediate): {summary['avg_acceptance_length_target']:.4f} / {summary['avg_acceptance_length_intermediate']:.4f}"
        )
        _rank0_print(
            f"[{dataset_key}] Intermediate precision vs target: {summary['intermediate_precision_vs_target']['precision']:.4f} "
            f"(tp={summary['intermediate_precision_vs_target']['true_positive_count']}, "
            f"fp={summary['intermediate_precision_vs_target']['false_positive_count']})"
        )
        if detail_path is not None:
            _rank0_print(f"Wrote per-position detail ({dataset_state['detail_rows_written']} rows) to {detail_path}")
        if draft_query_vectors_path is not None:
            _rank0_print(f"Wrote draft query vectors to {draft_query_vectors_path}")
        if draft_attention_maps_path is not None:
            _rank0_print(f"Wrote draft attention maps to {draft_attention_maps_path}")
        if response_path is not None:
            _rank0_print(f"Wrote responses to {response_path}")

    overall_summary = build_summary(
        args=args,
        data_dir=data_dir,
        dataset_keys=list(dataset_prompts.keys()),
        num_prompts=total_prompts,
        state=overall_state,
        summary_scope="aggregate",
    )
    overall_summary_path = save_summary(overall_summary, Path(args.output_dir)) if _is_main_process() else (Path(args.output_dir) / "intermediate_vs_target_summary.json")
    _rank0_print(f"Wrote aggregate summary to {overall_summary_path}")
    _rank0_print("Aggregate position-wise accept rate (target):", overall_summary["position_accept_rate_target"])
    _rank0_print("Aggregate position-wise accept rate (intermediate):", overall_summary["position_accept_rate_intermediate"])
    _rank0_print("Aggregate avg acceptance length (target):", overall_summary["avg_acceptance_length_target"])
    _rank0_print("Aggregate avg acceptance length (intermediate):", overall_summary["avg_acceptance_length_intermediate"])
    _rank0_print(
        "Aggregate intermediate precision vs target:",
        overall_summary["intermediate_precision_vs_target"],
    )
    _maybe_barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
