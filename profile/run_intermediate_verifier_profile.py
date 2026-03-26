#!/usr/bin/env python3
"""
Profile script: Draft -> serial Intermediate verifier -> Target verifier.

Compares position-wise:
- Intermediate verifier vs Target: top-5 tokens per position, acceptance rate, avg acceptance length.
- Accept decision is always made by Target top-1; we record what Intermediate would have done.

Default models:
  Draft: double7/vicuna-68m
  Intermediate verifier: lmsys/vicuna-7b-v1.3
  Target: lmsys/vicuna-13b-v1.3

Datasets are read from local data directory jsonl files:
  alpaca, gsm8k, humaneval, mt_bench, qa

Usage:
  python profile/run_intermediate_verifier_profile.py \
    --data-dir /path/to/processed_datasets \
    --datasets alpaca,gsm8k,humaneval,mt_bench,qa \
    --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

# Add project root for optional ssd.paths
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("Install transformers: pip install transformers")

# Default model names (HF ids; override with env SSD_PROFILE_*_MODEL or --draft/--intermediate/--target)
def _model_path(name: str, default_hf: str) -> str:
    env = os.environ.get(f"SSD_PROFILE_{name.upper()}_MODEL")
    return env if env else default_hf


DEFAULT_DRAFT = "double7/vicuna-68m"
DEFAULT_INTERMEDIATE = "lmsys/vicuna-7b-v1.3"
DEFAULT_TARGET = "lmsys/vicuna-13b-v1.3"

DATASET_KEYS_SUPPORTED = ["alpaca", "gsm8k", "humaneval", "mt_bench", "qa"]


def extract_first_present(example: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        if key in example and example[key] is not None:
            return str(example[key])
    return default


def build_alpaca_prompt(example: dict[str, Any]) -> str:
    instruction = extract_first_present(example, ["instruction", "text", "prompt", "question"])
    input_text = extract_first_present(example, ["input"], default="")
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}\n\nAnswer:"
    return f"{instruction}\n\nAnswer:"


def build_gsm8k_prompt(example: dict[str, Any]) -> str:
    question = extract_first_present(example, ["question", "text", "problem", "prompt"])
    return f"Solve the following math word problem.\n\nQuestion:\n{question}\n\nAnswer:"


def build_humaneval_prompt(example: dict[str, Any]) -> str:
    prompt = extract_first_present(example, ["prompt", "text", "question"])
    return prompt


def build_mt_bench_prompt(example: dict[str, Any]) -> str:
    turns = example.get("turns")
    if isinstance(turns, list) and turns:
        user_query = str(turns[0])
    else:
        user_query = extract_first_present(example, ["question", "prompt", "text"])
    return f"{user_query}\n\nAnswer:"


def build_qa_prompt(example: dict[str, Any]) -> str:
    question = extract_first_present(example, ["question", "prompt", "text"])
    context = extract_first_present(example, ["context", "passage"], default="")
    if context:
        return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return f"Question:\n{question}\n\nAnswer:"


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
        raise FileNotFoundError(
            f"Could not find jsonl for dataset '{dataset_key}' under {data_dir}"
        )
    return sorted(candidates)[0]


def get_dataset_prompts(
    dataset_key: str,
    data_dir: Path,
    max_samples: int | None,
) -> list[tuple[str, str, str]]:
    """Returns list of (prompt_text, sample_id, dataset_key)."""
    out: list[tuple[str, str, str]] = []
    if dataset_key not in DATASET_KEYS_SUPPORTED:
        raise ValueError(f"Unsupported dataset key: {dataset_key}. Supported={DATASET_KEYS_SUPPORTED}")
    dataset_path = _resolve_dataset_jsonl(data_dir, dataset_key)
    rows = _read_jsonl(dataset_path, max_samples=max_samples)
    for i, x in enumerate(rows):
        if dataset_key == "alpaca":
            prompt = build_alpaca_prompt(x)
        elif dataset_key == "gsm8k":
            prompt = build_gsm8k_prompt(x)
        elif dataset_key == "humaneval":
            prompt = build_humaneval_prompt(x)
        elif dataset_key == "mt_bench":
            prompt = build_mt_bench_prompt(x)
        else:
            prompt = build_qa_prompt(x)
        sid = extract_first_present(x, ["id", "name", "task_id"], str(i))
        out.append((prompt, sid, dataset_key))
    print(f"[dataset] {dataset_key}: path={dataset_path}, samples={len(out)}")
    return out


def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, use_fast=True)


def encode_prompt(tokenizer, prompt_text: str, use_chat_template: bool, max_prompt_tokens: int | None):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    if max_prompt_tokens is not None and len(tokens) > max_prompt_tokens:
        tokens = tokens[:max_prompt_tokens]
    return tokens


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


def get_logits_at_positions(model, input_ids: torch.Tensor, positions: list[int], device):
    """Run one forward and return logits at the given position indices (0-indexed)."""
    inp = input_ids.to(device)
    with torch.inference_mode():
        out = model(inp)
    logits = out.logits  # [1, seq_len, vocab_size]
    return torch.stack([logits[0, p] for p in positions], dim=0).unsqueeze(0)  # [1, len(positions), V]


def run_one_verify_round(
    draft_model,
    inter_model,
    target_model,
    tokenizer,
    prompt_ids: list[int],
    recovery_token_id: int,
    k: int,
    device_draft,
    device_inter,
    device_target,
    topk: int = 5,
):
    """
    Run: draft produces K tokens given (prompt + recovery). Then intermediate and target
    run one forward on (prompt + recovery + draft_0 .. draft_{K-1}) and return logits at K+1 positions.

    Returns:
      draft_tokens: list of K ints (draft_0 .. draft_{K-1})
      logits_inter: [1, K+1, V] at positions predicting (recovery, draft_0, ..., draft_{K-1})
      logits_target: [1, K+1, V]
    """
    L = len(prompt_ids)
    # Build input for draft: prompt + recovery
    seq = list(prompt_ids) + [recovery_token_id]
    draft_tokens = []

    for j in range(k):
        inp = torch.tensor([seq], dtype=torch.long, device=device_draft)
        with torch.inference_mode():
            out = draft_model(inp)
        next_logits = out.logits[0, -1]
        next_tok = next_logits.argmax(dim=-1).item()
        draft_tokens.append(next_tok)
        seq.append(next_tok)

    # seq = prompt_ids + [recovery] + draft_tokens (length L + 1 + K)
    # We need logits at positions L-1, L, L+1, ..., L+K-1 (predicting recovery, draft_0, ..., draft_{K-1})
    positions_needed = list(range(L - 1, L + k))
    full_seq = list(prompt_ids) + [recovery_token_id] + draft_tokens

    # Use CPU tensor and .to(device) so draft/inter/target can be on different GPUs (no cross-device tensors)
    inp_full = torch.tensor([full_seq], dtype=torch.long)
    logits_inter = get_logits_at_positions(inter_model, inp_full, positions_needed, device_inter)
    logits_target = get_logits_at_positions(target_model, inp_full, positions_needed, device_target)

    return draft_tokens, logits_inter, logits_target


def compute_position_stats(
    draft_tokens: list[int],
    logits_inter: torch.Tensor,
    logits_target: torch.Tensor,
    topk: int,
):
    """
    For positions j in 0..K-1 (draft positions):
    - draft_tok = draft_tokens[j]
    - inter_top5, target_top5
    - accept_by_target = (draft_tok == target_top1)
    - accept_by_intermediate = (draft_tok == inter_top1)

    Returns lists (one per position): inter_top5, target_top5, accept_target, accept_inter, draft_tok.
    """
    K = len(draft_tokens)
    # logits_inter / logits_target: [1, K+1, V]. Index 0 predicts recovery; indices 1..K predict draft_0..draft_{K-1}
    inter_top5_list = []
    target_top5_list = []
    accept_target_list = []
    accept_inter_list = []
    draft_tok_list = []

    for j in range(K):
        draft_tok = draft_tokens[j]
        # logits at index j+1 predict draft_tokens[j]
        inter_logits_j = logits_inter[0, j + 1]
        target_logits_j = logits_target[0, j + 1]
        inter_top5 = inter_logits_j.topk(min(topk, inter_logits_j.size(0))).indices.cpu().tolist()
        target_top5 = target_logits_j.topk(min(topk, target_logits_j.size(0))).indices.cpu().tolist()
        inter_top1 = inter_top5[0]
        target_top1 = target_top5[0]

        accept_target_list.append(1 if draft_tok == target_top1 else 0)
        accept_inter_list.append(1 if draft_tok == inter_top1 else 0)
        inter_top5_list.append(inter_top5)
        target_top5_list.append(target_top5)
        draft_tok_list.append(draft_tok)

    return inter_top5_list, target_top5_list, accept_target_list, accept_inter_list, draft_tok_list


def acceptance_length(accept_list: list[int]) -> int:
    """First index where accept is 0, or len(accept_list) if all accept.
    This is the number of consecutive accepted draft tokens in one verify round.
    """
    for i, a in enumerate(accept_list):
        if a == 0:
            return i
    return len(accept_list)


# --- Metric definitions (for summary and user reference) ---
# Position-wise acceptance rate at position j:
#   = (rounds where positions 0, 1, ..., j were ALL accepted) / (total rounds).
#   If any position before j rejected, that round counts as reject at j.
#   So rate at j = P(accept at 0 and 1 and ... and j) -> 0.8, 0.64, 0.512, ... when conditional p=0.8.
#
# Average acceptance length:
#   = mean over rounds of (first rejection index, or K if all accepted).
#   E[length] = rate[0] + rate[1] + ... + rate[K-1] (with the above definition of rate).


def get_recovery_token_from_logits(logits: torch.Tensor, accept_len: int, k: int) -> int:
    """Recovery (bonus) token: model's prediction at the position after accepted prefix.
    logits: [1, K+1, V]. Index 0 predicts recovery; indices 1..K predict draft_0..draft_{K-1}.
    If accept_len < K: prediction at position accept_len+1 (what to use after rejecting at accept_len).
    If accept_len == K: prediction at index K (next token after all K accepted).
    """
    if accept_len < k:
        # reject at position accept_len; recovery = prediction at that position (index accept_len+1)
        return logits[0, accept_len + 1].argmax(dim=-1).item()
    return logits[0, k].argmax(dim=-1).item()


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Profile intermediate vs target verifier on local data directory datasets."
    )
    parser.add_argument("--draft", type=str, default=_model_path("draft", DEFAULT_DRAFT), help="Draft model HF id (default: double7/vicuna-68m)")
    parser.add_argument("--intermediate", type=str, default=_model_path("intermediate", DEFAULT_INTERMEDIATE), help="Intermediate verifier HF id")
    parser.add_argument("--target", type=str, default=_model_path("target", DEFAULT_TARGET), help="Target model HF id")
    parser.add_argument("--output-dir", type=str, default="profile/results", help="Directory to save stats and per-position data")
    parser.add_argument(
        "--datasets",
        type=str,
        default="alpaca,gsm8k,humaneval,mt_bench,qa",
        help="Comma-separated dataset keys: alpaca,gsm8k,humaneval,mt_bench,qa",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("SSD_DATASET_DIR", ""),
        help="Root directory containing dataset jsonl files/subdirs.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset.")
    parser.add_argument("--alpaca-max-new-tokens", type=int, default=256)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=256)
    parser.add_argument("--humaneval-max-new-tokens", type=int, default=512)
    parser.add_argument("--mt-bench-max-new-tokens", type=int, default=512)
    parser.add_argument("--qa-max-new-tokens", type=int, default=256)
    parser.add_argument("--k", type=int, default=5, help="Number of draft tokens per round")
    parser.add_argument("--chat-template", action="store_true", default=True, help="Apply chat template (default: True)")
    parser.add_argument("--no-chat-template", action="store_false", dest="chat_template")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Truncate prompt to this length")
    parser.add_argument("--topk", type=int, default=5, help="Top-k tokens to record per position")
    parser.add_argument("--device-draft", type=str, default="cuda:0", help="Draft model device (multi-GPU safe)")
    parser.add_argument("--device-intermediate", type=str, default="cuda:0", help="Intermediate verifier device")
    parser.add_argument("--device-target", type=str, default="cuda:0", help="Target model device (e.g. cuda:1)")
    parser.add_argument("--save-per-position-detail", action="store_true", help="Save per-sample per-position top-5 details (can be large)")
    parser.add_argument(
        "--verification-jsonl",
        type=str,
        default="verification_metrics.jsonl",
        help="Per-verification metrics JSONL filename under output-dir.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.data_dir:
        raise ValueError("--data-dir is required (or set SSD_DATASET_DIR)")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    verification_jsonl_path = os.path.join(args.output_dir, args.verification_jsonl)

    dataset_keys = parse_csv_list(args.datasets)
    prompts_and_ids: list[tuple[str, str, str]] = []  # (prompt_text, sample_id, dataset_key)
    for key in dataset_keys:
        prompts_and_ids.extend(
            get_dataset_prompts(key, data_dir, args.max_samples)
        )
    if not prompts_and_ids:
        print("No prompts loaded. Exiting.")
        return 1
    print(f"Loaded {len(prompts_and_ids)} prompts from {dataset_keys}")

    tokenizer = get_tokenizer(args.target)
    # Load without device_map so accelerate is not required; move to device with .to()
    def _load_causal_lm(path: str, device: str):
        try:
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        return model.to(device)

    print("Loading draft model...")
    draft_model = _load_causal_lm(args.draft, args.device_draft)
    print("Loading intermediate model...")
    inter_model = _load_causal_lm(args.intermediate, args.device_intermediate)
    print("Loading target model...")
    target_model = _load_causal_lm(args.target, args.device_target)

    # Aggregates over all rounds and samples
    position_accept_target: dict[int, list[int]] = defaultdict(list)   # position -> list of 0/1
    position_accept_inter: dict[int, list[int]] = defaultdict(list)
    accept_len_target_list: list[int] = []
    accept_len_inter_list: list[int] = []
    per_position_details: list[dict] = [] if args.save_per_position_detail else None

    # 1) Same accept length: P(same length), P(bonus token same | same length)
    same_accept_len_count = 0
    same_accept_len_bonus_same_count = 0
    # 2) Different accept length: inter's bonus in next draft? accepted by next target?
    diff_accept_len_count = 0
    diff_accept_len_inter_bonus_in_next_draft_count = 0
    diff_accept_len_inter_bonus_accept_next_target_count = 0
    diff_accept_len_with_next_round_count = 0  # only count when we have a next round
    pending_inter_recovery: int | None = None

    total_rounds = 0
    with open(verification_jsonl_path, "w", encoding="utf-8") as verification_f:
        for sample_idx, (prompt_text, sample_id, dataset_key) in enumerate(prompts_and_ids):
            pending_inter_recovery = None  # reset per sample so we don't carry over across prompts
            max_new_tokens = get_dataset_max_new_tokens(
                dataset_key,
                args.alpaca_max_new_tokens,
                args.gsm8k_max_new_tokens,
                args.humaneval_max_new_tokens,
                args.mt_bench_max_new_tokens,
                args.qa_max_new_tokens,
            )
            prompt_ids = encode_prompt(tokenizer, prompt_text, args.chat_template, args.max_prompt_tokens)
            if not prompt_ids:
                continue

            # First token from target (prefill)
            inp = torch.tensor([prompt_ids], dtype=torch.long, device=args.device_target)
            with torch.inference_mode():
                out = target_model(inp)
            recovery_token_id = out.logits[0, -1].argmax(dim=-1).item()

            generated_count = 0
            current_recovery = recovery_token_id
            prompt_ids_for_round = list(prompt_ids)
            request_round_idx = 0

            while generated_count < max_new_tokens:
                draft_tokens, logits_inter, logits_target = run_one_verify_round(
                    draft_model,
                    inter_model,
                    target_model,
                    tokenizer,
                    prompt_ids_for_round,
                    current_recovery,
                    args.k,
                    args.device_draft,
                    args.device_intermediate,
                    args.device_target,
                    topk=args.topk,
                )

                inter_top5_list, target_top5_list, accept_target_list, accept_inter_list, draft_tok_list = compute_position_stats(
                    draft_tokens, logits_inter, logits_target, args.topk
                )

                # Position-wise rate: accept at j only if 0..j all accepted (else treat as reject)
                # So rate at j = P(accept at 0 and 1 and ... and j) -> 0.8, 0.64, 0.512, ...
                target_prefix_accept = [
                    1 if all(accept_target_list[: j + 1]) else 0
                    for j in range(len(draft_tokens))
                ]
                inter_prefix_accept = [
                    1 if all(accept_inter_list[: j + 1]) else 0
                    for j in range(len(draft_tokens))
                ]
                for j in range(len(draft_tokens)):
                    position_accept_target[j].append(target_prefix_accept[j])
                    position_accept_inter[j].append(inter_prefix_accept[j])

                n_accept_target = acceptance_length(accept_target_list)
                n_accept_inter = acceptance_length(accept_inter_list)
                accept_len_target_list.append(n_accept_target)
                accept_len_inter_list.append(n_accept_inter)
                round_accept_rate_target = n_accept_target / len(draft_tokens) if draft_tokens else 0.0
                round_accept_rate_inter = n_accept_inter / len(draft_tokens) if draft_tokens else 0.0

                verification_row = {
                    "request_index": sample_idx,
                    "sample_id": sample_id,
                    "dataset": dataset_key,
                    "verification_round": request_round_idx,
                    "global_round": total_rounds,
                    "k": len(draft_tokens),
                    "target": {
                        "acceptance_per_position_raw": accept_target_list,
                        "acceptance_per_position_prefix": target_prefix_accept,
                        "acceptance_rate": round_accept_rate_target,
                        "acceptance_length": n_accept_target,
                    },
                    "intermediate": {
                        "acceptance_per_position_raw": accept_inter_list,
                        "acceptance_per_position_prefix": inter_prefix_accept,
                        "acceptance_rate": round_accept_rate_inter,
                        "acceptance_length": n_accept_inter,
                    },
                }
                verification_f.write(json.dumps(verification_row, ensure_ascii=False) + "\n")
                request_round_idx += 1

                # Target and intermediate recovery (bonus) tokens for this round
                if n_accept_target < len(draft_tokens):
                    target_recovery = logits_target[0, n_accept_target + 1].argmax(dim=-1).item()
                else:
                    target_recovery = logits_target[0, args.k].argmax(dim=-1).item()
                inter_recovery = get_recovery_token_from_logits(logits_inter, n_accept_inter, args.k)

                # 2) Consume pending: previous round had different length; check if inter's bonus is in this draft and accepted
                if pending_inter_recovery is not None:
                    diff_accept_len_with_next_round_count += 1
                    in_next_draft = pending_inter_recovery in draft_tokens
                    if in_next_draft:
                        diff_accept_len_inter_bonus_in_next_draft_count += 1
                        first_j = next(j for j in range(len(draft_tokens)) if draft_tokens[j] == pending_inter_recovery)
                        if first_j < n_accept_target:
                            diff_accept_len_inter_bonus_accept_next_target_count += 1
                    pending_inter_recovery = None

                # 1) Same accept length: count and P(bonus same | same length)
                if n_accept_target == n_accept_inter:
                    same_accept_len_count += 1
                    if target_recovery == inter_recovery:
                        same_accept_len_bonus_same_count += 1
                else:
                    diff_accept_len_count += 1
                    pending_inter_recovery = inter_recovery  # check in next round

                if args.save_per_position_detail:
                    for j, (dt, it5, tt5, at, ai) in enumerate(zip(draft_tok_list, inter_top5_list, target_top5_list, accept_target_list, accept_inter_list)):
                        per_position_details.append({
                            "sample_id": sample_id,
                            "round": total_rounds,
                            "position": j,
                            "draft_token_id": dt,
                            "intermediate_top5": it5,
                            "target_top5": tt5,
                            "accept_by_target": at,
                            "accept_by_intermediate": ai,
                        })

                total_rounds += 1

                # Advance by target accept: n_accept draft tokens accepted; recovery already computed as target_recovery
                current_recovery = target_recovery
                prompt_ids_for_round = prompt_ids_for_round + [current_recovery] + draft_tokens[:n_accept_target]
                generated_count += n_accept_target + 1
                # current_recovery already holds the new recovery for the next round's draft start

                if generated_count >= max_new_tokens or current_recovery == tokenizer.eos_token_id:
                    break

    # Summary stats
    max_pos = max(position_accept_target.keys()) if position_accept_target else 0
    position_avg_accept_target = {}
    position_avg_accept_inter = {}
    for j in range(max_pos + 1):
        if j in position_accept_target and position_accept_target[j]:
            position_avg_accept_target[j] = sum(position_accept_target[j]) / len(position_accept_target[j])
        if j in position_accept_inter and position_accept_inter[j]:
            position_avg_accept_inter[j] = sum(position_accept_inter[j]) / len(position_accept_inter[j])

    avg_accept_len_target = sum(accept_len_target_list) / len(accept_len_target_list) if accept_len_target_list else 0
    avg_accept_len_inter = sum(accept_len_inter_list) / len(accept_len_inter_list) if accept_len_inter_list else 0

    # 1) Same accept length: P(same length), P(bonus same | same length)
    prob_same_accept_length = same_accept_len_count / total_rounds if total_rounds else 0
    prob_bonus_same_given_same_length = (
        same_accept_len_bonus_same_count / same_accept_len_count if same_accept_len_count else 0
    )
    # 2) Different accept length (only rounds that had a next round): P(inter bonus in next draft), P(inter bonus accepted next target)
    prob_inter_bonus_in_next_draft = (
        diff_accept_len_inter_bonus_in_next_draft_count / diff_accept_len_with_next_round_count
        if diff_accept_len_with_next_round_count else 0
    )
    prob_inter_bonus_accept_next_target = (
        diff_accept_len_inter_bonus_accept_next_target_count / diff_accept_len_with_next_round_count
        if diff_accept_len_with_next_round_count else 0
    )

    summary = {
        "config": {
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
            "num_prompts": len(prompts_and_ids),
            "total_verify_rounds": total_rounds,
        },
        "metric_definitions": {
            "position_accept_rate": "At position j: (rounds where positions 0,1,...,j were ALL accepted) / total_rounds. If any earlier position rejected, round counts as reject at j. So rate j = P(accept 0 and ... and j) -> 0.8, 0.64, 0.512, ...",
            "avg_acceptance_length": "Per round: first index where draft != target top-1 (or K if all accepted). Mean over rounds. With position_accept_rate above, E[length] = rate[0]+rate[1]+...+rate[K-1].",
        },
        "position_accept_rate_target": position_avg_accept_target,
        "position_accept_rate_intermediate": position_avg_accept_inter,
        "avg_acceptance_length_target": avg_accept_len_target,
        "avg_acceptance_length_intermediate": avg_accept_len_inter,
        "per_position_counts": {str(j): len(position_accept_target[j]) for j in range(max_pos + 1) if j in position_accept_target},
        "same_accept_length": {
            "count": same_accept_len_count,
            "prob_same_accept_length": prob_same_accept_length,
            "bonus_same_count": same_accept_len_bonus_same_count,
            "prob_bonus_same_given_same_length": prob_bonus_same_given_same_length,
        },
        "different_accept_length": {
            "count": diff_accept_len_count,
            "count_with_next_round": diff_accept_len_with_next_round_count,
            "inter_bonus_in_next_draft_count": diff_accept_len_inter_bonus_in_next_draft_count,
            "prob_inter_bonus_in_next_draft": prob_inter_bonus_in_next_draft,
            "inter_bonus_accept_next_target_count": diff_accept_len_inter_bonus_accept_next_target_count,
            "prob_inter_bonus_accept_next_target": prob_inter_bonus_accept_next_target,
        },
    }

    out_path = os.path.join(args.output_dir, "intermediate_vs_target_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {out_path}")
    print("Position-wise accept rate (target):", position_avg_accept_target)
    print("Position-wise accept rate (intermediate):", position_avg_accept_inter)
    print("Avg acceptance length (target):", avg_accept_len_target)
    print("Avg acceptance length (intermediate):", avg_accept_len_inter)
    print("1) Same accept length: P(same) =", prob_same_accept_length, "| P(bonus same | same length) =", prob_bonus_same_given_same_length)
    print("2) Different accept length (with next round): P(inter bonus in next draft) =", prob_inter_bonus_in_next_draft, "| P(inter bonus accept next target) =", prob_inter_bonus_accept_next_target)
    print(f"Verification-level metrics JSONL: {verification_jsonl_path}")

    if per_position_details is not None and per_position_details:
        detail_path = os.path.join(args.output_dir, "per_position_top5_detail.jsonl")
        with open(detail_path, "w", encoding="utf-8") as f:
            for row in per_position_details:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote per-position detail ({len(per_position_details)} rows) to {detail_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
