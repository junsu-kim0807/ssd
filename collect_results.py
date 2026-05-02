#!/usr/bin/env python3
"""
Collect motivation, insight, trace, and cost results from speculative decoding runs.

Global input root:
  ./results

Key-specific roots:
  motivation, insight, trace: ./results/metadata
  results:                    ./results/cost

Expected metadata run layout:
  <input_root>/metadata/<method>/b<batch_size>/k<speculative_length>/<target>+<draft>/t<temperature>/<dataset>/
      analysis.jsonl
      metadata.jsonl

Expected cost run layout:
  <input_root>/cost/<method>/b<batch_size>/k<speculative_length>/<target>+<draft>/t<temperature>/<dataset>/
      cost_breakdown.json

Default:
  python collect_results.py

Default output:
  results.pt

Output top-level schema:
  {
    "motivation": [...],
    "insight": [...],
    "trace": [...],
    "results": [...]
  }

Notes:
  - "motivation" reads analysis.jsonl and drops the "notes" field.
  - "insight" reads metadata.jsonl and computes:
      1) misspeculation top-k inclusiveness (top1..top10) for target_accept_len == 0 rows,
      2) misspeculation confidence correlation for target_accept_len == 0 rows,
      3) confidence_distribution for all metadata rows,
      4) confidence_misspeuclation_position_correlation (top2..top5, top10, others for target_accept_len == 0 rows, plus raw_distribution["top1"] for all target rows with residual confidence < 0.8)
      5) oracle_acceptance_length from metadata.jsonl plus analysis.jsonl,
      6) markov_chain over consecutive target_accept_len values within each request_id.
  - "trace" reads metadata.jsonl and stores target_accept_len for request IDs
    4..14 and step IDs 1..20 by default. Missing steps are stored as None.
  - "results" reads cost_breakdown.json and drops the "notes" field.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Align with ``ssd.utils.profiler_metadata.FIRST_DRAFT_METADATA_TOPK`` (first-position draft top-k in metadata rows).
FIRST_DRAFT_METADATA_TOPK = 10

# -----------------------------
# Basic parsing helpers
# -----------------------------


def _to_number_if_clean(value: str) -> Any:
    """Convert numeric-looking strings to int or float. Otherwise return original string."""
    if value == "":
        return value

    # Support compact decimal tags used in paths: 0p3 -> 0.3, 1p0 -> 1.0
    compact_decimal = re.fullmatch(r"-?\d+p\d+", value)
    if compact_decimal:
        value = value.replace("p", ".", 1)

    try:
        as_float = float(value)
    except ValueError:
        return value

    if math.isfinite(as_float) and as_float.is_integer():
        return int(as_float)
    return as_float


def _parse_prefixed_int(tag: str, prefix: str, field_name: str) -> int:
    if not tag.startswith(prefix):
        raise ValueError(f"Expected {field_name} tag to start with {prefix!r}, got {tag!r}")

    raw = tag[len(prefix) :]
    if raw == "":
        raise ValueError(f"Missing numeric value in {field_name} tag: {tag!r}")

    return int(raw)


def _parse_k_tag(tag: str) -> int | None:
    """Parse speculative-length path tag: k<int> or kna."""
    if tag == "kna":
        return None
    return _parse_prefixed_int(tag, "k", "speculative_length")


def _parse_prefixed_number(tag: str, prefix: str, field_name: str) -> Any:
    if not tag.startswith(prefix):
        raise ValueError(f"Expected {field_name} tag to start with {prefix!r}, got {tag!r}")

    return _to_number_if_clean(tag[len(prefix) :])


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file. Blank lines are ignored."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_no}, "
                    f"got {type(obj).__name__}"
                )

            yield obj


def read_json_object(path: Path) -> Dict[str, Any]:
    """Read one JSON object from a .json file."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")

    return obj


def strip_notes(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy without the notes field."""
    return {k: v for k, v in obj.items() if k != "notes"}


def read_first_jsonl_object(path: Path) -> Optional[Dict[str, Any]]:
    """Read the first JSON object from a JSONL file. Return None when empty or missing."""
    if not path.exists():
        return None

    for obj in iter_jsonl(path):
        return obj

    return None


def _as_int(value: Any) -> Optional[int]:
    """Best-effort conversion to int. Return None for missing or invalid values."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    """Best-effort conversion to float. Return None for missing or invalid values."""
    if value is None:
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    return out if math.isfinite(out) else None


def _mean(values: List[int]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


# -----------------------------
# Run discovery and context
# -----------------------------


def find_metadata_run_dirs(metadata_root: Path) -> List[Path]:
    """Return directories containing analysis.jsonl or metadata.jsonl."""
    run_dirs = set()

    for name in ("analysis.jsonl", "metadata.jsonl"):
        for path in metadata_root.rglob(name):
            if path.is_file():
                run_dirs.add(path.parent)

    return sorted(run_dirs)


def find_cost_run_dirs(cost_root: Path) -> List[Path]:
    """Return directories containing cost_breakdown.json."""
    return sorted({path.parent for path in cost_root.rglob("cost_breakdown.json") if path.is_file()})


def parse_context(run_dir: Path, keyed_root: Path) -> Dict[str, Any]:
    """
    Parse context from a run directory.

    Relative layout:
      method / b16 / k3 / target+draft / t0 / dataset
    """
    try:
        rel = run_dir.relative_to(keyed_root)
        parts = rel.parts
    except ValueError:
        parts = run_dir.parts

    pivot_expansion_policy: str | None = None
    pivot_round: Any | None = None
    pivot_topk: int | None = None
    pivot_expansion_pct: Any | None = None

    if len(parts) >= 10 and parts[0] == "pivot" and parts[1] in {
        "dynamic",
        "static",
        "dynamic_expansion",
    }:
        # New pivot layout:
        # pivot/<policy>/b*/k*/pair/t*/r_*/topk*/pct*/dataset
        method = "pivot"
        pivot_expansion_policy = parts[1]
        batch_size = _parse_prefixed_int(parts[2], "b", "batch size")
        speculative_length = _parse_k_tag(parts[3])
        model_pair = parts[4]
        temperature = _parse_prefixed_number(parts[5], "t", "temperature")
        pivot_round = _parse_prefixed_number(parts[6], "r_", "pivot round")
        pivot_topk = _parse_prefixed_int(parts[7], "topk", "pivot_topk")
        pivot_expansion_pct = _parse_prefixed_number(parts[8], "pct", "pivot_expansion_pct")
        dataset = parts[9]
    elif len(parts) >= 9 and parts[0] == "pivot":
        # Legacy pivot layout:
        # pivot/b*/k*/pair/t*/r_*/topk*/pct*/dataset
        method = "pivot"
        batch_size = _parse_prefixed_int(parts[1], "b", "batch size")
        speculative_length = _parse_k_tag(parts[2])
        model_pair = parts[3]
        temperature = _parse_prefixed_number(parts[4], "t", "temperature")
        pivot_round = _parse_prefixed_number(parts[5], "r_", "pivot round")
        pivot_topk = _parse_prefixed_int(parts[6], "topk", "pivot_topk")
        pivot_expansion_pct = _parse_prefixed_number(parts[7], "pct", "pivot_expansion_pct")
        dataset = parts[8]
    else:
        if len(parts) < 6:
            raise ValueError(
                f"Cannot parse run context from {run_dir}. Expected at least 6 path components "
                f"under keyed root: method/b*/k*/target+draft/t*/dataset"
            )
        # Legacy/default layout:
        # method/b*/k*/pair/t*/dataset
        method = parts[0]
        batch_size = _parse_prefixed_int(parts[1], "b", "batch size")
        speculative_length = _parse_k_tag(parts[2])
        model_pair = parts[3]
        temperature = _parse_prefixed_number(parts[4], "t", "temperature")
        dataset = parts[5]

    model_components = model_pair.split("+")
    target_model = model_components[0] if model_components else None
    draft_model = model_components[-1] if len(model_components) >= 2 else None

    ctx: Dict[str, Any] = {
        "method": method,
        "batch_size": batch_size,
        "speculative_length": speculative_length,
        "target_model": target_model,
        "draft_model": draft_model,
        "temperature": temperature,
        "dataset": dataset,
        "run_dir": str(run_dir),
    }

    if len(model_components) > 2:
        ctx["intermediate_models"] = model_components[1:-1]
    if pivot_expansion_policy is not None:
        ctx["pivot_expansion_policy"] = pivot_expansion_policy
    if pivot_round is not None:
        ctx["pivot_round"] = pivot_round
    if pivot_topk is not None:
        ctx["pivot_topk"] = pivot_topk
    if pivot_expansion_pct is not None:
        ctx["pivot_expansion_pct"] = pivot_expansion_pct

    return ctx


def safe_context(run_dir: Path, keyed_root: Path) -> Optional[Dict[str, Any]]:
    try:
        return parse_context(run_dir, keyed_root)
    except Exception as exc:
        print(f"[warn] Skipping {run_dir}: {exc}")
        return None


# -----------------------------
# motivation: analysis.jsonl
# -----------------------------


def build_motivation_entries(analysis_path: Path, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for obj in iter_jsonl(analysis_path):
        entry = dict(ctx)
        entry.update(strip_notes(obj))
        entries.append(entry)

    return entries


# -----------------------------
# insight: metadata.jsonl
# -----------------------------


TARGET_VERIFY_MODELS = ("target", "pivot_target")


def is_target_verify_row(row: Dict[str, Any]) -> bool:
    """Return True for target-side verification rows only."""
    return row.get("verification_model") in TARGET_VERIFY_MODELS


def is_target_misspeculation_row(row: Dict[str, Any]) -> bool:
    """A misspeculation row is a *target* verify slot with target_accept_len == 0."""
    if not is_target_verify_row(row):
        return False
    return row.get("target_accept_len") == 0


def _get_row_speculative_length(
    row: Dict[str, Any],
    ctx: Dict[str, Any],
    analysis_obj: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Resolve num_speculative_token for a metadata row, then fall back to run context."""
    candidates = [
        row.get("num_speculative_token"),
        ctx.get("speculative_length"),
    ]
    if analysis_obj is not None:
        candidates.append(analysis_obj.get("speculate_k"))

    for candidate in candidates:
        out = _as_int(candidate)
        if out is not None:
            return out

    return None


def _target_accept_len_rows(
    rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    analysis_obj: Optional[Dict[str, Any]],
) -> List[Tuple[int, Optional[int]]]:
    """Return target accept lengths with their per-row speculative length in file order."""
    out: List[Tuple[int, Optional[int]]] = []

    for row in rows:
        if not is_target_verify_row(row):
            continue

        accept_len = _as_int(row.get("target_accept_len"))
        if accept_len is None:
            continue

        out.append((accept_len, _get_row_speculative_length(row, ctx, analysis_obj)))

    return out


def compute_oracle_acceptance_length(
    rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    analysis_obj: Optional[Dict[str, Any]],
    mean_check_atol: float = 1e-6,
    mean_check_rtol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Build the oracle acceptance-length summary requested for the insight key.

    Procedure:
      1. Read avg_target_accept_len, avg_target_accept_len_incl_recovery, and
         total_target_verification_rounds from analysis.jsonl.
      2. Read target_accept_len from all target verification rows in metadata.jsonl.
      3. Check that the metadata mean equals avg_target_accept_len.
      4. Replace every 0 accept length with num_speculative_token.
      5. Accumulate replaced_accept_len + 1 until reaching generated-token count.
      6. Truncate at that point and report the oracle average.
    """
    target_pairs = _target_accept_len_rows(rows, ctx, analysis_obj)
    target_accept_len_list = [accept_len for accept_len, _ in target_pairs]

    true_avg_accept_len = None
    avg_target_accept_len_incl_recovery = None
    total_target_verification_rounds = None
    num_generated_tokens = None

    if analysis_obj is not None:
        true_avg_accept_len = _as_float(analysis_obj.get("avg_target_accept_len"))
        avg_target_accept_len_incl_recovery = _as_float(
            analysis_obj.get("avg_target_accept_len_incl_recovery")
        )
        total_target_verification_rounds = _as_int(
            analysis_obj.get("total_target_verification_rounds")
        )

        if (
            avg_target_accept_len_incl_recovery is not None
            and total_target_verification_rounds is not None
        ):
            num_generated_tokens = (
                avg_target_accept_len_incl_recovery * total_target_verification_rounds
            )

    metadata_avg_target_accept_len = _mean(target_accept_len_list)
    mean_abs_error = None
    mean_matches = None

    if true_avg_accept_len is not None and metadata_avg_target_accept_len is not None:
        mean_abs_error = abs(metadata_avg_target_accept_len - true_avg_accept_len)
        mean_matches = math.isclose(
            metadata_avg_target_accept_len,
            true_avg_accept_len,
            rel_tol=mean_check_rtol,
            abs_tol=mean_check_atol,
        )

    default_k = None
    k_candidates: List[int] = []
    for _, row_k in target_pairs:
        if row_k is not None:
            k_candidates.append(row_k)
    if k_candidates:
        default_k = max(k_candidates)
    else:
        default_k = _get_row_speculative_length({}, ctx, analysis_obj)

    zero_replacement_missing_k_count = 0
    oracle_target_accept_len_list: List[int] = []
    for accept_len, row_k in target_pairs:
        if accept_len == 0:
            replacement = row_k if row_k is not None else default_k
            if replacement is None:
                zero_replacement_missing_k_count += 1
                replacement = accept_len
            oracle_target_accept_len_list.append(replacement)
        else:
            oracle_target_accept_len_list.append(accept_len)

    sum_accepted_tokens = 0.0
    truncation_index = None

    if num_generated_tokens is not None:
        for idx, accept_len in enumerate(oracle_target_accept_len_list):
            sum_accepted_tokens += accept_len + 1
            if sum_accepted_tokens >= num_generated_tokens:
                truncation_index = idx
                break

    if truncation_index is None:
        truncated_oracle_target_accept_len_list = list(oracle_target_accept_len_list)
        if num_generated_tokens is None:
            sum_accepted_tokens_until_truncation = None
        else:
            sum_accepted_tokens_until_truncation = sum(
                accept_len + 1 for accept_len in truncated_oracle_target_accept_len_list
            )
    else:
        truncated_oracle_target_accept_len_list = oracle_target_accept_len_list[
            : truncation_index + 1
        ]
        sum_accepted_tokens_until_truncation = sum_accepted_tokens

    oracle_avg_acceptance_length = _mean(truncated_oracle_target_accept_len_list)
    oracle_avg_committed_length = (
        oracle_avg_acceptance_length + 1
        if oracle_avg_acceptance_length is not None
        else None
    )

    return {
        "analysis_jsonl_available": analysis_obj is not None,
        "true_avg_accept_len": true_avg_accept_len,
        "avg_target_accept_len_incl_recovery": avg_target_accept_len_incl_recovery,
        "total_target_verification_rounds": total_target_verification_rounds,
        "num_generated_tokens": num_generated_tokens,
        "num_target_accept_len_rows": len(target_accept_len_list),
        "target_accept_len_list": target_accept_len_list,
        "metadata_avg_target_accept_len": metadata_avg_target_accept_len,
        "metadata_avg_matches_true_avg_accept_len": mean_matches,
        "metadata_avg_abs_error": mean_abs_error,
        "zero_replacement_value_default": default_k,
        "zero_replacement_missing_k_count": zero_replacement_missing_k_count,
        "oracle_target_accept_len_list": oracle_target_accept_len_list,
        "truncation_index": truncation_index,
        "truncated_rounds": len(truncated_oracle_target_accept_len_list),
        "truncated_oracle_target_accept_len_list": truncated_oracle_target_accept_len_list,
        "sum_accepted_tokens_until_truncation": sum_accepted_tokens_until_truncation,
        "oracle_reached_num_generated_tokens": (
            bool(
                num_generated_tokens is not None
                and sum_accepted_tokens_until_truncation is not None
                and sum_accepted_tokens_until_truncation >= num_generated_tokens
            )
        ),
        "oracle_avg_acceptance_length": oracle_avg_acceptance_length,
        "oracle_avg_committed_length": oracle_avg_committed_length,
    }


def compute_markov_chain(
    rows: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    analysis_obj: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute P(next target_accept_len | current target_accept_len) per request.

    metadata.jsonl is usually ordered by decode step, then by request. Adjacent
    file rows can therefore correspond to different requests. A temporal Markov
    transition must instead be computed within each request_id sequence:

      request r: step t accept_len -> step t+1 accept_len

    The global transition matrix below pools those per-request transitions.
    """
    target_accept_lens_file_order: List[int] = []
    row_k_values: List[int] = []

    # request_id -> list of (sort_key, accept_len)
    events_by_request: Dict[int, List[Tuple[Tuple[int, int, int, int], int]]] = {}

    skipped_missing_accept_len_rows = 0
    skipped_missing_request_id_rows = 0

    for file_idx, row in enumerate(rows):
        if not is_target_verify_row(row):
            continue

        accept_len = _as_int(row.get("target_accept_len"))
        if accept_len is None:
            skipped_missing_accept_len_rows += 1
            continue

        target_accept_lens_file_order.append(accept_len)

        row_k = _get_row_speculative_length(row, ctx, analysis_obj)
        if row_k is not None:
            row_k_values.append(row_k)

        request_id = _as_int(row.get("request_id"))
        if request_id is None:
            skipped_missing_request_id_rows += 1
            continue

        step_id = _as_int(row.get("step_id"))
        target_round = _as_int(row.get("target_verification_round"))
        intermediate_round = _as_int(row.get("intermediate_verification_round"))

        sort_step = step_id if step_id is not None else file_idx
        sort_target_round = target_round if target_round is not None else 0
        sort_intermediate_round = intermediate_round if intermediate_round is not None else 0
        sort_key = (sort_step, sort_target_round, sort_intermediate_round, file_idx)

        events_by_request.setdefault(request_id, []).append((sort_key, accept_len))

    max_state_candidates = [0]
    max_state_candidates.extend(target_accept_lens_file_order)
    max_state_candidates.extend(row_k_values)

    ctx_k = _get_row_speculative_length({}, ctx, analysis_obj)
    if ctx_k is not None:
        max_state_candidates.append(ctx_k)

    max_state = max(max_state_candidates)
    states = list(range(max_state + 1))

    transition_counts: Dict[int, List[int]] = {
        state: [0 for _ in states] for state in states
    }
    outgoing_counts: Dict[int, int] = {state: 0 for state in states}

    target_accept_len_by_request: Dict[str, List[int]] = {}
    request_sequence_lengths: Dict[str, int] = {}
    skipped_transitions = 0
    num_transitions = 0
    num_requests_with_transitions = 0

    for request_id, events in sorted(events_by_request.items()):
        events.sort(key=lambda item: item[0])
        seq = [accept_len for _, accept_len in events]

        target_accept_len_by_request[str(request_id)] = seq
        request_sequence_lengths[str(request_id)] = len(seq)

        if len(seq) < 2:
            continue

        num_requests_with_transitions += 1

        for cur, nxt in zip(seq, seq[1:]):
            if cur < 0 or nxt < 0 or cur > max_state or nxt > max_state:
                skipped_transitions += 1
                continue

            transition_counts[cur][nxt] += 1
            outgoing_counts[cur] += 1
            num_transitions += 1

    transition_probs: Dict[str, List[float]] = {}
    transition_counts_out: Dict[str, List[int]] = {}
    outgoing_counts_out: Dict[str, int] = {}

    for state in states:
        denom = outgoing_counts[state]
        counts = transition_counts[state]
        if denom > 0:
            probs = [count / denom for count in counts]
        else:
            probs = [0.0 for _ in counts]

        transition_probs[str(state)] = probs
        transition_counts_out[str(state)] = counts
        outgoing_counts_out[str(state)] = denom

    file_order_transition_count = max(0, len(target_accept_lens_file_order) - 1)

    return {
        "transition_scope": "per_request",
        "transition_order": "sort by (step_id, target_verification_round, intermediate_verification_round, file_idx) within each request_id",
        "states": states,
        "target_accept_len_list": target_accept_lens_file_order,
        "target_accept_len_by_request": target_accept_len_by_request,
        "request_sequence_lengths": request_sequence_lengths,
        "num_target_accept_len_rows": len(target_accept_lens_file_order),
        "num_request_sequences": len(events_by_request),
        "num_requests_with_transitions": num_requests_with_transitions,
        "num_transitions": num_transitions,
        "file_order_transition_count_not_used": file_order_transition_count,
        "cross_request_file_order_transitions_excluded": file_order_transition_count - num_transitions,
        "skipped_missing_accept_len_rows": skipped_missing_accept_len_rows,
        "skipped_missing_request_id_rows": skipped_missing_request_id_rows,
        "skipped_transitions": skipped_transitions,
        "transition_probabilities": transition_probs,
        "transition_counts": transition_counts_out,
        "outgoing_counts": outgoing_counts_out,
    }


def compute_topk_inclusiveness(
    miss_rows: List[Dict[str, Any]],
    max_topk: int = FIRST_DRAFT_METADATA_TOPK,
) -> Dict[str, Any]:
    counts = {f"top{k}": 0 for k in range(1, max_topk + 1)}
    n = len(miss_rows)

    for row in miss_rows:
        recovery = row.get("target_recovery_token")
        first_ids = row.get("first_draft_token_ids") or []

        if recovery is None or not isinstance(first_ids, list):
            continue

        for k in range(1, max_topk + 1):
            if recovery in first_ids[:k]:
                counts[f"top{k}"] += 1

    out: Dict[str, Any] = {
        "num_misspeculation_steps": n,
        **counts,
    }

    for k in range(1, max_topk + 1):
        out[f"top{k}_prob"] = (counts[f"top{k}"] / n) if n > 0 else None

    return out


def extract_first_draft_confidence_pair(row: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Extract confidence values from one metadata row.

    Returns:
      top1_confidence = first_draft_token_confidence[0]
      residual_confidence = first_draft_token_confidence[0] - first_draft_token_confidence[1]

    Missing or malformed values are returned as None.
    """
    confs = row.get("first_draft_token_confidence") or []

    if not isinstance(confs, list) or len(confs) < 1:
        return None, None

    top1_confidence = confs[0]

    if len(confs) >= 2 and confs[0] is not None and confs[1] is not None:
        residual_confidence = confs[0] - confs[1]
    else:
        residual_confidence = None

    return top1_confidence, residual_confidence


def compute_confidence_correlation_rows(miss_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Confidence rows over only misspeculation rows, target_accept_len == 0."""
    rows: List[Dict[str, Any]] = []

    for row in miss_rows:
        top1_confidence, residual_confidence = extract_first_draft_confidence_pair(row)

        rows.append(
            {
                "step_id": row.get("step_id"),
                "request_id": row.get("request_id"),
                "top1_confidence": top1_confidence,
                "residual_confidence": residual_confidence,
                "target_recovery_token": row.get("target_recovery_token"),
                "first_draft_token_ids": row.get("first_draft_token_ids"),
                "first_draft_token_confidence": row.get("first_draft_token_confidence"),
            }
        )

    return rows


def compute_confidence_distribution_rows(all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Confidence rows over all metadata rows.

    For each row:
      top1_confidence = first_draft_token_confidence[0]
      residual_confidence = first_draft_token_confidence[0] - first_draft_token_confidence[1]
    """
    rows: List[Dict[str, Any]] = []

    for row in all_rows:
        top1_confidence, residual_confidence = extract_first_draft_confidence_pair(row)

        rows.append(
            {
                "step_id": row.get("step_id"),
                "request_id": row.get("request_id"),
                "verification_model": row.get("verification_model"),
                "target_accept_len": row.get("target_accept_len"),
                "inter_accept_len": row.get("inter_accept_len"),
                "top1_confidence": top1_confidence,
                "residual_confidence": residual_confidence,
                "first_draft_token_ids": row.get("first_draft_token_ids"),
                "first_draft_token_confidence": row.get("first_draft_token_confidence"),
            }
        )

    return rows


MISSPEC_POSITION_CATEGORIES = ("top2", "top3", "top4", "top5", "top10", "others")
LOW_RESIDUAL_TOP1_THRESHOLD = 0.8


def _empty_category_lists() -> Dict[str, List[Any]]:
    return {category: [] for category in MISSPEC_POSITION_CATEGORIES}


def _target_recovery_rank_category(
    row: Dict[str, Any],
    max_topk: int = FIRST_DRAFT_METADATA_TOPK,
) -> Tuple[str, Optional[int]]:
    """
    Classify the misspeculated target token by its rank in ``first_draft_token_ids``.

    For target_accept_len == 0 rows, compare ``target_recovery_token`` to the draft
    top-``max_topk`` at position 0. Categories: top2..top5 (ranks 2--5), ``top10`` for
    ranks 6--10 (indices 5--9), and ``others`` (including top1 matches and misses).
    """
    recovery = row.get("target_recovery_token")
    first_ids = row.get("first_draft_token_ids") or []

    if recovery is None or not isinstance(first_ids, list):
        return "others", None

    try:
        rank_idx = first_ids[:max_topk].index(recovery)
    except ValueError:
        return "others", None

    if rank_idx == 0:
        return "others", rank_idx
    if rank_idx == 1:
        return "top2", rank_idx
    if rank_idx == 2:
        return "top3", rank_idx
    if rank_idx == 3:
        return "top4", rank_idx
    if rank_idx == 4:
        return "top5", rank_idx
    if 5 <= rank_idx <= 9:
        return "top10", rank_idx
    return "others", rank_idx


def _ensure_len(values: List[Any], target_len: int, fill_value: Any = 0.0) -> None:
    while len(values) < target_len:
        values.append(fill_value)


def _average_or_none(total: float, count: int) -> Optional[float]:
    return total / count if count > 0 else None


def _accumulate_first_draft_confidence(
    row: Dict[str, Any],
    sums: List[float],
    counts: List[int],
) -> bool:
    """
    Accumulate one row's ``first_draft_token_confidence`` vector.

    Returns True if at least one finite value was accumulated.
    """
    first_conf = row.get("first_draft_token_confidence") or []
    if not isinstance(first_conf, list):
        return False

    _ensure_len(sums, len(first_conf), 0.0)
    _ensure_len(counts, len(first_conf), 0)

    has_value = False
    for pos, value in enumerate(first_conf):
        value_float = _as_float(value)
        if value_float is None:
            continue
        sums[pos] += value_float
        counts[pos] += 1
        has_value = True

    return has_value


def _build_raw_distribution_entry(
    count: int,
    sums: List[float],
    counts: List[int],
) -> Dict[str, Any]:
    avg = [
        _average_or_none(total, count_at_pos)
        for total, count_at_pos in zip(sums, counts)
    ]

    return {
        "count": count,
        "avg_first_draft_token_confidence": avg,
        "sum_first_draft_token_confidence": sums,
        "num_values_per_position": counts,
    }


def compute_confidence_misspeuclation_position_correlation(
    miss_rows: List[Dict[str, Any]],
    all_target_rows: List[Dict[str, Any]],
    max_topk: int = FIRST_DRAFT_METADATA_TOPK,
    low_residual_top1_threshold: float = LOW_RESIDUAL_TOP1_THRESHOLD,
) -> Dict[str, Any]:
    """
    Correlate misspeculation confidence with the target-token draft rank bucket.

    Misspeculation rank buckets:
      - top2, top3, top4, top5, top10, others
      - computed only over target misspeculation rows where target_accept_len == 0

    Added raw_distribution['top1'] bucket:
      - computed over all target-side verification rows, not only misspeculation rows
      - includes rows whose residual confidence is below ``low_residual_top1_threshold``
      - residual confidence is first_draft_token_confidence[0] - first_draft_token_confidence[1]

    Output schema:
      selection_criteria:
        top1_confidence[category] contains first_draft_token_confidence[0]
        residual_confidence[category] contains
          first_draft_token_confidence[0] - first_draft_token_confidence[1]

      raw_distribution[category] contains:
        count
        avg_first_draft_token_confidence
        sum_first_draft_token_confidence
        num_values_per_position

    Important:
      raw_distribution['top1'] has a different meaning from the misspeculation
      rank buckets. It is a low-residual-confidence bucket over all target rows.
    """
    selection_criteria: Dict[str, Dict[str, List[Any]]] = {
        "top1_confidence": _empty_category_lists(),
        "residual_confidence": _empty_category_lists(),
    }

    category_counts: Dict[str, int] = {
        category: 0 for category in MISSPEC_POSITION_CATEGORIES
    }
    unexpected_top1_match_count = 0
    missing_target_or_topk_count = 0

    raw_sums: Dict[str, List[float]] = {
        category: [] for category in MISSPEC_POSITION_CATEGORIES
    }
    raw_value_counts: Dict[str, List[int]] = {
        category: [] for category in MISSPEC_POSITION_CATEGORIES
    }

    rows_by_category: Dict[str, List[Dict[str, Any]]] = {
        category: [] for category in MISSPEC_POSITION_CATEGORIES
    }

    for row in miss_rows:
        category, rank_idx = _target_recovery_rank_category(row, max_topk=max_topk)
        category_counts[category] += 1

        if rank_idx == 0:
            unexpected_top1_match_count += 1
        if rank_idx is None:
            first_ids = row.get("first_draft_token_ids") or []
            if row.get("target_recovery_token") is None or not isinstance(first_ids, list):
                missing_target_or_topk_count += 1

        top1_confidence, residual_confidence = extract_first_draft_confidence_pair(row)
        selection_criteria["top1_confidence"][category].append(_as_float(top1_confidence))
        selection_criteria["residual_confidence"][category].append(_as_float(residual_confidence))

        _accumulate_first_draft_confidence(
            row=row,
            sums=raw_sums[category],
            counts=raw_value_counts[category],
        )

        rows_by_category[category].append(
            {
                "step_id": row.get("step_id"),
                "request_id": row.get("request_id"),
                "target_recovery_token": row.get("target_recovery_token"),
                "target_rank_category": category,
                "target_rank_index": rank_idx,
                "first_draft_token_ids": row.get("first_draft_token_ids"),
                "first_draft_token_confidence": row.get("first_draft_token_confidence"),
            }
        )

    raw_distribution: Dict[str, Dict[str, Any]] = {}

    # New top1 bucket: all target verification rows with low residual confidence.
    top1_low_residual_sums: List[float] = []
    top1_low_residual_counts: List[int] = []
    top1_low_residual_row_count = 0
    top1_low_residual_missing_confidence_count = 0
    top1_low_residual_rows: List[Dict[str, Any]] = []

    for row in all_target_rows:
        top1_confidence, residual_confidence = extract_first_draft_confidence_pair(row)
        residual_confidence_float = _as_float(residual_confidence)

        if residual_confidence_float is None:
            top1_low_residual_missing_confidence_count += 1
            continue

        if residual_confidence_float >= low_residual_top1_threshold:
            continue

        accumulated = _accumulate_first_draft_confidence(
            row=row,
            sums=top1_low_residual_sums,
            counts=top1_low_residual_counts,
        )
        if not accumulated:
            top1_low_residual_missing_confidence_count += 1
            continue

        top1_low_residual_row_count += 1
        top1_low_residual_rows.append(
            {
                "step_id": row.get("step_id"),
                "request_id": row.get("request_id"),
                "verification_model": row.get("verification_model"),
                "target_accept_len": row.get("target_accept_len"),
                "target_recovery_token": row.get("target_recovery_token"),
                "top1_confidence": _as_float(top1_confidence),
                "residual_confidence": residual_confidence_float,
                "first_draft_token_ids": row.get("first_draft_token_ids"),
                "first_draft_token_confidence": row.get("first_draft_token_confidence"),
            }
        )

    raw_distribution["top1"] = _build_raw_distribution_entry(
        count=top1_low_residual_row_count,
        sums=top1_low_residual_sums,
        counts=top1_low_residual_counts,
    )
    raw_distribution["top1"]["source"] = "all_target_rows_with_low_residual_confidence"
    raw_distribution["top1"]["residual_confidence_threshold"] = low_residual_top1_threshold
    raw_distribution["top1"]["selection_rule"] = (
        "first_draft_token_confidence[0] - first_draft_token_confidence[1] "
        f"< {low_residual_top1_threshold}"
    )

    for category in MISSPEC_POSITION_CATEGORIES:
        raw_distribution[category] = _build_raw_distribution_entry(
            count=category_counts[category],
            sums=raw_sums[category],
            counts=raw_value_counts[category],
        )
        raw_distribution[category]["source"] = "misspeculation_rows_by_target_recovery_rank"

    selection_criteria_list_lengths = {
        criterion: {
            category: len(values)
            for category, values in category_values.items()
        }
        for criterion, category_values in selection_criteria.items()
    }

    selection_criteria_total_list_lengths = {
        criterion: sum(lengths.values())
        for criterion, lengths in selection_criteria_list_lengths.items()
    }

    num_all_target_rows = len(all_target_rows)
    top1_low_residual_probability_among_target_rows = (
        top1_low_residual_row_count / num_all_target_rows
        if num_all_target_rows > 0
        else None
    )

    return {
        "category_definition": (
            "top2, top3, top4, top5, top10, and others use target_recovery_token "
            "rank in first_draft_token_ids for target_accept_len == 0 rows. "
            "raw_distribution['top1'] uses all target rows with residual confidence "
            f"less than {low_residual_top1_threshold}."
        ),
        "categories": ["top1", *list(MISSPEC_POSITION_CATEGORIES)],
        "misspeculation_categories": list(MISSPEC_POSITION_CATEGORIES),
        "num_misspeculation_steps": len(miss_rows),
        "num_all_target_rows": num_all_target_rows,
        "low_residual_top1_threshold": low_residual_top1_threshold,
        "top1_low_residual_count": top1_low_residual_row_count,
        "top1_low_residual_probability_among_target_rows": (
            top1_low_residual_probability_among_target_rows
        ),
        "top1_low_residual_missing_confidence_count": (
            top1_low_residual_missing_confidence_count
        ),
        "category_counts": category_counts,
        "category_probabilities": {
            category: (count / len(miss_rows) if miss_rows else None)
            for category, count in category_counts.items()
        },
        "selection_criteria": selection_criteria,
        "selection_criteria_list_lengths": selection_criteria_list_lengths,
        "selection_criteria_total_list_lengths": selection_criteria_total_list_lengths,
        "raw_distribution": raw_distribution,
        "rows_by_category": rows_by_category,
        "top1_low_residual_rows": top1_low_residual_rows,
        "unexpected_top1_match_count": unexpected_top1_match_count,
        "missing_target_or_topk_count": missing_target_or_topk_count,
    }


def build_insight_entry(
    metadata_path: Path,
    ctx: Dict[str, Any],
    analysis_path: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = list(iter_jsonl(metadata_path))
    target_rows = [row for row in rows if is_target_verify_row(row)]
    miss_rows = [row for row in rows if is_target_misspeculation_row(row)]
    analysis_obj = read_first_jsonl_object(analysis_path) if analysis_path is not None else None

    entry = dict(ctx)
    entry["num_metadata_rows"] = len(rows)
    entry["num_target_rows"] = len(target_rows)
    entry["num_misspeculation_steps"] = len(miss_rows)
    entry["misspeculation_topk_inclusiveness"] = compute_topk_inclusiveness(
        miss_rows, max_topk=FIRST_DRAFT_METADATA_TOPK
    )
    entry["misspeculation_confidence_correlation"] = compute_confidence_correlation_rows(miss_rows)
    entry["confidence_distribution"] = compute_confidence_distribution_rows(rows)
    entry["confidence_misspeuclation_position_correlation"] = (
        compute_confidence_misspeuclation_position_correlation(
            miss_rows=miss_rows,
            all_target_rows=target_rows,
            max_topk=FIRST_DRAFT_METADATA_TOPK,
            low_residual_top1_threshold=LOW_RESIDUAL_TOP1_THRESHOLD,
        )
    )
    entry["oracle_acceptance_length"] = compute_oracle_acceptance_length(rows, ctx, analysis_obj)
    entry["markov_chain"] = compute_markov_chain(rows, ctx, analysis_obj)

    return entry


# -----------------------------
# trace: metadata.jsonl
# -----------------------------


def build_trace_entry(
    metadata_path: Path,
    ctx: Dict[str, Any],
    request_start: int,
    request_end: int,
    step_start: int,
    step_end: int,
) -> Dict[str, Any]:
    request_ids = list(range(request_start, request_end + 1))
    step_ids = list(range(step_start, step_end + 1))

    trace: Dict[int, Dict[int, Any]] = {
        request_id: {step_id: None for step_id in step_ids}
        for request_id in request_ids
    }

    duplicate_rows = 0
    seen: set[Tuple[int, int]] = set()

    for row in iter_jsonl(metadata_path):
        if not is_target_verify_row(row):
            continue
        req = row.get("request_id")
        step = row.get("step_id")
        accept_len = row.get("target_accept_len")

        if req not in trace:
            continue
        if step not in trace[req]:
            continue
        if accept_len is None:
            continue

        key = (int(req), int(step))
        if key in seen:
            duplicate_rows += 1
            continue

        trace[int(req)][int(step)] = accept_len
        seen.add(key)

    target_accept_len_by_request = {
        f"request_{request_id}": [trace[request_id][step_id] for step_id in step_ids]
        for request_id in request_ids
    }

    entry = dict(ctx)
    entry["request_start"] = request_start
    entry["request_end"] = request_end
    entry["step_start"] = step_start
    entry["step_end"] = step_end
    entry["target_accept_len_by_request"] = target_accept_len_by_request
    entry["duplicate_target_accept_len_rows_ignored"] = duplicate_rows

    return entry


# -----------------------------
# results: cost_breakdown.json
# -----------------------------


def build_result_entry(cost_breakdown_path: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    cost = strip_notes(read_json_object(cost_breakdown_path))

    entry = dict(ctx)
    entry.update(cost)

    return entry


# -----------------------------
# Key-specific collectors
# -----------------------------


def collect_from_metadata_root(
    metadata_root: Path,
    request_start: int,
    request_end: int,
    step_start: int,
    step_end: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect keys backed by <input_root>/metadata."""
    collected: Dict[str, List[Dict[str, Any]]] = {
        "motivation": [],
        "insight": [],
        "trace": [],
    }

    if not metadata_root.exists():
        print(f"[warn] metadata root does not exist, skipping metadata-backed keys: {metadata_root}")
        return collected

    run_dirs = find_metadata_run_dirs(metadata_root)

    for run_dir in run_dirs:
        ctx = safe_context(run_dir, metadata_root)
        if ctx is None:
            continue

        analysis_path = run_dir / "analysis.jsonl"
        metadata_path = run_dir / "metadata.jsonl"

        if analysis_path.exists():
            collected["motivation"].extend(build_motivation_entries(analysis_path, ctx))

        if metadata_path.exists():
            collected["insight"].append(build_insight_entry(metadata_path, ctx, analysis_path))
            collected["trace"].append(
                build_trace_entry(
                    metadata_path,
                    ctx,
                    request_start=request_start,
                    request_end=request_end,
                    step_start=step_start,
                    step_end=step_end,
                )
            )

    return collected


def collect_from_cost_root(cost_root: Path) -> List[Dict[str, Any]]:
    """Collect the results key backed by <input_root>/cost."""
    results: List[Dict[str, Any]] = []

    if not cost_root.exists():
        print(f"[warn] cost root does not exist, skipping results key: {cost_root}")
        return results

    run_dirs = find_cost_run_dirs(cost_root)

    for run_dir in run_dirs:
        ctx = safe_context(run_dir, cost_root)
        if ctx is None:
            continue

        cost_breakdown_path = run_dir / "cost_breakdown.json"
        if cost_breakdown_path.exists():
            results.append(build_result_entry(cost_breakdown_path, ctx))

    return results


def collect(input_root: Path, request_start: int, request_end: int, step_start: int, step_end: int) -> Dict[str, Any]:
    """
    Collect all output keys from the global results root.

    Key routing:
      motivation, insight, trace -> <input_root>/metadata
      results                    -> <input_root>/cost
    """
    metadata_root = input_root / "metadata"
    cost_root = input_root / "cost"

    metadata_collected = collect_from_metadata_root(
        metadata_root=metadata_root,
        request_start=request_start,
        request_end=request_end,
        step_start=step_start,
        step_end=step_end,
    )

    return {
        "motivation": metadata_collected["motivation"],
        "insight": metadata_collected["insight"],
        "trace": metadata_collected["trace"],
        "results": collect_from_cost_root(cost_root),
    }


# -----------------------------
# Output
# -----------------------------


def write_output(data: Dict[str, Any], output_path: Path, pretty_json: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Writing .pt requires torch to be installed.") from exc

        torch.save(data, output_path)
        return

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2 if pretty_json else None)
            f.write("\n")
        return

    if suffix == ".jsonl":
        # Single JSONL record whose outer keys are motivation, insight, trace, results.
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        return

    raise ValueError(f"Unsupported output suffix {suffix!r}. Use .pt, .json, or .jsonl.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect speculative decoding run summaries.")
    parser.add_argument("--input-root", type=Path, default=Path("./results"))
    parser.add_argument("--output", type=Path, default=Path("results.pt"))
    parser.add_argument("--request-start", type=int, default=4)
    parser.add_argument("--request-end", type=int, default=14)
    parser.add_argument("--step-start", type=int, default=1)
    parser.add_argument("--step-end", type=int, default=20)
    parser.add_argument("--pretty-json", action="store_true", help="Pretty print only when output suffix is .json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"input-root does not exist: {args.input_root}")

    data = collect(
        input_root=args.input_root,
        request_start=args.request_start,
        request_end=args.request_end,
        step_start=args.step_start,
        step_end=args.step_end,
    )

    write_output(data, args.output, pretty_json=args.pretty_json)

    print(f"[done] wrote {args.output}")
    print(
        "[summary] "
        f"motivation={len(data['motivation'])}, "
        f"insight={len(data['insight'])}, "
        f"trace={len(data['trace'])}, "
        f"results={len(data['results'])}"
    )


if __name__ == "__main__":
    main()
