#!/usr/bin/env python3
"""
Collect motivation, insight, and trace summaries from speculative decoding metadata runs.

Expected run layout:
  <input_root>/<method>/b<batch_size>/k<speculate_k>/<target>+<draft>/t<temperature>/<dataset>/
      analysis.jsonl
      metadata.jsonl

Default:
  python collect_results.py \
    --input-root ./results \
    --output results.jsonl

The input root is the global results root. Key-specific roots are:
  motivation, insight, trace: <input_root>/metadata
  result: <input_root>/cost

Output top-level schema:
  {
    "motivation": [...],
    "insight": [...],
    "trace": [...],
    "result": []
  }

Notes:
  - "motivation" reads analysis.jsonl and drops the "notes" field.
  - "insight" reads metadata.jsonl and computes misspeculation top-k inclusiveness
    and confidence correlation rows for target_accept_len == 0.
  - "trace" reads metadata.jsonl and stores target_accept_len for request IDs
    4..14 and step IDs 1..20 by default. Missing steps are stored as null.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _to_number_if_clean(value: str) -> Any:
    """Convert numeric-looking strings to int or float. Otherwise return original string."""
    if value == "":
        return value
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
                raise ValueError(f"Expected JSON object in {path} at line {line_no}, got {type(obj).__name__}")
            yield obj


def find_run_dirs(input_root: Path) -> List[Path]:
    """Return directories containing analysis.jsonl or metadata.jsonl."""
    run_dirs = set()
    for name in ("analysis.jsonl", "metadata.jsonl"):
        for path in input_root.rglob(name):
            if path.is_file():
                run_dirs.add(path.parent)
    return sorted(run_dirs)


def parse_context(run_dir: Path, input_root: Path) -> Dict[str, Any]:
    """
    Parse context from the run directory.

    Relative layout:
      method / b16 / k3 / target+draft / t0 / dataset
    """
    try:
        rel = run_dir.relative_to(input_root)
        parts = rel.parts
    except ValueError:
        parts = run_dir.parts

    if len(parts) < 6:
        raise ValueError(
            f"Cannot parse run context from {run_dir}. Expected at least 6 path components "
            f"under input_root: method/b*/k*/target+draft/t*/dataset"
        )

    method = parts[0]
    batch_size = _parse_prefixed_int(parts[1], "b", "batch size")
    speculate_k = _parse_prefixed_int(parts[2], "k", "speculate_k")
    model_pair = parts[3]
    temperature = _parse_prefixed_number(parts[4], "t", "temperature")
    dataset = parts[5]

    model_components = model_pair.split("+")
    target_model = model_components[0] if model_components else None
    draft_model = model_components[-1] if len(model_components) >= 2 else None

    ctx: Dict[str, Any] = {
        "method": method,
        "batch_size": batch_size,
        "speculate_k": speculate_k,
        "target_model": target_model,
        "draft_model": draft_model,
        "temperature": temperature,
        "dataset": dataset,
        "run_dir": str(run_dir),
    }

    if len(model_components) > 2:
        ctx["intermediate_models"] = model_components[1:-1]

    return ctx


def safe_context(run_dir: Path, input_root: Path) -> Optional[Dict[str, Any]]:
    try:
        return parse_context(run_dir, input_root)
    except Exception as exc:
        print(f"[warn] Skipping {run_dir}: {exc}")
        return None


def build_motivation_entry(analysis_path: Path, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for obj in iter_jsonl(analysis_path):
        analysis = {k: v for k, v in obj.items() if k != "notes"}
        entry = dict(ctx)
        entry.update(analysis)
        entries.append(entry)
    return entries


def is_target_misspeculation_row(row: Dict[str, Any]) -> bool:
    """A misspeculation row is target_accept_len == 0."""
    return row.get("target_accept_len") == 0


def compute_topk_inclusiveness(miss_rows: List[Dict[str, Any]], max_topk: int = 5) -> Dict[str, Any]:
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


def compute_confidence_correlation_rows(miss_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in miss_rows:
        confs = row.get("first_draft_token_confidence") or []
        if not isinstance(confs, list) or len(confs) < 1:
            top1_confidence = None
            residual_confidence = None
        else:
            top1_confidence = confs[0]
            residual_confidence = (confs[0] - confs[1]) if len(confs) >= 2 else None

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


def build_insight_entry(metadata_path: Path, ctx: Dict[str, Any]) -> Dict[str, Any]:
    rows = list(iter_jsonl(metadata_path))
    miss_rows = [row for row in rows if is_target_misspeculation_row(row)]

    entry = dict(ctx)
    entry["num_metadata_rows"] = len(rows)
    entry["num_misspeculation_steps"] = len(miss_rows)
    entry["misspeculation_topk_inclusiveness"] = compute_topk_inclusiveness(miss_rows, max_topk=5)
    entry["misspeculation_confidence_correlation"] = compute_confidence_correlation_rows(miss_rows)
    return entry


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

    run_dirs = find_run_dirs(metadata_root)

    for run_dir in run_dirs:
        ctx = safe_context(run_dir, metadata_root)
        if ctx is None:
            continue

        analysis_path = run_dir / "analysis.jsonl"
        metadata_path = run_dir / "metadata.jsonl"

        if analysis_path.exists():
            collected["motivation"].extend(build_motivation_entry(analysis_path, ctx))

        if metadata_path.exists():
            collected["insight"].append(build_insight_entry(metadata_path, ctx))
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
    """
    Placeholder for the result key backed by <input_root>/cost.

    The concrete cost aggregation schema will be added later. Keeping this function
    separate makes the key-specific input routing explicit now.
    """
    if not cost_root.exists():
        print(f"[warn] cost root does not exist, skipping result key: {cost_root}")
        return []
    return []


def collect(input_root: Path, request_start: int, request_end: int, step_start: int, step_end: int) -> Dict[str, Any]:
    """
    Collect all output keys from the global results root.

    Key routing:
      motivation, insight, trace -> <input_root>/metadata
      result                     -> <input_root>/cost
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
        "result": collect_from_cost_root(cost_root),
    }


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
        # Single JSONL record whose outer keys are motivation, insight, trace, result.
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        return

    raise ValueError(f"Unsupported output suffix {suffix!r}. Use .jsonl, .json, or .pt.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect speculative decoding metadata summaries.")
    parser.add_argument("--input-root", type=Path, default=Path("./results"))
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"))
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
        f"result={len(data['result'])}"
    )


if __name__ == "__main__":
    main()
