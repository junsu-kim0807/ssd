"""SSDProfiler: optional run/step profiling (cost, metadata, kernel trace).

See project plan: disabled when ``profiler_output_dir`` is unset; strict no-op
on hot paths. Run-level JSON is written only in ``finish_run()``; per-step
JSONL rows are appended in ``finish_step()`` only.

Prefill engine steps contribute only ``prefill_wall_time_s`` (outer step wall).
They do not add to draft/verify/sync/postprocess run totals and do not emit
prefill JSONL rows, in any ``profiler_mode``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol

ProfilerMode = Literal["cost_breakdown", "metadata", "cost_metadata", "kernel_breakdown"]

PROFILER_MODES: tuple[str, ...] = (
    "cost_breakdown",
    "metadata",
    "cost_metadata",
    "kernel_breakdown",
)


def profiler_is_active(profiler_output_dir: str | None) -> bool:
    return bool(profiler_output_dir and profiler_output_dir.strip())


def wants_metadata_rows(mode: str) -> bool:
    return mode in ("metadata", "cost_metadata")


def wants_cost_aggregates(mode: str) -> bool:
    return mode in ("cost_breakdown", "cost_metadata")


def wants_profile_trace(mode: str) -> bool:
    """Whether verifiers should attach VerifyProfileTrace (softmax); not cost_breakdown or kernel_breakdown."""
    return mode in ("metadata", "cost_metadata")


@dataclass
class StepProfileState:
    step_id: int
    is_prefill: bool
    batch_size: int
    seq_ids: list[int]
    step_wall_time_s: float = 0.0
    draft_time_s: float = 0.0
    verification_time_s: float = 0.0
    sync_time_s: float = 0.0
    postprocess_time_s: float = 0.0
    draft_time_worker_s: float = 0.0  # async draft process wall (one scalar per step)
    """Decode step: requests that entered draft / verify for this engine step (usually ``len(seqs)``)."""
    num_draft_requests_step: int = 0
    num_verification_requests_step: int = 0
    stage_stack: list[tuple[str, float]] = field(default_factory=list)

    def start_stage(self, name: str, t: float) -> None:
        self.stage_stack.append((name, t))

    def end_stage(self, name: str, t: float) -> None:
        if not self.stage_stack:
            return
        n, t0 = self.stage_stack.pop()
        if n != name:
            return
        dt = t - t0
        if name == "draft":
            self.draft_time_s += dt
        elif name in ("target_verify", "intermediate_verify", "pivot_verify", "verify"):
            self.verification_time_s += dt
        elif name == "sync":
            self.sync_time_s += dt
        elif name == "postprocess":
            self.postprocess_time_s += dt
        elif name == "draft_prefill":
            self.draft_time_s += dt
        elif name == "target_prefill":
            self.verification_time_s += dt


class ProfilerSink:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata_path = root / "metadata.jsonl"
        self._cost_metadata_path = root / "cost_metadata.jsonl"
        self._kernel_dir = root / "kernel_breakdown"
        self._opened: dict[str, Any] = {}

    def append_jsonl(self, rel: str, obj: dict[str, Any]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def write_cost_breakdown(self, payload: dict[str, Any]) -> None:
        path = self.root / "cost_breakdown.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def kernel_subdir(self) -> Path:
        self._kernel_dir.mkdir(parents=True, exist_ok=True)
        return self._kernel_dir


class _ProfilerProtocol(Protocol):
    def start_run(self, config: Any, tokenizer: Any) -> None: ...
    def finish_run(self) -> None: ...
    def start_step(self, seqs: list[Any], is_prefill: bool) -> None: ...
    def finish_step(self, num_output_tokens: int) -> None: ...
    def start_stage(self, stage_name: str) -> None: ...
    def finish_stage(self, stage_name: str) -> None: ...
    def on_async_draft_worker_time_s(self, dt: float) -> None: ...
    def on_async_draft_prefill_worker_time_s(self, dt: float) -> None: ...
    def add_step_async_spec_rpc_time_s(self, dt: float) -> None: ...
    def record_verify_step(self, seqs: list[Any], trace: Any | None) -> None: ...
    def record_decode_verify_batch(self, seqs: list[Any], verify_result: Any) -> None: ...
    def bump_draft_requests(self, n: int) -> None: ...
    def flush_spec_decode_rows(self, seqs: list[Any], is_prefill: bool, rows: list[dict[str, Any]]) -> None: ...
    def wants_metadata_computation(self) -> bool: ...
    def inter_target_counts_for_seq(self, seq_id: int) -> tuple[int, int]: ...


@dataclass
class _NoOpProfiler:
    def start_run(self, config: Any, tokenizer: Any) -> None:
        return

    def finish_run(self) -> None:
        return

    def start_step(self, seqs: list[Any], is_prefill: bool) -> None:
        return

    def finish_step(self, num_output_tokens: int) -> None:
        return

    def start_stage(self, stage_name: str) -> None:
        return

    def finish_stage(self, stage_name: str) -> None:
        return

    def on_async_draft_worker_time_s(self, dt: float) -> None:
        return

    def on_async_draft_prefill_worker_time_s(self, dt: float) -> None:
        return

    def add_step_async_spec_rpc_time_s(self, dt: float) -> None:
        return

    def record_verify_step(self, seqs: list[Any], trace: Any | None) -> None:
        return

    def record_decode_verify_batch(self, seqs: list[Any], verify_result: Any) -> None:
        return

    def bump_draft_requests(self, n: int) -> None:
        return

    def flush_spec_decode_rows(self, seqs: list[Any], is_prefill: bool, rows: list[dict[str, Any]]) -> None:
        return

    def wants_metadata_computation(self) -> bool:
        return False

    def inter_target_counts_for_seq(self, seq_id: int) -> tuple[int, int]:
        return (0, 0)


NOOP_PROFILER: _ProfilerProtocol = _NoOpProfiler()


class SSDProfiler:
    """Active profiler: keep in-memory run totals; append JSONL per step; flush JSON at end."""

    def __init__(self, config: Any) -> None:
        self._mode: ProfilerMode = getattr(config, "profiler_mode", "cost_metadata")  # type: ignore[assignment]
        self._sink = ProfilerSink(Path(config.profiler_output_dir))
        self._spec_policy = config.spec_policy
        self._draft_async = config.draft_async
        self._speculate_k = config.speculate_k
        self._tokenizer = None
        self._step_id = -1
        self._state: StepProfileState | None = None
        self._t_step_outer: float = 0.0

        # Run-level (finish_run only)
        self._run_execution_wall_s = 0.0
        self._run_prefill_wall_s = 0.0  # sum of engine-step walls where is_prefill=True (decode breakdown excluded)
        self._run_draft_s = 0.0
        self._run_verify_s = 0.0
        self._run_sync_s = 0.0
        self._run_postprocess_s = 0.0
        self._run_draft_worker_s = 0.0
        self._num_prefill_engine_steps = 0
        self._num_decode_engine_steps = 0
        self._run_prefill_tokens = 0
        self._run_decode_tokens = 0
        self._num_draft_requests = 0
        self._num_verification_requests = 0
        self._num_intermediate_verification_requests = 0
        self._num_target_verification_requests = 0

        # Per-seq observation counters (not sequence semantics)
        self._inter_verify_count_by_seq: dict[int, int] = {}
        self._target_verify_count_by_seq: dict[int, int] = {}

        self._kernel_prof: Any | None = None

    def start_run(self, config: Any, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        if self._mode == "kernel_breakdown":
            try:
                import torch
                from torch.profiler import ProfilerActivity, profile

                kdir = self._sink.kernel_subdir()
                rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
                trace_path = kdir / f"trace.rank{rank}.json"
                self._kernel_prof = profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    with_stack=False,
                    profile_memory=False,
                )
                self._kernel_prof.__enter__()
                self._kernel_meta_path = trace_path
            except Exception:
                self._kernel_prof = None

    def finish_run(self) -> None:
        if self._kernel_prof is not None:
            try:
                self._kernel_prof.__exit__(None, None, None)
                path = getattr(self, "_kernel_meta_path", None)
                if path is not None:
                    self._kernel_prof.export_chrome_trace(str(path))
            except Exception:
                pass
            self._kernel_prof = None

        if wants_cost_aggregates(self._mode):
            wall = float(self._run_execution_wall_s)
            decode_toks = int(self._run_decode_tokens)
            throughput = (decode_toks / wall) if wall > 0.0 else 0.0
            payload = {
                "execution_wall_time_s": self._run_execution_wall_s,
                "prefill_wall_time_s": self._run_prefill_wall_s,
                "num_prefill_token": int(self._run_prefill_tokens),
                "num_decode_tokens": decode_toks,
                "throughput": throughput,
                "draft_time_s": self._run_draft_s,
                "verification_time_s": self._run_verify_s,
                "sync_time_s": self._run_sync_s,
                "postprocess_time_s": self._run_postprocess_s,
                "draft_worker_wall_time_s": self._run_draft_worker_s,
                "num_prefill_engine_steps": self._num_prefill_engine_steps,
                "num_decode_engine_steps": self._num_decode_engine_steps,
                "num_draft": self._num_draft_requests,
                "num_verification": self._num_verification_requests,
                "num_intermediate_verification": self._num_intermediate_verification_requests,
                "num_target_verification": self._num_target_verification_requests,
                "notes": {
                    "num_prefill_token": "sum of per-prefill-engine-step token counts passed to finish_step",
                    "num_decode_tokens": "sum of per-decode-engine-step token counts passed to finish_step",
                    "throughput": "num_decode_tokens / execution_wall_time_s (decode tokens over total step wall)",
                    "num_draft": "cumulative count of requests entering draft stage",
                    "num_verification": "cumulative count of requests entering verification stage",
                    "num_decode_engine_steps": "LLMEngine.step calls where is_prefill=False",
                    "prefill_wall_time_s": (
                        "sum of outer prefill step wall times only; prefill does not contribute to "
                        "draft_time_s / verification_time_s / sync_time_s / postprocess_time_s below"
                    ),
                    "draft_worker_wall_time_s": "async decode steps: sum of draft-process wall tensors per step",
                    "draft_time_s": "async: same as draft_worker_wall_time_s; sync: rank0 draft stage wall",
                    "sync_time_s": (
                        "async: rank0 speculate RPC wall minus draft_worker_wall (protocol/wait overhead); "
                        "sync: 0"
                    ),
                    "cost_metadata_jsonl": (
                        "per-row draft_time_s uses draft worker wall when async (same basis as run-level draft_time_s)"
                    ),
                },
            }
            self._sink.write_cost_breakdown(payload)

    def start_step(self, seqs: list[Any], is_prefill: bool) -> None:
        self._step_id += 1
        self._t_step_outer = perf_counter()
        ids = [s.seq_id for s in seqs]
        self._state = StepProfileState(
            step_id=self._step_id,
            is_prefill=is_prefill,
            batch_size=len(seqs),
            seq_ids=ids,
        )

    def finish_step(self, num_output_tokens: int) -> None:
        if self._state is None:
            return
        st = self._state
        wall = perf_counter() - self._t_step_outer
        st.step_wall_time_s = wall

        self._run_execution_wall_s += wall
        if st.is_prefill:
            self._num_prefill_engine_steps += 1
            self._run_prefill_wall_s += wall
            self._run_prefill_tokens += int(num_output_tokens)
        else:
            self._num_decode_engine_steps += 1
            self._run_decode_tokens += int(num_output_tokens)
            # Decode steps only: stage timers and async worker scalars (prefill uses outer wall only).
            if self._draft_async:
                self._run_draft_s += st.draft_time_worker_s
            else:
                self._run_draft_s += st.draft_time_s
            self._run_verify_s += st.verification_time_s
            self._run_sync_s += st.sync_time_s
            self._run_postprocess_s += st.postprocess_time_s
            self._run_draft_worker_s += st.draft_time_worker_s

        if wants_metadata_rows(self._mode):
            # Rows require merge from step — SpecDecodeStep calls _flush_spec_decode_rows
            pass

        self._state = None

    def flush_spec_decode_rows(
        self,
        seqs: list[Any],
        is_prefill: bool,
        rows: list[dict[str, Any]],
    ) -> None:
        """Append one JSON object per request for this step (metadata / cost_metadata)."""
        if not wants_metadata_rows(self._mode):
            return
        for row in rows:
            if self._mode == "metadata":
                self._sink.append_jsonl("metadata.jsonl", row)
            else:
                self._sink.append_jsonl("cost_metadata.jsonl", row)

    def start_stage(self, stage_name: str) -> None:
        if self._state is not None:
            self._state.start_stage(stage_name, perf_counter())

    def finish_stage(self, stage_name: str) -> None:
        if self._state is not None:
            self._state.end_stage(stage_name, perf_counter())

    def on_async_draft_worker_time_s(self, dt: float) -> None:
        if self._state is not None:
            self._state.draft_time_worker_s += dt

    def on_async_draft_prefill_worker_time_s(self, dt: float) -> None:
        """Prefill is accounted only via outer step wall in finish_step; do not fold worker wall into draft totals."""
        return

    def add_step_async_spec_rpc_time_s(self, dt: float) -> None:
        if self._state is not None:
            self._state.sync_time_s += dt

    def bump_draft_requests(self, n: int) -> None:
        self._num_draft_requests += n
        if self._state is not None:
            self._state.num_draft_requests_step = n

    def inter_target_counts_for_seq(self, seq_id: int) -> tuple[int, int]:
        return (
            self._inter_verify_count_by_seq.get(seq_id, 0),
            self._target_verify_count_by_seq.get(seq_id, 0),
        )

    def record_verify_step(self, seqs: list[Any], trace: Any | None) -> None:
        if trace is None:
            return
        vms = getattr(trace, "verification_models", None)
        if not vms:
            return
        self._num_verification_requests += len(vms)
        for seq, vm in zip(seqs, vms):
            sid = seq.seq_id
            if vm == "intermediate":
                self._num_intermediate_verification_requests += 1
                self._inter_verify_count_by_seq[sid] = self._inter_verify_count_by_seq.get(sid, 0) + 1
            elif vm in ("target", "pivot_target"):
                self._num_target_verification_requests += 1
                self._target_verify_count_by_seq[sid] = self._target_verify_count_by_seq.get(sid, 0) + 1
            elif vm == "pivot_intermediate":
                self._num_intermediate_verification_requests += 1
                self._inter_verify_count_by_seq[sid] = self._inter_verify_count_by_seq.get(sid, 0) + 1

    def record_decode_verify_batch(self, seqs: list[Any], verify_result: Any) -> None:
        n = len(seqs)
        if self._state is not None:
            self._state.num_verification_requests_step = n
        trace = getattr(verify_result, "profile_trace", None)
        if trace is not None:
            self.record_verify_step(seqs, trace)
            return
        self._num_verification_requests += n
        if getattr(verify_result, "is_hv_intermediate", False):
            self._num_intermediate_verification_requests += n
            for seq in seqs:
                sid = seq.seq_id
                self._inter_verify_count_by_seq[sid] = self._inter_verify_count_by_seq.get(sid, 0) + 1
        else:
            self._num_target_verification_requests += n
            for seq in seqs:
                sid = seq.seq_id
                self._target_verify_count_by_seq[sid] = self._target_verify_count_by_seq.get(sid, 0) + 1

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def speculate_k(self) -> int:
        return self._speculate_k

    @property
    def spec_policy(self) -> str:
        return self._spec_policy

    @property
    def draft_async(self) -> bool:
        return self._draft_async

    @property
    def current_step_state(self) -> StepProfileState | None:
        return self._state

    def current_step_elapsed_s(self) -> float:
        if self._state is None:
            return 0.0
        return perf_counter() - self._t_step_outer

    @property
    def step_id(self) -> int:
        return self._step_id

    def wants_kernel(self) -> bool:
        return self._mode == "kernel_breakdown"

    def wants_metadata_computation(self) -> bool:
        return self._mode in ("metadata", "cost_metadata")


def make_profiler(config: Any) -> _ProfilerProtocol:
    if not profiler_is_active(getattr(config, "profiler_output_dir", None)):
        return NOOP_PROFILER
    mode = getattr(config, "profiler_mode", "cost_metadata")
    if mode not in PROFILER_MODES:
        return NOOP_PROFILER
    return SSDProfiler(config)
