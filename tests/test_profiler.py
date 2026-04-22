"""Unit tests for SSDProfiler and metadata helpers (no full Config / HF models)."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ssd.utils.profiler import SSDProfiler, make_profiler, NOOP_PROFILER
from ssd.utils.profiler_metadata import (
    draft_metadata_from_logits,
    prefill_metadata_rows,
    trace_to_row_indexed,
)


def _prof_cfg(**kwargs):
    base = dict(
        profiler_mode="cost_metadata",
        profiler_output_dir=None,
        spec_policy="default",
        draft_async=False,
        speculate_k=2,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_make_profiler_noop_without_output_dir():
    p = make_profiler(_prof_cfg(profiler_output_dir=None))
    assert p is NOOP_PROFILER


def test_draft_metadata_from_logits_shapes():
    b, k, v = 2, 3, 50
    logits_q = torch.randn(b, k, v)
    spec = torch.zeros(b, k + 1, dtype=torch.long)
    spec[:, 0] = 1
    spec[:, 1:] = torch.randint(0, v, (b, k))
    f_ids, f_conf, d_ids, d_conf = draft_metadata_from_logits(logits_q, spec, k)
    assert len(f_ids) == b and len(f_ids[0]) == 5
    assert len(d_ids[0]) == k


def test_prefill_step_only_prefill_wall_not_draft(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_breakdown",
            profiler_output_dir=str(tmp_path),
        )
    )

    class S:
        seq_id = 0

    p.start_run(SimpleNamespace(), None)
    p.start_step([S()], is_prefill=True)
    p.start_stage("draft_prefill")
    p.finish_stage("draft_prefill")
    p.start_stage("target_prefill")
    p.finish_stage("target_prefill")
    p.finish_step(1)
    p.finish_run()
    data = json.loads((tmp_path / "cost_breakdown.json").read_text())
    assert data["num_prefill_engine_steps"] == 1
    assert data["prefill_wall_time_s"] > 0
    assert data["draft_time_s"] == 0.0
    assert data["verification_time_s"] == 0.0


def test_cost_breakdown_finish_run_writes_json(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_breakdown",
            profiler_output_dir=str(tmp_path),
        )
    )

    class S:
        seq_id = 0

    p.start_run(SimpleNamespace(), None)
    p.start_step([S()], is_prefill=False)
    p.finish_step(1)
    p.finish_run()
    out = tmp_path / "cost_breakdown.json"
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["num_decode_engine_steps"] == 1


def test_trace_to_row_indexed_cost_fields(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_metadata",
            profiler_output_dir=str(tmp_path),
        )
    )
    p.start_run(SimpleNamespace(), None)

    class S:
        seq_id = 7

    p.start_step([S()], is_prefill=False)

    tr = SimpleNamespace(
        verification_models=["target"],
        token_ids_per_position=[[1, 2, 3]],
        token_confidence_per_position=[[0.1, 0.2, 0.3]],
        accept_len=[2],
        recovery_tokens=[9],
        bonus_tokens=[4],
        inter_token_ids_per_position=None,
        inter_token_confidence_per_position=None,
        inter_accept_len=None,
        inter_recovery_token=None,
        inter_bonus_token=None,
    )
    row = trace_to_row_indexed(
        profiler=p,
        seq=S(),
        batch_index=0,
        batch_size=1,
        is_prefill=False,
        speculate_k=2,
        spec_policy="default",
        draft_async=False,
        cache_hit=1,
        trace=tr,
        first_draft_token_ids=[1, 0, 0, 0, 0],
        first_draft_token_confidence=[0.5, 0.0, 0.0, 0.0, 0.0],
        draft_token_ids_per_position=[1, 2],
        draft_token_confidence_per_position=[0.4, 0.3],
        step_wall_time_s=0.01,
        draft_time_s=0.02,
        verification_time_s=0.03,
        sync_time_s=0.0,
        num_draft=1,
        num_verification=1,
        cost_fields=True,
    )
    assert row["request_id"] == 7
    assert row["step_wall_time_s"] == 0.01
    assert row["verification_model"] == "target"


def test_prefill_metadata_rows(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_metadata",
            profiler_output_dir=str(tmp_path),
            draft_async=True,
        )
    )
    p.start_run(SimpleNamespace(), None)

    class S:
        seq_id = 3

    seqs = [S(), S()]
    p.start_step(seqs, is_prefill=True)
    p.start_stage("draft_prefill")
    p.finish_stage("draft_prefill")
    p.start_stage("target_prefill")
    p.finish_stage("target_prefill")
    rows = prefill_metadata_rows(
        profiler=p,
        seqs=seqs,
        speculate_k=2,
        spec_policy="default",
        draft_async=True,
        cost_fields=True,
    )
    assert len(rows) == 2
    assert rows[0]["is_prefill"] is True
    assert rows[0]["num_draft"] == 2 and rows[0]["num_verification"] == 2
