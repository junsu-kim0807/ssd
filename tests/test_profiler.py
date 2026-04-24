"""Unit tests for SSDProfiler and metadata helpers (no full Config / HF models)."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ssd.utils.profiler import (
    SSDProfiler,
    _target_accept_len_distribution_tables,
    make_profiler,
    NOOP_PROFILER,
)
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
    assert data["num_prefill_token"] == 1
    assert data["num_decode_tokens"] == 0
    assert data["prefill_wall_time_s"] > 0
    assert data["draft_time_s"] == 0.0
    assert data["verification_time_s"] == 0.0


def test_metadata_finish_run_writes_analysis_jsonl(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="metadata",
            profiler_output_dir=str(tmp_path),
            spec_policy="default",
        )
    )

    class S:
        seq_id = 0

    p.start_run(SimpleNamespace(), None)
    p.start_step([S(), S()], is_prefill=False)
    tr = SimpleNamespace(
        verification_models=["target", "target"],
        accept_len=[0, 3],
        inter_accept_len=None,
        inter_target_prefix_accept_len=None,
    )
    vr = SimpleNamespace(profile_trace=tr, is_hv_intermediate=False)
    p.record_decode_verify_batch([S(), S()], vr)
    p.finish_step(1)
    p.finish_run()
    ap = tmp_path / "analysis.jsonl"
    assert ap.is_file()
    line = ap.read_text(encoding="utf-8").strip().splitlines()[-1]
    data = json.loads(line)
    assert data["total_target_verification_rounds"] == 2
    assert data["misspeculation_rounds"] == 1
    assert abs(data["misspeculation_probability"] - 0.5) < 1e-9
    tbd = data["target_batch_accept_distributions"]
    assert isinstance(tbd, dict)
    assert abs(tbd["0"] - 0.5) < 1e-9
    assert abs(tbd["3"] - 0.5) < 1e-9
    assert data["avg_intermediate_accept_len"] is None
    assert data["accept_distribution_rounds"] == 2
    rp = data["accept_rate_per_position"]
    lp = data["accept_length_per_round"]
    assert abs(rp["0"] - 0.5) < 1e-9
    assert abs(rp["1"] - 0.5) < 1e-9
    assert abs(rp["2"] - 0.5) < 1e-9
    assert rp["3"] == 0.0
    assert abs(lp["0"] - 0.5) < 1e-9
    assert abs(lp["3"] - 0.5) < 1e-9
    assert abs(sum(lp.values()) - 1.0) < 1e-9
    for k in ("accept_rate_per_position", "accept_length_per_round"):
        assert k in data["notes"]


def test_metadata_target_batch_hist_averaged_over_steps(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="metadata",
            profiler_output_dir=str(tmp_path),
            spec_policy="default",
        )
    )

    class S:
        seq_id = 0

    p.start_run(SimpleNamespace(), None)
    p.start_step([S(), S()], is_prefill=False)
    tr1 = SimpleNamespace(
        verification_models=["target", "target"],
        accept_len=[0, 3],
        inter_accept_len=None,
        inter_target_prefix_accept_len=None,
    )
    p.record_decode_verify_batch([S(), S()], SimpleNamespace(profile_trace=tr1, is_hv_intermediate=False))
    p.finish_step(1)
    p.start_step([S(), S()], is_prefill=False)
    tr2 = SimpleNamespace(
        verification_models=["target", "target"],
        accept_len=[0, 0],
        inter_accept_len=None,
        inter_target_prefix_accept_len=None,
    )
    p.record_decode_verify_batch([S(), S()], SimpleNamespace(profile_trace=tr2, is_hv_intermediate=False))
    p.finish_step(1)
    p.finish_run()
    data = json.loads((tmp_path / "analysis.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    tbd = data["target_batch_accept_distributions"]
    assert abs(tbd["0"] - 0.75) < 1e-9
    assert abs(tbd["3"] - 0.25) < 1e-9


def test_target_accept_len_distribution_tables():
    rate, pmf = _target_accept_len_distribution_tables([7, 3, 0], speculate_k=3)
    assert rate["0"] == 2 / 3
    assert abs(rate["6"] - 1 / 3) < 1e-9
    assert rate["7"] == 0.0
    assert abs(sum(pmf.values()) - 1.0) < 1e-9
    assert abs(pmf["7"] - 1 / 3) < 1e-9
    for i in range(7):
        assert rate[str(i)] >= rate[str(i + 1)]


def test_metadata_analysis_hierarchical_avgs(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="metadata",
            profiler_output_dir=str(tmp_path),
            spec_policy="hierarchical",
        )
    )

    class S:
        seq_id = 0

    p.start_run(SimpleNamespace(), None)
    p.start_step([S()], is_prefill=False)
    tr_i = SimpleNamespace(
        verification_models=["intermediate"],
        accept_len=[0],
        inter_accept_len=[2],
        inter_target_prefix_accept_len=None,
    )
    p.record_decode_verify_batch([S()], SimpleNamespace(profile_trace=tr_i, is_hv_intermediate=True))
    tr_t = SimpleNamespace(
        verification_models=["target"],
        accept_len=[1],
        inter_accept_len=None,
        inter_target_prefix_accept_len=[5],
    )
    p.record_decode_verify_batch([S()], SimpleNamespace(profile_trace=tr_t, is_hv_intermediate=False))
    p.finish_step(1)
    p.finish_run()
    data = json.loads((tmp_path / "analysis.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert abs(data["avg_intermediate_accept_len"] - 2.0) < 1e-9
    assert abs(data["avg_target_accept_len"] - 1.0) < 1e-9
    assert abs(data["avg_inter_target_prefix_accept_len"] - 5.0) < 1e-9


def test_cost_metadata_does_not_write_analysis_jsonl(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_metadata",
            profiler_output_dir=str(tmp_path),
        )
    )
    p.start_run(SimpleNamespace(), None)
    p.finish_run()
    assert not (tmp_path / "analysis.jsonl").exists()


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
    assert data["num_decode_tokens"] == 1
    assert data["num_preemption"] == 0
    assert abs(data["avg_decode_scheduled_batch_size"] - 1.0) < 1e-9
    assert data["num_prefill_token"] == 0
    assert data["throughput"] > 0
    assert "avg_target_accept_len" in data
    assert data["avg_target_accept_len"] is None
    assert "hierarchical_intermediate_verification_time_s" not in data
    assert data["avg_draft_time_per_batch"] is None
    assert data["avg_intermediate_verification_time_per_batch"] is None
    assert data["avg_target_verification_time_per_batch"] is None


def test_cost_breakdown_hierarchical_fields(tmp_path):
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_breakdown",
            profiler_output_dir=str(tmp_path),
            spec_policy="hierarchical",
        )
    )

    class S:
        seq_id = 0

    p.start_run(SimpleNamespace(), None)
    p.start_step([S(), S()], is_prefill=False)
    p.bump_draft_requests(2)
    p.start_stage("draft")
    p.finish_stage("draft")
    p.accum_hierarchical_verify_time(0.02, True)
    p.record_decode_verify_batch(
        [S(), S()], SimpleNamespace(profile_trace=None, is_hv_intermediate=True)
    )
    p.finish_step(1)
    p.start_step([S(), S()], is_prefill=False)
    p.bump_draft_requests(2)
    p.start_stage("draft")
    p.finish_stage("draft")
    p.accum_hierarchical_verify_time(0.05, False)
    p.record_decode_verify_batch(
        [S(), S()], SimpleNamespace(profile_trace=None, is_hv_intermediate=False)
    )
    p.finish_step(1)
    tr_i = SimpleNamespace(
        verification_models=["intermediate"],
        inter_accept_len=[3],
    )
    p._record_profile_accept_samples([S()], tr_i)
    tr = SimpleNamespace(
        verification_models=["target"],
        accept_len=[2],
        inter_target_prefix_accept_len=[1],
    )
    p._record_profile_accept_samples([S()], tr)
    p.finish_run()
    data = json.loads((tmp_path / "cost_breakdown.json").read_text())
    assert data["hierarchical_intermediate_verification_time_s"] == 0.02
    assert data["hierarchical_target_verification_time_s"] == 0.05
    assert abs(data["avg_target_accept_len"] - 2.0) < 1e-6
    assert abs(data["avg_inter_target_prefix_accept_len"] - 1.0) < 1e-6
    assert abs(data["avg_intermediate_accept_len"] - 3.0) < 1e-6
    assert abs(data["avg_decode_scheduled_batch_size"] - 2.0) < 1e-9
    assert abs(data["hv_avg_verify_batch_size_intermediate"] - 2.0) < 1e-9
    assert abs(data["hv_avg_verify_batch_size_target"] - 2.0) < 1e-9
    # avg_decode_scheduled_batch_size = 2; num_draft = 4 → draft_time / (4/2) = draft_time/2
    assert data["avg_draft_time_per_batch"] is not None
    assert abs(data["avg_draft_time_per_batch"] - data["draft_time_s"] / 2.0) < 1e-9
    assert abs(data["avg_intermediate_verification_time_per_batch"] - 0.02) < 1e-9
    assert abs(data["avg_target_verification_time_per_batch"] - 0.05) < 1e-9


def test_profile_greedy_token_confidence_not_all_ones():
    from ssd.utils.profiler_metadata import profile_greedy_token_confidence

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 4, 128, generator=g)
    greedy, conf = profile_greedy_token_confidence(logits)
    assert greedy.shape == (2, 4)
    assert conf.shape == (2, 4)
    assert not torch.allclose(conf, torch.ones_like(conf))


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


def test_trace_to_row_indexed_hierarchical_intermediate_chain_columns(tmp_path):
    """Intermediate verify trace must not populate target_*; chain goes to intermediate_verify_chain_*."""
    p = SSDProfiler(
        _prof_cfg(
            profiler_mode="cost_metadata",
            profiler_output_dir=str(tmp_path),
            spec_policy="hierarchical",
        )
    )
    p.start_run(SimpleNamespace(), None)

    class S:
        seq_id = 1

    p.start_step([S()], is_prefill=False)
    tr = SimpleNamespace(
        verification_models=["intermediate"],
        token_ids_per_position=[[10, 11, 12]],
        token_confidence_per_position=[[0.5, 0.6, 0.7]],
        accept_len=[0],
        recovery_tokens=[99],
        bonus_tokens=[88],
        inter_token_ids_per_position=[[10, 11]],
        inter_token_confidence_per_position=[[0.5, 0.6]],
        inter_accept_len=[1],
        inter_recovery_token=[99],
        inter_bonus_token=[88],
    )
    row = trace_to_row_indexed(
        profiler=p,
        seq=S(),
        batch_index=0,
        batch_size=1,
        is_prefill=False,
        speculate_k=2,
        spec_policy="hierarchical",
        draft_async=False,
        cache_hit=None,
        trace=tr,
        first_draft_token_ids=[0, 0, 0, 0, 0],
        first_draft_token_confidence=[0.0, 0.0, 0.0, 0.0, 0.0],
        draft_token_ids_per_position=[0, 0],
        draft_token_confidence_per_position=[0.0, 0.0],
        step_wall_time_s=0.01,
        draft_time_s=0.0,
        verification_time_s=0.0,
        sync_time_s=0.0,
        num_draft=1,
        num_verification=1,
        cost_fields=True,
    )
    assert row["verification_model"] == "intermediate"
    assert row["target_token_ids_per_position"] is None
    assert row["target_accept_len"] is None
    assert row["intermediate_verify_chain_token_ids_per_position"] == [10, 11, 12]
    assert row["intermediate_verify_chain_token_confidence_per_position"] == [0.5, 0.6, 0.7]
    assert row["inter_accept_len"] == 1


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
