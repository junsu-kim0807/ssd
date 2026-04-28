from abc import ABC, abstractmethod
import json
import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from ssd.engine.spec_policy_traits import uses_pivot_tree_scratch
from ssd.engine.verifier_hierarchical import VerifierHierarchical
from ssd.utils.misc import decode_tokens
from ssd.utils.profiler import SSDProfiler
from ssd.utils.profiler_metadata import draft_metadata_from_logits, trace_to_row_indexed


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        pass

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner, tokenizer: AutoTokenizer):
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer

    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if __debug__:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)

        prefill_query_tokens = 0
        if is_prefill:
            for seq in seqs:
                remain = len(seq) - seq.num_cached_tokens
                prefill_query_tokens += 1 if remain == 0 else remain

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if __debug__:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else prefill_query_tokens

    def prefill(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=True)

    def decode(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=False)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
        eagle: bool,
        tokenizer: AutoTokenizer,
        async_spec: bool,
        profiler: object | None = None,
    ):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.eagle = eagle
        self.tokenizer = tokenizer
        self.async_spec = async_spec
        self.profiler = profiler

    def _hv_correctness_debug_log(
        self,
        seqs: list[Sequence],
        speculate_result,
        out_verify_result,
        pre_comp_lens: list[int],
        round_idx_start: list[int],
        dbg_target_cands: list[list[int]] | None,
        fused_subround_idx: int | None = None,
    ) -> None:
        """One JSON line per sequence when ``config.debug_mode`` and hierarchical (bench --debug)."""
        if not isinstance(self.verifier, VerifierHierarchical):
            return
        spec_rows = speculate_result.speculations.detach().cpu().tolist()
        for bi, seq in enumerate(seqs):
            row = {
                "hv_correctness_debug": True,
                "seq_id": seq.seq_id,
                "hv_round_idx_at_step_start": round_idx_start[bi],
                "draft_spec_token_ids": spec_rows[bi],
                "intermediate_verify_accepted_token_ids": out_verify_result.new_suffixes[bi]
                if out_verify_result.is_hv_intermediate
                else None,
                "target_verify_accepted_token_ids": out_verify_result.new_suffixes[bi]
                if not out_verify_result.is_hv_intermediate
                else None,
                "target_verify_input_token_ids": dbg_target_cands[bi] if dbg_target_cands is not None else None,
                "verify_recovery_token_id": out_verify_result.recovery_tokens[bi],
                "output_new_completion_token_ids": list(
                    seq.completion_token_ids[pre_comp_lens[bi] :]
                ),
            }
            if fused_subround_idx is not None:
                row["hv_fused_subround_idx"] = int(fused_subround_idx)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def _profiler_active(self) -> bool:
        return isinstance(self.profiler, SSDProfiler)

    def _spec_correctness_debug_log(
        self,
        seqs: list[Sequence],
        speculate_result,
        out_verify_result,
        pre_comp_lens: list[int],
    ) -> None:
        """Non-hierarchical debug JSON rows mirroring HV correctness log schema."""
        spec_rows = speculate_result.speculations.detach().cpu().tolist()
        winner_rows = getattr(out_verify_result, "winning_branch_row_idx_per_parent", None)
        for bi, seq in enumerate(seqs):
            # Vanilla/non-pivot path: row index equals batch index.
            # Pivot flat path: ``spec_rows`` is expanded-row shaped, so use the selected
            # winner absolute row index for this parent request.
            row_idx = bi
            if winner_rows is not None and bi < len(winner_rows) and winner_rows[bi] is not None:
                cand = int(winner_rows[bi])
                if 0 <= cand < len(spec_rows):
                    row_idx = cand
            row = {
                "hv_correctness_debug": True,
                "seq_id": seq.seq_id,
                "hv_round_idx_at_step_start": 0,
                "draft_spec_token_ids": spec_rows[row_idx],
                "intermediate_verify_accepted_token_ids": None,
                "target_verify_accepted_token_ids": out_verify_result.new_suffixes[bi],
                # Vanilla speculative verify consumes one K+1 candidate row.
                "target_verify_input_token_ids": spec_rows[row_idx],
                "verify_recovery_token_id": out_verify_result.recovery_tokens[bi],
                "output_new_completion_token_ids": list(
                    seq.completion_token_ids[pre_comp_lens[bi] :]
                ),
                "hv_fused_subround_idx": 0,
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def prefill(self, seqs: list[Sequence]) -> int:
        # Prefill timing: only the outer engine step wall (LLMEngine start_step/finish_step).
        # No inner draft_prefill/target_prefill stages and no per-request JSONL rows — all profiler modes.

        actual_prefill_tokens = 0
        for seq in seqs:
            remain = len(seq) - seq.num_cached_tokens
            actual_prefill_tokens += 1 if remain == 0 else remain

        if not self.eagle and self.async_spec:
            empty_verify_result = VerifyResult([], [], None)
            self.speculator.prefill(seqs, empty_verify_result)
            verify_result = self.verifier.prefill(seqs, eagle=False)
        else:
            verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
            self.speculator.prefill(seqs, verify_result)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            # After (re)prefill, target/draft KV cover the full committed tape ``token_ids``
            # (``num_tokens``); keep prompt boundary only in ``num_prompt_tokens``.
            ntok = seq.num_tokens
            seq.num_cached_tokens = ntok
            seq.num_draft_cached_tokens = ntok
            if getattr(self.scheduler, "hierarchical", False):
                seq.num_inter_cached_tokens = ntok

        return actual_prefill_tokens

    def decode(self, seqs: list[Sequence]) -> int:
        if hasattr(self.speculator, "reset_step_state"):
            self.speculator.reset_step_state()
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _t0 = perf_counter()

        pr = self.profiler
        _nvtx = None
        if self._profiler_active() and pr.wants_kernel():
            try:
                import torch.cuda.nvtx as _nvtx_mod

                _nvtx = _nvtx_mod
                _nvtx.range_push("spec_decode")
            except Exception:
                _nvtx = None

        hierarchical = getattr(self.scheduler, "hierarchical", False)
        dbg_hv_pre_comp: list[int] | None = None
        dbg_hv_round0: list[int] | None = None
        dbg_target_cands: list[list[int]] | None = None
        if hierarchical and getattr(self.scheduler.config, "debug_mode", False):
            dbg_hv_pre_comp = [seq.num_completion_tokens for seq in seqs]
            dbg_hv_round0 = [seq.hv_round_idx for seq in seqs]
        dbg_spec_pre_comp: list[int] | None = None
        if (not hierarchical) and getattr(self.scheduler.config, "debug_mode", False):
            dbg_spec_pre_comp = [seq.num_completion_tokens for seq in seqs]

        if hierarchical:
            saved = [
                (
                    len(seq.token_ids),
                    seq.num_tokens,
                    seq.last_token,
                    seq.num_draft_cached_tokens,
                    seq.num_cached_tokens,
                )
                for seq in seqs
            ]
        else:
            saved = [
                (len(seq.token_ids), seq.num_tokens, seq.last_token, seq.num_draft_cached_tokens, seq.num_cached_tokens)
                for seq in seqs
            ]

        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )
        scratch_owner = None
        speculate_result = None
        out_verify_result = None
        pivot_tree_policy = uses_pivot_tree_scratch(getattr(self.scheduler.config, "spec_policy", ""))
        def _merge_scratch_owner(cur, other):
            if other is None:
                return cur
            if cur is None:
                return other
            if hasattr(cur, "merge"):
                return cur.merge(other)
            return cur
        try:
            #### STEP 1: SPECULATE ####
            if self._profiler_active():
                pr.bump_draft_requests(len(seqs))
                pr.start_stage("draft")
            speculate_result = self.speculator.speculate(seqs, in_verify_result)
            if self._profiler_active():
                pr.finish_stage("draft")
            scratch_owner = _merge_scratch_owner(
                scratch_owner,
                getattr(getattr(speculate_result, "branch_bundle", None), "scratch_owner", None),
            )

            if _prof:
                torch.cuda.synchronize()
                _t1 = perf_counter()

            #### STEP 2: VERIFY ####
            if self._profiler_active():
                if hierarchical:
                    _t_hv_verify = perf_counter()
                else:
                    pr.start_stage("verify")
            out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)
            scratch_owner = _merge_scratch_owner(
                scratch_owner,
                getattr(getattr(out_verify_result, "scratch_commit_bundle", None), "scratch_owner", None),
            )
            if self._profiler_active():
                if hierarchical:
                    pr.accum_hierarchical_verify_time(
                        perf_counter() - _t_hv_verify, out_verify_result.is_hv_intermediate
                    )
                else:
                    pr.finish_stage("verify")
                pr.record_decode_verify_batch(seqs, out_verify_result)

            if _prof:
                torch.cuda.synchronize()
                _t2 = perf_counter()

            if hierarchical and getattr(self.scheduler.config, "debug_mode", False):
                if isinstance(self.verifier, VerifierHierarchical) and not out_verify_result.is_hv_intermediate:
                    dbg_target_cands = self.verifier._build_target_candidates(seqs, speculate_result)

            # Restore original seq state before postprocess (undo speculate + verify modifications)
            if hierarchical:
                for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
                    del seq.token_ids[orig_len:]
                    seq.num_tokens = orig_nt
                    seq.last_token = orig_lt
                    seq.num_draft_cached_tokens = orig_ndc
                    seq.num_cached_tokens = orig_nct
                    # Keep ``num_inter_cached_tokens`` from verify (e.g. += K+1 on intermediate).
            else:
                for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
                    del seq.token_ids[orig_len:]
                    seq.num_tokens = orig_nt
                    seq.last_token = orig_lt
                    seq.num_draft_cached_tokens = orig_ndc
                    seq.num_cached_tokens = orig_nct

            #### STEP 3: POSTPROCESS ####
            dbg_vanilla_raw_suffixes: list[list[int]] | None = None
            dbg_vanilla_trunc_preview: list[tuple[list[int], bool]] | None = None
            if (
                (not hierarchical)
                and dbg_spec_pre_comp is not None
                and getattr(self.scheduler.config, "debug_mode", False)
            ):
                dbg_vanilla_raw_suffixes = [list(s) for s in out_verify_result.new_suffixes]
                dbg_vanilla_trunc_preview = []
                for seq, raw in zip(seqs, dbg_vanilla_raw_suffixes):
                    trunc, finished = self.scheduler._handle_eos_and_max_new_tokens(seq, list(raw))
                    dbg_vanilla_trunc_preview.append((trunc, finished))

            if self._profiler_active():
                pr.start_stage("postprocess")
            if hierarchical:
                mode = getattr(out_verify_result, "postprocess_mode", None)
                is_inter = out_verify_result.is_hv_intermediate if mode is None else (mode == "hv_intermediate")
                if is_inter:
                    self.scheduler.postprocess_hv_intermediate_round(
                        seqs,
                        out_verify_result.new_suffixes,
                        out_verify_result.recovery_tokens,
                    )
                else:
                    self.scheduler.postprocess_hv_target_round(
                        seqs,
                        out_verify_result.new_suffixes,
                        out_verify_result.recovery_tokens,
                        eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
                    )
            else:
                if pivot_tree_policy:
                    self.scheduler.postprocess_pivot_tree_scratch(
                        seqs,
                        out_verify_result.new_suffixes,
                        out_verify_result.recovery_tokens,
                        commit_bundle=out_verify_result.scratch_commit_bundle,
                        eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
                        target_model_runner=getattr(self.verifier, "target_model_runner", None),
                        draft_model_runner=getattr(self.verifier, "draft_model_runner", None),
                    )
                else:
                    self.scheduler.postprocess_speculate(
                        seqs,
                        out_verify_result.new_suffixes,
                        out_verify_result.recovery_tokens,
                        eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
                    )
            if self._profiler_active():
                pr.finish_stage("postprocess")

            if (
                hierarchical
                and dbg_hv_pre_comp is not None
                and dbg_hv_round0 is not None
                and getattr(self.scheduler.config, "debug_mode", False)
            ):
                self._hv_correctness_debug_log(
                    seqs,
                    speculate_result,
                    out_verify_result,
                    dbg_hv_pre_comp,
                    dbg_hv_round0,
                    dbg_target_cands,
                )
            elif (
                (not hierarchical)
                and dbg_spec_pre_comp is not None
                and getattr(self.scheduler.config, "debug_mode", False)
            ):
                self._spec_correctness_debug_log(
                    seqs,
                    speculate_result,
                    out_verify_result,
                    dbg_spec_pre_comp,
                )
                if (
                    dbg_vanilla_raw_suffixes is not None
                    and dbg_vanilla_trunc_preview is not None
                ):
                    for i, seq in enumerate(seqs):
                        raw = dbg_vanilla_raw_suffixes[i]
                        trunc, finished_preview = dbg_vanilla_trunc_preview[i]
                        post_delta = seq.num_completion_tokens - dbg_spec_pre_comp[i]
                        row = {
                            "hv_target_commit_debug": True,
                            "seq_id": seq.seq_id,
                            "pre_num_tokens": int(saved[i][1]),
                            "pre_completion_tokens": dbg_spec_pre_comp[i],
                            "raw_suffix_len": len(raw),
                            "raw_suffix_head": raw[:3],
                            "truncated_suffix_len_preview": len(trunc),
                            "truncated_suffix_head_preview": trunc[:3],
                            "post_completion_delta": post_delta,
                            "post_num_tokens": seq.num_tokens,
                            "next_recovery_token": int(out_verify_result.recovery_tokens[i]),
                            "finished_preview": bool(finished_preview),
                            "is_finished": bool(seq.is_finished),
                            "lost_by_truncation_or_commit": len(raw) - post_delta,
                        }
                        print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            if scratch_owner is not None:
                scratch_owner.release_unreleased(
                    self.scheduler.block_manager, self.scheduler.draft_block_manager
                )

        if _prof:
            torch.cuda.synchronize()
            _t3 = perf_counter()
            cache_hits = speculate_result.cache_hits
            hits_str = f"hits={cache_hits.sum().item()}/{len(cache_hits)}" if cache_hits is not None else ""
            toks = sum(len(s) for s in out_verify_result.new_suffixes)
            print(f"[PROFILE target] handshake={(_t1-_t0)*1000:.2f}ms verify={(_t2-_t1)*1000:.2f}ms postprocess={(_t3-_t2)*1000:.2f}ms total={(_t3-_t0)*1000:.2f}ms {hits_str} toks={toks}", flush=True)

        if self._profiler_active() and pr.wants_metadata_computation():
            st = pr.current_step_state
            if st is not None and speculate_result.logits_q is not None and speculate_result.logits_q.numel() > 0:
                k = self.verifier.lookahead
                f_ids, f_conf, d_ids, d_conf = draft_metadata_from_logits(
                    speculate_result.logits_q, speculate_result.speculations, k
                )
                bundle = getattr(speculate_result, "branch_bundle", None)
                winners = getattr(out_verify_result, "winning_branch_idx_per_parent", None)
                if bundle is not None and winners is not None:
                    # Collapse expanded-row draft metadata to parent winners.
                    by_parent_rows: list[list[int]] = [[] for _ in range(len(seqs))]
                    for row_idx, pidx in enumerate(bundle.parent_index_per_branch):
                        by_parent_rows[int(pidx)].append(row_idx)
                    f_ids_w: list[list[int]] = []
                    f_conf_w: list[list[float]] = []
                    d_ids_w: list[list[int]] = []
                    d_conf_w: list[list[float]] = []
                    for pidx in range(len(seqs)):
                        rows = by_parent_rows[pidx]
                        if rows:
                            w = int(winners[pidx]) if pidx < len(winners) else 0
                            w = max(0, min(w, len(rows) - 1))
                            r = rows[w]
                            f_ids_w.append(f_ids[r])
                            f_conf_w.append(f_conf[r])
                            d_ids_w.append(d_ids[r])
                            d_conf_w.append(d_conf[r])
                        else:
                            f_ids_w.append([0, 0, 0, 0, 0])
                            f_conf_w.append([0.0, 0.0, 0.0, 0.0, 0.0])
                            d_ids_w.append([0] * k)
                            d_conf_w.append([0.0] * k)
                    f_ids, f_conf, d_ids, d_conf = f_ids_w, f_conf_w, d_ids_w, d_conf_w
                B = len(seqs)
                step_wall = pr.current_step_elapsed_s()
                nd = st.num_draft_requests_step or B
                nv = st.num_verification_requests_step or B
                draft_row_s = (
                    float(st.draft_time_worker_s)
                    if pr.draft_async and st.draft_time_worker_s > 0.0
                    else float(st.draft_time_s)
                )
                rows = []
                for bi, seq in enumerate(seqs):
                    ch = None
                    if speculate_result.cache_hits is not None:
                        ch = int(speculate_result.cache_hits[bi].item())
                    rows.append(
                        trace_to_row_indexed(
                            profiler=pr,
                            seq=seq,
                            batch_index=bi,
                            batch_size=B,
                            is_prefill=False,
                            speculate_k=k,
                            spec_policy=pr.spec_policy,
                            draft_async=pr.draft_async,
                            cache_hit=ch,
                            trace=out_verify_result.profile_trace,
                            first_draft_token_ids=f_ids[bi],
                            first_draft_token_confidence=f_conf[bi],
                            draft_token_ids_per_position=d_ids[bi],
                            draft_token_confidence_per_position=d_conf[bi],
                            step_wall_time_s=step_wall,
                            draft_time_s=draft_row_s,
                            verification_time_s=st.verification_time_s,
                            sync_time_s=st.sync_time_s,
                            num_draft=nd,
                            num_verification=nv,
                            cost_fields=(pr.mode == "cost_metadata"),
                        )
                    )
                pr.flush_spec_decode_rows(seqs, False, rows)

        if _nvtx is not None:
            try:
                _nvtx.range_pop()
            except Exception:
                pass

        return sum(len(s) for s in out_verify_result.new_suffixes)


def _hv_rollback_committed_tape(seqs: list[Sequence], saved: list[tuple]) -> None:
    """Drop transient speculative ``token_ids`` tail; restore committed target tape fields.

    Draft KV logical frontier must stay aligned with HV provisional tape after each
    ``_hv_apply_local_intermediate_round`` (see ``prepare_decode_tensors_from_seqs``:
    ``num_draft_cached_tokens == len(seq) - 1 + hv_num_provisional_tokens`` when P>0).
    The step-entry snapshot's ``orig_ndc`` is only valid while ``hv_num_provisional_tokens == 0``
    (prefill matches full committed draft cache, not ``len-1``).
    """
    for seq, t in zip(seqs, saved):
        orig_len, orig_nt, orig_lt, orig_ndc, orig_nct = t
        del seq.token_ids[orig_len:]
        seq.num_tokens = orig_nt
        seq.last_token = orig_lt
        seq.num_cached_tokens = orig_nct
        if seq.hv_num_provisional_tokens > 0:
            seq.num_draft_cached_tokens = len(seq) - 1 + seq.hv_num_provisional_tokens
        else:
            seq.num_draft_cached_tokens = orig_ndc


def _hv_materialize_provisional_for_target_draft(
    seqs: list[Sequence],
) -> list[tuple[int, int, int, int, int]]:
    """Physically append HV provisional tail before fused target draft."""
    snap: list[tuple[int, int, int, int, int]] = []
    for seq in seqs:
        prov = list(seq.hv_provisional_token_ids)
        assert prov, (
            "target subround requires a non-empty provisional chain "
            f"(seq_id={seq.seq_id}, hv_round_idx={seq.hv_round_idx})"
        )
        assert seq.recovery_token_id is not None, (
            f"target subround requires recovery_token_id (seq_id={seq.seq_id})"
        )
        assert prov[-1] == seq.recovery_token_id, (
            "last provisional token must equal current recovery "
            f"(seq_id={seq.seq_id}, prov_tail={prov[-5:]}, recovery={seq.recovery_token_id})"
        )
        snap.append(
            (
                len(seq.token_ids),
                seq.num_tokens,
                int(seq.last_token),
                seq.num_draft_cached_tokens,
                seq.hv_num_provisional_tokens,
            )
        )
        seq.token_ids.extend(prov)
        seq.num_tokens += len(prov)
        seq.last_token = int(prov[-1])
        # Disable draft lazy path; run vanilla decode indexing for target subround.
        seq.hv_num_provisional_tokens = 0
        # Draft KV should be full context before the next token to emit.
        seq.num_draft_cached_tokens = seq.num_tokens - 1
    return snap


def _hv_unmaterialize_after_target_draft(
    seqs: list[Sequence],
    snap: list[tuple[int, int, int, int, int]],
) -> None:
    """Undo temporary physical provisional append before final target commit."""
    for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_hvp) in zip(seqs, snap):
        del seq.token_ids[orig_len:]
        seq.num_tokens = orig_nt
        seq.last_token = orig_lt
        seq.num_draft_cached_tokens = orig_ndc
        seq.hv_num_provisional_tokens = orig_hvp


class HierarchicalFusedStep(SpecDecodeStep):
    """One engine step runs ``r`` intermediate verifies + one target verify (fused hierarchical)."""

    def _hv_fused_flush_profiler_rows(
        self,
        pr: SSDProfiler,
        seqs: list[Sequence],
        speculate_result,
        verify_result,
        subround_idx: int,
        engine_step_id: int,
        draft_wall_s: float,
        verify_wall_s: float,
    ) -> None:
        if not pr.wants_metadata_computation():
            return
        st = pr.current_step_state
        if st is None or speculate_result.logits_q is None or speculate_result.logits_q.numel() == 0:
            return
        k = self.verifier.lookahead
        f_ids, f_conf, d_ids, d_conf = draft_metadata_from_logits(
            speculate_result.logits_q, speculate_result.speculations, k
        )
        B = len(seqs)
        step_wall = pr.current_step_elapsed_s()
        nd = st.num_draft_requests_step or B
        nv = st.num_verification_requests_step or B
        draft_row_s = (
            float(st.draft_time_worker_s)
            if pr.draft_async and st.draft_time_worker_s > 0.0
            else draft_wall_s
        )
        rows = []
        for bi, seq in enumerate(seqs):
            ch = None
            if speculate_result.cache_hits is not None:
                ch = int(speculate_result.cache_hits[bi].item())
            rows.append(
                trace_to_row_indexed(
                    profiler=pr,
                    seq=seq,
                    batch_index=bi,
                    batch_size=B,
                    is_prefill=False,
                    speculate_k=k,
                    spec_policy=pr.spec_policy,
                    draft_async=pr.draft_async,
                    cache_hit=ch,
                    trace=verify_result.profile_trace,
                    first_draft_token_ids=f_ids[bi],
                    first_draft_token_confidence=f_conf[bi],
                    draft_token_ids_per_position=d_ids[bi],
                    draft_token_confidence_per_position=d_conf[bi],
                    step_wall_time_s=step_wall,
                    draft_time_s=draft_row_s,
                    verification_time_s=verify_wall_s,
                    sync_time_s=st.sync_time_s,
                    num_draft=nd,
                    num_verification=nv,
                    cost_fields=(pr.mode == "cost_metadata"),
                    hv_fused_subround_idx=subround_idx,
                    hv_fused_engine_step_id=engine_step_id,
                )
            )
        pr.flush_spec_decode_rows(seqs, False, rows)

    def decode(self, seqs: list[Sequence]) -> int:
        if hasattr(self.speculator, "reset_step_state"):
            self.speculator.reset_step_state()
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _t_outer0 = perf_counter()

        verifier = self.verifier
        assert isinstance(verifier, VerifierHierarchical), "HierarchicalFusedStep requires VerifierHierarchical"
        assert not self.eagle and not self.async_spec
        assert self.scheduler.config.hv_ignore_intermediate_eos, (
            "HierarchicalFusedStep requires Config.hv_ignore_intermediate_eos=True "
            "(intermediate EOS must not jump hv_round_idx to r per-seq; avoids mixed-round batches)."
        )

        pr = self.profiler
        _nvtx = None
        if self._profiler_active() and pr.wants_kernel():
            try:
                import torch.cuda.nvtx as _nvtx_mod

                _nvtx = _nvtx_mod
                _nvtx.range_push("spec_decode")
            except Exception:
                _nvtx = None

        dbg_hv_pre_comp: list[int] | None = None
        dbg_hv_round0: list[int] | None = None
        if getattr(self.scheduler.config, "debug_mode", False):
            dbg_hv_pre_comp = [seq.num_completion_tokens for seq in seqs]
            dbg_hv_round0 = [seq.hv_round_idx for seq in seqs]

        saved = [
            (
                len(seq.token_ids),
                seq.num_tokens,
                seq.last_token,
                seq.num_draft_cached_tokens,
                seq.num_cached_tokens,
            )
            for seq in seqs
        ]

        r = verifier.r
        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )

        fused_engine_step_id = (
            int(pr.decode_metadata_step_id()) if self._profiler_active() else 0
        )

        for u in range(r):
            _hv_rollback_committed_tape(seqs, saved)
            if getattr(self.scheduler.config, "debug_mode", False):
                for seq in seqs:
                    print(
                        "[HV_BLOCK_DEBUG:fused_round_start] "
                        f"subround={u} "
                        f"seq_id={seq.seq_id} "
                        f"num_tokens={seq.num_tokens} "
                        f"num_cached_tokens={seq.num_cached_tokens} "
                        f"num_draft_cached_tokens={seq.num_draft_cached_tokens} "
                        f"num_inter_cached_tokens={seq.num_inter_cached_tokens} "
                        f"hv_num_provisional_tokens={seq.hv_num_provisional_tokens} "
                        f"hv_round_idx={seq.hv_round_idx} "
                        f"draft_blocks={len(seq.draft_block_table)} "
                        f"inter_blocks={len(seq.inter_block_table)} "
                        f"target_blocks={len(seq.block_table)}",
                        flush=True,
                    )
            if self._profiler_active():
                pr.bump_draft_requests(len(seqs))
                pr.start_stage("draft")
            t_d0 = perf_counter()
            speculate_result = self.speculator.speculate(seqs, in_verify_result)
            draft_wall_s = perf_counter() - t_d0
            if self._profiler_active():
                pr.finish_stage("draft")
            t_v0 = perf_counter()
            vr_inter = verifier.verify_intermediate_round(
                seqs, speculate_result, emit_step_metrics=False
            )
            verify_wall_s = perf_counter() - t_v0
            if self._profiler_active():
                pr.accum_hierarchical_verify_time(verify_wall_s, True)
                pr.record_decode_verify_batch(seqs, vr_inter)
                self._hv_fused_flush_profiler_rows(
                    pr,
                    seqs,
                    speculate_result,
                    vr_inter,
                    u,
                    fused_engine_step_id,
                    draft_wall_s,
                    verify_wall_s,
                )

            if (
                dbg_hv_pre_comp is not None
                and dbg_hv_round0 is not None
                and getattr(self.scheduler.config, "debug_mode", False)
            ):
                self._hv_correctness_debug_log(
                    seqs,
                    speculate_result,
                    vr_inter,
                    dbg_hv_pre_comp,
                    dbg_hv_round0,
                    None,
                    fused_subround_idx=u,
                )

            _hv_rollback_committed_tape(seqs, saved)
            self.scheduler._hv_apply_local_intermediate_round(
                seqs, vr_inter.new_suffixes, vr_inter.recovery_tokens
            )

        _hv_rollback_committed_tape(seqs, saved)
        if getattr(self.scheduler.config, "debug_mode", False):
            for seq in seqs:
                print(
                    "[HV_BLOCK_DEBUG:fused_round_start] "
                    f"subround=target "
                    f"seq_id={seq.seq_id} "
                    f"num_tokens={seq.num_tokens} "
                    f"num_cached_tokens={seq.num_cached_tokens} "
                    f"num_draft_cached_tokens={seq.num_draft_cached_tokens} "
                    f"num_inter_cached_tokens={seq.num_inter_cached_tokens} "
                    f"hv_num_provisional_tokens={seq.hv_num_provisional_tokens} "
                    f"hv_round_idx={seq.hv_round_idx} "
                    f"draft_blocks={len(seq.draft_block_table)} "
                    f"inter_blocks={len(seq.inter_block_table)} "
                    f"target_blocks={len(seq.block_table)}",
                    flush=True,
                )
        if self._profiler_active():
            pr.bump_draft_requests(len(seqs))
            pr.start_stage("draft")
        materialized = _hv_materialize_provisional_for_target_draft(seqs)
        t_d0 = perf_counter()
        speculate_result_tgt = self.speculator.speculate(
            seqs,
            in_verify_result,
            recovery_already_appended=True,
        )
        draft_wall_s = perf_counter() - t_d0
        if self._profiler_active():
            pr.finish_stage("draft")
        t_v0 = perf_counter()
        vr_tgt = verifier.verify_target_round(
            seqs, speculate_result_tgt, emit_step_metrics=True
        )
        verify_wall_s = perf_counter() - t_v0
        if self._profiler_active():
            pr.accum_hierarchical_verify_time(verify_wall_s, False)
            pr.record_decode_verify_batch(seqs, vr_tgt)
            self._hv_fused_flush_profiler_rows(
                pr,
                seqs,
                speculate_result_tgt,
                vr_tgt,
                r,
                fused_engine_step_id,
                draft_wall_s,
                verify_wall_s,
            )

        dbg_target_cands: list[list[int]] | None = None
        if getattr(self.scheduler.config, "debug_mode", False):
            dbg_target_cands = verifier._build_target_candidates(seqs, speculate_result_tgt)

        _hv_unmaterialize_after_target_draft(seqs, materialized)
        _hv_rollback_committed_tape(seqs, saved)

        if self._profiler_active():
            pr.start_stage("postprocess")
        self.scheduler.postprocess_hv_target_round(
            seqs,
            vr_tgt.new_suffixes,
            vr_tgt.recovery_tokens,
            eagle_acts=vr_tgt.eagle_acts if self.eagle else None,
        )
        if self._profiler_active():
            pr.finish_stage("postprocess")

        if (
            dbg_hv_pre_comp is not None
            and dbg_hv_round0 is not None
            and getattr(self.scheduler.config, "debug_mode", False)
        ):
            self._hv_correctness_debug_log(
                seqs,
                speculate_result_tgt,
                vr_tgt,
                dbg_hv_pre_comp,
                dbg_hv_round0,
                dbg_target_cands,
                fused_subround_idx=r,
            )

        if _prof:
            torch.cuda.synchronize()
            _t_outer1 = perf_counter()
            toks = sum(len(s) for s in vr_tgt.new_suffixes)
            print(
                f"[PROFILE target] fused_decode_total_ms={(_t_outer1-_t_outer0)*1000:.2f}ms toks={toks}",
                flush=True,
            )

        if _nvtx is not None:
            try:
                _nvtx.range_pop()
            except Exception:
                pass

        return sum(len(s) for s in vr_tgt.new_suffixes)
