from abc import ABC, abstractmethod
import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
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

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if __debug__:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

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

    def _profiler_active(self) -> bool:
        return isinstance(self.profiler, SSDProfiler)

    def prefill(self, seqs: list[Sequence]) -> int:
        # Prefill timing: only the outer engine step wall (LLMEngine start_step/finish_step).
        # No inner draft_prefill/target_prefill stages and no per-request JSONL rows — all profiler modes.

        if not self.eagle and self.async_spec:
            empty_verify_result = VerifyResult([], [], None)
            self.speculator.prefill(seqs, empty_verify_result)
            verify_result = self.verifier.prefill(seqs, eagle=False)
        else:
            verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
            self.speculator.prefill(seqs, verify_result)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens
            if getattr(self.scheduler, "hierarchical", False):
                seq.num_inter_cached_tokens = seq.num_prompt_tokens

        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
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
        if hierarchical:
            saved = [
                (
                    len(seq.token_ids),
                    seq.num_tokens,
                    seq.last_token,
                    seq.num_draft_cached_tokens,
                    seq.num_cached_tokens,
                    seq.num_inter_cached_tokens,
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
        #### STEP 1: SPECULATE ####
        if self._profiler_active():
            pr.bump_draft_requests(len(seqs))
            pr.start_stage("draft")
        speculate_result = self.speculator.speculate(seqs, in_verify_result)
        if self._profiler_active():
            pr.finish_stage("draft")

        if _prof:
            torch.cuda.synchronize()
            _t1 = perf_counter()

        if __debug__:
            speculations = speculate_result.speculations
            print(f"[SpecDecodeStep] speculations: {speculations}", flush=True)
            speculations_list = speculations.tolist()

            for i, speculation in enumerate(speculations_list):
                decoded_tokens = decode_tokens(speculation, self.tokenizer)
                print(f"[SpecDecodeStep] speculation {i}: {decoded_tokens}", flush=True)

        #### STEP 2: VERIFY ####
        if self._profiler_active():
            pr.start_stage("verify")
        out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)
        if self._profiler_active():
            pr.finish_stage("verify")
            pr.record_decode_verify_batch(seqs, out_verify_result)

        if _prof:
            torch.cuda.synchronize()
            _t2 = perf_counter()

        if __debug__:
            recovery_tokens = out_verify_result.recovery_tokens
            new_suffixes = out_verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f"[SpecDecodeStep] verification {i}: {decoded_tokens}", flush=True)

        # Restore original seq state before postprocess (undo speculate + verify modifications)
        if hierarchical:
            for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct, orig_nic) in zip(seqs, saved):
                del seq.token_ids[orig_len:]
                seq.num_tokens = orig_nt
                seq.last_token = orig_lt
                seq.num_draft_cached_tokens = orig_ndc
                seq.num_cached_tokens = orig_nct
                seq.num_inter_cached_tokens = orig_nic
        else:
            for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
                del seq.token_ids[orig_len:]
                seq.num_tokens = orig_nt
                seq.last_token = orig_lt
                seq.num_draft_cached_tokens = orig_ndc
                seq.num_cached_tokens = orig_nct

        #### STEP 3: POSTPROCESS ####
        if self._profiler_active():
            pr.start_stage("postprocess")
        if hierarchical:
            if out_verify_result.is_hv_intermediate:
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
            self.scheduler.postprocess_speculate(
                seqs,
                out_verify_result.new_suffixes,
                out_verify_result.recovery_tokens,
                eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
            )
        if self._profiler_active():
            pr.finish_stage("postprocess")

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
