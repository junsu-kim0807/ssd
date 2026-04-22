"""Hierarchical verification: one verify per step (intermediate XOR target)."""

import os
from time import perf_counter

import torch
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import (
    SpeculateResult,
    VerifyResult,
    VerifierBase,
    VerifyProfileTrace,
)
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.utils.verify import verify_greedy_chain_variable


class VerifierHierarchical(VerifierBase):
    """Sync hierarchical policy: draft speculate always; verify is intermediate or target by round."""

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        intermediate_runner: ModelRunner,
        target_verify_interval: int,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer | None = None,
        metrics: dict | None = None,
        enable_profile_trace: bool = False,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.intermediate_runner = intermediate_runner
        self.r = target_verify_interval
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics if metrics is not None else {}
        self.enable_profile_trace = enable_profile_trace
        assert self.r >= 2, "hierarchical verification requires target_verify_interval >= 2 (need r-1 >= 1 intermediate rounds)"

    def _is_target_round(self, seq: Sequence) -> bool:
        return seq.hv_round_idx == self.r - 1

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        assert not eagle, "hierarchical verifier does not support EAGLE"
        result = self.target_model_runner.call("run", seqs, True)
        for seq, token_id in zip(seqs, result):
            seq.recovery_token_id = token_id
            seq.hv_round_idx = 0
            seq.hv_provisional_token_ids.clear()
            seq.hv_provisional_recovery_token_id = None
            seq.hv_num_provisional_tokens = 0
        self.intermediate_runner.call("intermediate_run", seqs, True)
        for seq in seqs:
            seq.num_inter_cached_tokens = seq.num_prompt_tokens
        return VerifyResult([], [seq.recovery_token_id for seq in seqs], None, is_hv_intermediate=False)

    def _verify_intermediate_round(
        self, seqs: list[Sequence], speculate_result: SpeculateResult
    ) -> VerifyResult:
        K = self.lookahead
        batch_size = len(seqs)

        # One autoregressive forward over the full speculative tail (same tokens as draft).
        # ``num_inter_cached_tokens`` is authoritative (restored after prior steps); do not
        # reset to ``num_cached_tokens`` or prior-round intermediate KV is discarded.
        # Use ``call`` so TP follower ranks run the same forward (``intermediate_run`` already does).
        logits_flat = self.intermediate_runner.call("run_intermediate_verify_suffix", seqs, K)
        for seq in seqs:
            seq.num_inter_cached_tokens += K + 1

        logits_inter = logits_flat.view(batch_size, K + 1, -1)
        preds = logits_inter.argmax(dim=-1)
        draft_tokens = speculate_result.speculations[:, 1:]
        preds_draft = preds[:, :-1]
        matches = draft_tokens == preds_draft
        mismatch = ~matches
        any_m = mismatch.any(dim=1)
        first_m = mismatch.int().argmax(dim=1)
        accept_n = torch.where(any_m, first_m, torch.full_like(first_m, K))
        batch_idx = torch.arange(batch_size, device=self.device)
        recovery_from_inter = preds[batch_idx, accept_n]

        new_suffixes: list[list[int]] = []
        recovery_tokens: list[int] = []
        starts = speculate_result.speculations[:, 0].tolist()
        for b in range(batch_size):
            n = int(accept_n[b].item())
            suffix = [starts[b]] + draft_tokens[b, :n].tolist()
            new_suffixes.append(suffix)
            recovery_tokens.append(int(recovery_from_inter[b].item()))

        profile_trace = None
        if self.enable_profile_trace:
            probs_head = torch.softmax(logits_inter[:, :-1, :].float(), dim=-1)
            inter_conf = probs_head.max(dim=-1).values.cpu().tolist()
            inter_ids = preds[:, :-1].cpu().tolist()
            probs_full = torch.softmax(logits_inter.float(), dim=-1)
            row_conf = probs_full.max(dim=-1).values.cpu().tolist()
            pred_rows = preds.cpu().tolist()
            profile_trace = VerifyProfileTrace(
                verification_models=["intermediate"] * batch_size,
                token_ids_per_position=[
                    [int(pred_rows[b][j]) for j in range(K + 1)] for b in range(batch_size)
                ],
                token_confidence_per_position=[
                    [float(row_conf[b][j]) for j in range(K + 1)] for b in range(batch_size)
                ],
                accept_len=[0] * batch_size,
                recovery_tokens=list(recovery_tokens),
                bonus_tokens=[int(preds[b, K].item()) for b in range(batch_size)],
                inter_token_ids_per_position=[[int(inter_ids[b][j]) for j in range(K)] for b in range(batch_size)],
                inter_token_confidence_per_position=[
                    [float(inter_conf[b][j]) for j in range(K)] for b in range(batch_size)
                ],
                inter_accept_len=[int(accept_n[b].item()) for b in range(batch_size)],
                inter_recovery_token=[int(recovery_from_inter[b].item()) for b in range(batch_size)],
                inter_bonus_token=[int(preds[b, K].item()) for b in range(batch_size)],
            )

        return VerifyResult(
            new_suffixes, recovery_tokens, None, is_hv_intermediate=True, profile_trace=profile_trace
        )

    def _build_target_candidates(self, seqs: list[Sequence], speculate_result: SpeculateResult) -> list[list[int]]:
        out: list[list[int]] = []
        for i, seq in enumerate(seqs):
            parts: list[int] = []
            if seq.hv_provisional_recovery_token_id is not None:
                parts.append(seq.hv_provisional_recovery_token_id)
            parts.extend(seq.hv_provisional_token_ids)
            spec_row = speculate_result.speculations[i].tolist()
            # spec_row[0] is next-step recovery; intermediate postprocess sets it equal to
            # ``hv_provisional_recovery_token_id``, so including both duplicates the token.
            if (
                spec_row
                and seq.hv_provisional_recovery_token_id is not None
                and spec_row[0] == seq.hv_provisional_recovery_token_id
            ):
                parts.extend(spec_row[1:])
            else:
                parts.extend(spec_row)
            out.append(parts)
        return out

    def _verify_target_round(
        self, seqs: list[Sequence], speculate_result: SpeculateResult
    ) -> VerifyResult:
        candidates = self._build_target_candidates(seqs, speculate_result)
        # prepare_verify_tensors_varlen uses seq.num_cached_tokens as the KV frontier.
        # Do not bump num_cached_tokens before the call (that would shift positions/slots).
        logits_flat = self.target_model_runner.call("run_verify_varlen", seqs, candidates)

        new_suffixes: list[list[int]] = []
        recovery_tokens: list[int] = []
        offset = 0
        tok_ids: list[list[int]] = []
        tok_conf: list[list[float]] = []
        acc_lens: list[int] = []
        bonus_toks: list[int] = []
        for i, seq in enumerate(seqs):
            L = len(candidates[i])
            logits_i = logits_flat[offset : offset + L]
            offset += L
            suffix, rec = verify_greedy_chain_variable(logits_i, candidates[i])
            new_suffixes.append(suffix)
            recovery_tokens.append(rec)
            if self.enable_profile_trace:
                pr = torch.softmax(logits_i.float(), dim=-1)
                preds_row = logits_i.argmax(dim=-1)
                tok_ids.append([int(preds_row[j].item()) for j in range(L)])
                tok_conf.append([float(pr[j].max().item()) for j in range(L)])
                acc_lens.append(max(0, len(suffix) - 1))
                bonus_toks.append(int(preds_row[-1].item()))

        profile_trace = None
        if self.enable_profile_trace:
            batch_size = len(seqs)
            profile_trace = VerifyProfileTrace(
                verification_models=["target"] * batch_size,
                token_ids_per_position=tok_ids,
                token_confidence_per_position=tok_conf,
                accept_len=acc_lens,
                recovery_tokens=list(recovery_tokens),
                bonus_tokens=bonus_toks,
            )

        return VerifyResult(
            new_suffixes, recovery_tokens, None, is_hv_intermediate=False, profile_trace=profile_trace
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        assert not eagle
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            t0 = perf_counter()

        if all(self._is_target_round(s) for s in seqs):
            vr = self._verify_target_round(seqs, speculate_result)
        else:
            assert all(not self._is_target_round(s) for s in seqs), "mixed target/intermediate rounds in one batch unsupported"
            vr = self._verify_intermediate_round(seqs, speculate_result)

        if _prof:
            torch.cuda.synchronize()
            print(f"[PROFILE verify hierarchical] {(perf_counter()-t0)*1000:.2f}ms inter={vr.is_hv_intermediate}", flush=True)

        self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend([len(s) for s in vr.new_suffixes])
        return vr
