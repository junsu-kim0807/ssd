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
        assert self.r >= 1, "hierarchical verification requires target_verify_interval >= 1"

    def _is_target_round(self, seq: Sequence) -> bool:
        """Target verify on the step where ``hv_round_idx == r`` (``r`` intermediate indices 0..r-1)."""
        return seq.hv_round_idx == self.r

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
            # Intermediate KV must match the same committed prefix as target after prefill
            # (including tokens after the original prompt when re-prefilling after preempt).
            seq.num_inter_cached_tokens = seq.num_tokens
        return VerifyResult([], [seq.recovery_token_id for seq in seqs], None, is_hv_intermediate=False)

    def verify_intermediate_round(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
        *,
        emit_step_metrics: bool = True,
    ) -> VerifyResult:
        K = self.lookahead
        batch_size = len(seqs)
        prev_nics = [seq.num_inter_cached_tokens for seq in seqs]

        # Packed varlen: optional ``token_ids[nic:c0)`` gap rows + ``(K+1)`` scored tail.
        # ``q_lens`` are packed per-sequence lengths (CUDAGraph bucket padding); scored rows are
        # always ``logits_i[score_starts[b] : score_starts[b] + K + 1]``.
        logits_flat, score_starts, q_lens = self.intermediate_runner.call(
            "run_intermediate_verify_suffix", seqs, K
        )

        draft_tokens = speculate_result.speculations[:, 1:].to(self.device)
        starts = speculate_result.speculations[:, 0]

        offset = 0
        new_suffixes: list[list[int]] = []
        recovery_tokens: list[int] = []
        accept_n_list: list[int] = []
        profile_scored: list[torch.Tensor] | None = [] if self.enable_profile_trace else None

        for b in range(batch_size):
            ss = score_starts[b]
            ql = q_lens[b]
            logits_i = logits_flat[offset : offset + ql]
            offset += ql
            scored = logits_i[ss : ss + K + 1]
            preds = scored.argmax(dim=-1)
            if profile_scored is not None:
                profile_scored.append(scored.detach())
            dt = draft_tokens[b]
            preds_draft = preds[:-1]
            matches = dt == preds_draft
            mismatch = ~matches
            any_m = mismatch.any()
            first_m = mismatch.int().argmax()
            n = int(first_m.item()) if bool(any_m.item()) else K
            accept_n_list.append(n)
            recovery_from_inter = preds[n]
            new_suffixes.append([int(starts[b].item())] + dt[:n].tolist())
            recovery_tokens.append(int(recovery_from_inter.item()))

        # Logical frontier: scored window begins at ``prev_nic + score_start`` (gap rows only
        # shift ``score_start``). Do not use ``num_cached_tokens`` here — with provisional
        # tape ``prev_nic`` can exceed ``c0``, and ``c0 + n + 1`` would shrink ``nic`` wrongly.
        for b, (seq, n) in enumerate(zip(seqs, accept_n_list)):
            seq.num_inter_cached_tokens = prev_nics[b] + score_starts[b] + int(n) + 1

        profile_trace = None
        if self.enable_profile_trace and profile_scored is not None:
            inter_token_ids_per_position: list[list[int]] = []
            inter_token_confidence_per_position: list[list[float]] = []
            token_ids_per_position: list[list[int]] = []
            token_confidence_per_position: list[list[float]] = []
            bonus_tokens: list[int] = []
            for scored in profile_scored:
                pr = torch.softmax(scored.float(), dim=-1)
                preds_row = scored.argmax(dim=-1)
                token_ids_per_position.append([int(preds_row[j].item()) for j in range(K + 1)])
                token_confidence_per_position.append([float(pr[j].max().item()) for j in range(K + 1)])
                probs_head = torch.softmax(scored[:-1, :].float(), dim=-1)
                inter_token_ids_per_position.append([int(preds_row[j].item()) for j in range(K)])
                inter_token_confidence_per_position.append(
                    [float(probs_head[j].max().item()) for j in range(K)]
                )
                bonus_tokens.append(int(preds_row[K].item()))
            profile_trace = VerifyProfileTrace(
                verification_models=["intermediate"] * batch_size,
                token_ids_per_position=token_ids_per_position,
                token_confidence_per_position=token_confidence_per_position,
                accept_len=[0] * batch_size,
                recovery_tokens=list(recovery_tokens),
                bonus_tokens=bonus_tokens,
                inter_token_ids_per_position=inter_token_ids_per_position,
                inter_token_confidence_per_position=inter_token_confidence_per_position,
                inter_accept_len=[int(accept_n_list[b]) for b in range(batch_size)],
                inter_recovery_token=[recovery_tokens[b] for b in range(batch_size)],
                inter_bonus_token=bonus_tokens,
            )

        vr = VerifyResult(
            new_suffixes, recovery_tokens, None, is_hv_intermediate=True, profile_trace=profile_trace
        )
        lens = [len(s) for s in vr.new_suffixes]
        if emit_step_metrics:
            self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend(lens)
        else:
            self.metrics.setdefault("hv_fused_intermediate_suffix_lens", []).extend(lens)
        return vr

    def _build_target_candidates(self, seqs: list[Sequence], speculate_result: SpeculateResult) -> list[list[int]]:
        out: list[list[int]] = []
        for i, seq in enumerate(seqs):
            parts: list[int] = []
            # Provisional list ends with intermediate recovery (merged in scheduler); then draft spec row.
            parts.extend(seq.hv_provisional_token_ids)
            spec_row = speculate_result.speculations[i].tolist()
            if spec_row and parts and spec_row[0] == parts[-1]:
                parts.extend(spec_row[1:])
            else:
                parts.extend(spec_row)
            out.append(parts)
        return out

    def verify_target_round(
        self,
        seqs: list[Sequence],
        speculate_result: SpeculateResult,
        *,
        emit_step_metrics: bool = True,
    ) -> VerifyResult:
        candidates = self._build_target_candidates(seqs, speculate_result)
        # prepare_verify_tensors_varlen uses seq.num_cached_tokens as the KV frontier.
        # Do not bump num_cached_tokens before the call (that would shift positions/slots).
        logits_ret = self.target_model_runner.call("run_verify_varlen", seqs, candidates)
        if isinstance(logits_ret, tuple):
            logits_flat, q_strides = logits_ret
        else:
            logits_flat = logits_ret
            q_strides = [len(candidates[i]) for i in range(len(seqs))]
        if logits_flat is not None and logits_flat.dim() == 3:
            logits_flat = logits_flat.reshape(-1, logits_flat.size(-1))

        new_suffixes: list[list[int]] = []
        recovery_tokens: list[int] = []
        offset = 0
        tok_ids: list[list[int]] = []
        tok_conf: list[list[float]] = []
        acc_lens: list[int] = []
        bonus_toks: list[int] = []
        inter_target_prefix_accepts: list[int] = []
        K = self.lookahead
        for i, _seq in enumerate(seqs):
            L = len(candidates[i])
            s = q_strides[i]
            logits_i = logits_flat[offset : offset + s][:L]
            offset += s
            cand = candidates[i]
            suffix, rec = verify_greedy_chain_variable(logits_i, cand)
            new_suffixes.append(suffix)
            recovery_tokens.append(rec)
            if self.enable_profile_trace:
                pr = torch.softmax(logits_i.float(), dim=-1)
                preds_row = logits_i.argmax(dim=-1)
                tok_ids.append([int(preds_row[j].item()) for j in range(L)])
                tok_conf.append([float(pr[j].max().item()) for j in range(L)])
                acc_lens.append(max(0, len(suffix) - 1))
                bonus_toks.append(int(preds_row[-1].item()))
                cap_excl = L - K
                itp = 0
                if cap_excl > 0:
                    for j in range(0, L - 1):
                        if j + 1 >= cap_excl:
                            break
                        if cand[j + 1] != int(preds_row[j].item()):
                            break
                        itp += 1
                inter_target_prefix_accepts.append(itp)

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
                inter_target_prefix_accept_len=inter_target_prefix_accepts,
            )

        vr = VerifyResult(
            new_suffixes, recovery_tokens, None, is_hv_intermediate=False, profile_trace=profile_trace
        )
        if emit_step_metrics:
            self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend(
                [len(s) for s in vr.new_suffixes]
            )
        return vr

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        assert not eagle
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            t0 = perf_counter()

        if all(self._is_target_round(s) for s in seqs):
            vr = self.verify_target_round(seqs, speculate_result, emit_step_metrics=True)
        else:
            assert all(not self._is_target_round(s) for s in seqs), "mixed target/intermediate rounds in one batch unsupported"
            vr = self.verify_intermediate_round(seqs, speculate_result, emit_step_metrics=True)

        if _prof:
            torch.cuda.synchronize()
            print(f"[PROFILE verify hierarchical] {(perf_counter()-t0)*1000:.2f}ms inter={vr.is_hv_intermediate}", flush=True)

        return vr
