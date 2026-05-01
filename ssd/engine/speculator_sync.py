import torch

from ssd.engine.sequence import Sequence
from ssd.engine.model_runner import ModelRunner
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase


class SpeculatorSync(SpeculatorBase):

    def __init__(self, lookahead: int, device: torch.device, draft_model_runner: ModelRunner):
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner
        self.use_eagle = bool(getattr(draft_model_runner.config, "use_eagle", False))

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        print('[spec_prefill] target prefill', flush=True)

        if self.use_eagle:
            eagle_acts = verify_result.eagle_acts
            assert eagle_acts is not None and not isinstance(eagle_acts, bool), (
                "Eagle sync prefill requires verifier to populate eagle_acts"
            )
            # j ↔ j-1 alignment: token at position j is conditioned on target act at j-1.
            # Stride must match the actual emit count from target prefill (which obeys
            # the prefix-cache fallback in prepare_prefill_tensors_from_seqs); using
            # raw len(seq) corrupts cross-seq offsets when any seq had a prefix hit.
            # NOTE: EAGLE + prefix cache is not fully supported — draft prefill below
            # still uses skip_first_token=1 from token 0, so non-zero num_cached_tokens
            # leaves draft conditioning gaps in [0..num_cached_tokens-1]. Asserting for
            # now so the failure mode is loud rather than silently degraded.
            sliced = []
            offset = 0
            for seq in seqs:
                emit_len = max(1, len(seq) - seq.num_cached_tokens)
                assert seq.num_cached_tokens == 0, (
                    "EAGLE sync prefill currently requires num_cached_tokens==0 "
                    f"(seq_id={seq.seq_id}, cached={seq.num_cached_tokens}, len={len(seq)}). "
                    "Prefix cache + EAGLE conditioning alignment is not implemented."
                )
                sliced.append(eagle_acts[offset:offset + emit_len - 1])
                offset += emit_len
            eagle_acts_shifted = torch.cat(sliced, dim=0).to(self.device)
            self.draft_model_runner.call(
                "run", seqs, True, True, False, eagle_acts_shifted, 1,
            )
        else:
            self.draft_model_runner.call("run", seqs, True)

        if len(seqs) > 0:
            print(
                f"[PREFILL] seq0 prompt_len={seqs[0].num_prompt_tokens} recovery={seqs[0].recovery_token_id}", flush=True)

        return SpeculateResult([], [])

    def speculate(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
        *,
        recovery_already_appended: bool = False,
    ) -> SpeculateResult:
        """Generate k speculative tokens using the draft model."""
        batch_size = len(seqs)
        speculations = torch.zeros(
            batch_size, self.lookahead + 1,
            dtype=torch.int64,
            device=self.device,
        )
        logits_q = []

        # Single batched write to GPU
        recovery_tokens = []
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            recovery_tokens.append(seq.recovery_token_id)
            if recovery_already_appended:
                assert seq.last_token == seq.recovery_token_id, (
                    "recovery_already_appended=True requires seq.last_token == seq.recovery_token_id "
                    f"(seq_id={seq.seq_id}, last_token={seq.last_token}, "
                    f"recovery_token_id={seq.recovery_token_id})"
                )
            else:
                # HV lazy mode: intermediate recovery may already be represented by the
                # provisional tail; avoid duplicating it on token_ids before first draft fwd.
                prov = seq.hv_provisional_token_ids
                skip_append = len(prov) > 0 and prov[-1] == seq.recovery_token_id
                if skip_append:
                    continue
                seq.append_token(seq.recovery_token_id)
        speculations[:, 0] = torch.tensor(
            recovery_tokens, dtype=torch.int64, device=self.device)

        # Eagle: stack target conditioning for each seq. The draft model's forward
        # auto-applies fc on the first step (3*d_model_target -> d_model_draft) and
        # passes through prenorm on subsequent steps.
        conditioning = None
        if self.use_eagle:
            for seq in seqs:
                assert seq.last_target_hidden_state is not None, (
                    "Eagle sync speculate requires seq.last_target_hidden_state "
                    "(set by Verifier.prefill / scheduler post-verify)"
                )
            conditioning = torch.stack(
                [seq.last_target_hidden_state for seq in seqs], dim=0,
            ).to(self.device)

        for k in range(self.lookahead + 1):
            # Draft model forward pass - emits [B] tokens, True is for draft_return_logits
            if self.use_eagle:
                token_ids, step_logits_q, conditioning = self.draft_model_runner.call(
                    "run", seqs, False, True, True, conditioning,
                )
            else:
                token_ids, step_logits_q = self.draft_model_runner.call(
                    "run", seqs, False, True, True)
            # make sure we include this even on last iter since we put K+1 tokens thru draft cache
            for s in seqs:
                s.num_draft_cached_tokens += 1

            if k == self.lookahead:
                break  # this extra fwd also

            logits_q.append(step_logits_q)

            for i, (seq, token_id) in enumerate(zip(seqs, token_ids)):
                seq.append_token(token_id)

            # Single batched write to GPU
            speculations[:, k + 1] = torch.tensor(
                token_ids, dtype=torch.int64, device=self.device)

        logits_q = torch.stack(logits_q, dim=1)  # [B, K, V]

        return SpeculateResult(speculations, logits_q)
