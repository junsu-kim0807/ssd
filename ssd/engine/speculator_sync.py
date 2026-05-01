import torch

from ssd.engine.sequence import Sequence
from ssd.engine.model_runner import ModelRunner
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase


class SpeculatorSync(SpeculatorBase):

    def __init__(self, lookahead: int, device: torch.device, draft_model_runner: ModelRunner):
        super().__init__(lookahead, device)
        self.draft_model_runner = draft_model_runner

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        assert verify_result.eagle_acts is None, (
            "Eagle is not currently supported for synchronous speculation"
        )
        print('[spec_prefill] target prefill', flush=True)
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
        assert verify_result.eagle_acts is None, (
            "Eagle is not currently supported for synchronous speculation"
        )

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

        for k in range(self.lookahead + 1):
            # Draft model forward pass - emits [B] tokens, True is for draft_return_logits
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
