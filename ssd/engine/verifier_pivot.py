import os
from time import perf_counter

import torch
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, VerifierBase, VerifyProfileTrace
from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.utils.verify import verify


class VerifierPivot(VerifierBase):
    """Pivot verifier with spec_hive intermediate gating.

    This keeps SSD's async target-verify call path unchanged, then chooses
    between intermediate-gated suffixes and target-verified suffixes.
    """

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer | None = None,
        metrics: dict | None = None,
        interval: int = 0,
        threshold: float = 0.8,
        expansion_pct: float = 1.0,
        enable_profile_trace: bool = False,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics if metrics is not None else {}

        self.interval = interval
        self.threshold = threshold
        self.expansion_pct = expansion_pct
        self.enable_profile_trace = enable_profile_trace
        self._intermediate_round_counters: dict[int, int] = {}

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        result = self.target_model_runner.call("run", seqs, True)
        if eagle:
            token_ids, eagle_acts = result
        else:
            token_ids = result
            eagle_acts = None

        offset = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
            self._intermediate_round_counters[seq.seq_id] = 0
            if eagle:
                seq_len = seq.num_prompt_tokens
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len

        return VerifyResult(
            [],
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
        )

    def _compute_topk_from_pct(self, vocab_size: int) -> int:
        raw = max(1, int(round(vocab_size * (self.expansion_pct / 100.0))))
        # Keep this bounded for stability and to mimic profile usage.
        return max(1, min(raw, 64))

    @staticmethod
    def _accept_lens_from_membership(match_mask: torch.Tensor) -> torch.Tensor:
        # match_mask: [B, K] where True means accepted at position
        B, K = match_mask.shape
        mismatch = ~match_mask
        any_mismatch = mismatch.any(dim=1)
        first_mismatch = mismatch.int().argmax(dim=1)
        return torch.where(any_mismatch, first_mismatch, torch.full_like(first_mismatch, K))

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        batch_size = len(seqs)
        if _prof:
            torch.cuda.synchronize()
            vt0 = perf_counter()

        target_start = perf_counter()
        result = self.target_model_runner.call("run", seqs, False, False, True)

        if eagle:
            logits_p_flat, eagle_acts_flat = result
        else:
            logits_p_flat = result
            eagle_acts_flat = None

        for seq in seqs:
            seq.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.view(batch_size, self.lookahead + 1, -1)
        logits_q = speculate_result.logits_q
        speculations = speculate_result.speculations
        assert logits_q.shape[0] == batch_size and logits_q.shape[1] == self.lookahead, (
            f"Unexpected logits_q shape={tuple(logits_q.shape)} for batch={batch_size}, K={self.lookahead}"
        )
        assert speculations.shape[0] == batch_size and speculations.shape[1] == self.lookahead + 1, (
            f"Unexpected speculations shape={tuple(speculations.shape)} for batch={batch_size}, K={self.lookahead}"
        )

        temps_target = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, device=self.device)
        temps_draft = torch.tensor(
            [seq.draft_temperature if seq.draft_temperature is not None else seq.temperature for seq in seqs],
            dtype=torch.float32,
            device=self.device,
        )

        target_suffixes, target_recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=logits_q,
            speculations=speculations,
            temperatures_target=temps_target,
            temperatures_draft=temps_draft,
            cache_hits=speculate_result.cache_hits,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
        )

        # ----- spec_hive intermediate gating -----
        draft_tokens = speculations[:, 1:]  # [B, K]
        probs_q = torch.softmax(logits_q.float(), dim=-1)  # [B, K, V]
        top1_conf = probs_q.max(dim=-1).values  # [B, K]
        cumprod_conf = torch.cumprod(top1_conf, dim=1)  # [B, K]
        threshold_cross = (cumprod_conf < self.threshold).any(dim=1)  # [B]

        vocab_size = logits_q.shape[-1]
        topk_expand = self._compute_topk_from_pct(vocab_size)
        topk_idx = torch.topk(logits_q, k=topk_expand, dim=-1).indices  # [B, K, topk]
        inter_match = (draft_tokens.unsqueeze(-1) == topk_idx).any(dim=-1)  # [B, K]
        inter_accept_lens = self._accept_lens_from_membership(inter_match)  # [B]

        starts = speculations[:, 0].tolist()
        inter_suffixes: list[list[int]] = []
        for b in range(batch_size):
            n = int(inter_accept_lens[b].item())
            suffix = [starts[b]] + draft_tokens[b, :n].tolist()
            inter_suffixes.append(suffix)

        # Reuse target recovery token to preserve downstream sampling behavior.
        final_suffixes: list[list[int]] = []
        final_recovery: list[int] = []
        forced_target_rounds = 0
        intermediate_rounds = 0
        target_rounds = 0
        use_target_flags: list[bool] = []

        for i, seq in enumerate(seqs):
            inter_rounds = self._intermediate_round_counters.get(seq.seq_id, 0)
            forced_target = self.interval > 0 and inter_rounds >= self.interval
            use_target = forced_target or bool(threshold_cross[i].item())
            use_target_flags.append(use_target)

            if use_target:
                final_suffixes.append(target_suffixes[i])
                final_recovery.append(target_recovery_tokens[i])
                self._intermediate_round_counters[seq.seq_id] = 0
                target_rounds += 1
                if forced_target:
                    forced_target_rounds += 1
            else:
                final_suffixes.append(inter_suffixes[i])
                final_recovery.append(target_recovery_tokens[i])
                self._intermediate_round_counters[seq.seq_id] = inter_rounds + 1
                intermediate_rounds += 1

        for suffix in final_suffixes:
            # Scheduler expects each accepted suffix to include recovery and be non-empty.
            assert len(suffix) >= 1, "pivot verifier produced empty suffix"
            assert len(suffix) <= self.lookahead + 1, (
                f"pivot verifier produced suffix length {len(suffix)} > K+1 ({self.lookahead + 1})"
            )

        self.metrics.setdefault("target_verify_times", []).append(perf_counter() - target_start)
        self.metrics["pivot_intermediate_rounds"] = self.metrics.get("pivot_intermediate_rounds", 0) + intermediate_rounds
        self.metrics["pivot_target_rounds"] = self.metrics.get("pivot_target_rounds", 0) + target_rounds
        self.metrics["pivot_forced_target_rounds"] = self.metrics.get("pivot_forced_target_rounds", 0) + forced_target_rounds
        self.metrics.setdefault("accepted_suffix_lens_with_recovery", []).extend([len(s) for s in final_suffixes])

        if speculate_result.cache_hits is not None:
            cache_hits = speculate_result.cache_hits.cpu()
            self.metrics.setdefault("cache_hits", []).append(cache_hits.float().mean().item())
            for i, suffix_len in enumerate([len(s) for s in final_suffixes]):
                if cache_hits[i] == 1:
                    self.metrics.setdefault("accepted_suffix_lens_on_hit", []).append(suffix_len)
                else:
                    self.metrics.setdefault("accepted_suffix_lens_on_miss", []).append(suffix_len)

        if _prof:
            torch.cuda.synchronize()
            vt1 = perf_counter()
            print(f"[PROFILE verify pivot] total={((vt1-vt0)*1000):.2f}ms inter_rounds={intermediate_rounds} target_rounds={target_rounds}", flush=True)

        eagle_acts = None
        if eagle and eagle_acts_flat is not None:
            eagle_acts = eagle_acts_flat.view(batch_size, self.lookahead + 1, -1)

        profile_trace: VerifyProfileTrace | None = None
        if self.enable_profile_trace and not eagle:
            from ssd.utils.verify import target_probs_p_batched

            probs_p = target_probs_p_batched(logits_p, temps_target)
            pred_ids = probs_p.argmax(dim=-1).cpu().tolist()
            pred_conf = probs_p.max(dim=-1).values.cpu().tolist()
            tgt_tok_ids = [[int(pred_ids[b][j]) for j in range(self.lookahead + 1)] for b in range(batch_size)]
            tgt_tok_conf = [[float(pred_conf[b][j]) for j in range(self.lookahead + 1)] for b in range(batch_size)]
            tgt_bonus = [int(logits_p[b, self.lookahead, :].argmax().item()) for b in range(batch_size)]
            K = self.lookahead
            v_models: list[str] = []
            inter_ids_list: list[list[int] | None] = []
            inter_conf_list: list[list[float] | None] = []
            inter_accept: list[int | None] = []
            inter_rec: list[int | None] = []
            inter_bonus: list[int | None] = []
            for i in range(batch_size):
                use_target = use_target_flags[i]
                if use_target:
                    v_models.append("pivot_target")
                    inter_ids_list.append(None)
                    inter_conf_list.append(None)
                    inter_accept.append(None)
                    inter_rec.append(None)
                    inter_bonus.append(None)
                else:
                    v_models.append("pivot_intermediate")
                    n = int(inter_accept_lens[i].item())
                    inter_ids_list.append([int(draft_tokens[i, j].item()) for j in range(K)])
                    inter_conf_list.append(
                        [float(probs_q[i, j, draft_tokens[i, j]].item()) for j in range(K)]
                    )
                    inter_accept.append(n)
                    inter_rec.append(int(target_recovery_tokens[i]))
                    inter_bonus.append(int(logits_p[i, K, :].argmax().item()))
            profile_trace = VerifyProfileTrace(
                verification_models=v_models,
                token_ids_per_position=tgt_tok_ids,
                token_confidence_per_position=tgt_tok_conf,
                accept_len=[max(0, len(final_suffixes[b]) - 1) for b in range(batch_size)],
                recovery_tokens=list(final_recovery),
                bonus_tokens=tgt_bonus,
                inter_token_ids_per_position=inter_ids_list,
                inter_token_confidence_per_position=inter_conf_list,
                inter_accept_len=inter_accept,
                inter_recovery_token=inter_rec,
                inter_bonus_token=inter_bonus,
            )

        return VerifyResult(
            new_suffixes=final_suffixes,
            recovery_tokens=final_recovery,
            eagle_acts=eagle_acts,
            profile_trace=profile_trace,
        )
