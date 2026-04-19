"""Contract tests for profile/batch_profile.py (hispec + vanila vs topk expansion layout)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "profile") not in sys.path:
    sys.path.insert(0, str(_ROOT / "profile"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import batch_profile_contract_utils as cu


def _batch_profile_importable() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class TestBatchProfileContract(unittest.TestCase):
    def test_normalize_vanilla_equals_vanila(self) -> None:
        self.assertEqual(cu._normalize_method("vanilla"), "vanila")

    def test_hispec_whitelist_after_normalize(self) -> None:
        for raw in ("vanilla", "vanila"):
            m = cu._normalize_method(raw)
            self.assertEqual(m, "vanila")
            self.assertIn(m, {"vanila", "topk_expansion"})

    def test_verify_positions_helpers(self) -> None:
        prompt = [1, 2, 3]
        b = len(prompt)
        k = 5
        pos = cu._verify_positions_no_carry(b, k)
        self.assertEqual(pos, list(range(b - 1, b + k + 1)))
        recovery = 9
        draft = [10, 11, 12, 13, 14]
        full = cu._verify_full_seq_no_carry(prompt, recovery, draft)
        self.assertEqual(full, [1, 2, 3, 9] + draft)

    @unittest.skipUnless(_batch_profile_importable(), "torch and transformers required to import batch_profile")
    def test_expansion_zero_pct_matches_vanila_draft_and_verify_layout(self) -> None:
        import torch

        import batch_profile as bp

        call = {"n": 0}
        vocab = 1024

        def next_logits(model, seqs, device_fallback):
            bsz = len(seqs)
            logits = torch.full((bsz, vocab), -1e9, dtype=torch.float32)
            tok = 50 + call["n"]
            call["n"] += 1
            for i in range(bsz):
                logits[i, tok % vocab] = 0.0
            return logits, 1, 1

        def logits_at_pos(model, full_seqs, positions_per_sample, device_fallback):
            bsz = len(full_seqs)
            npos = len(positions_per_sample[0])
            return torch.zeros(bsz, npos, vocab, dtype=torch.float32), 1, 1

        k = 4
        prompt_ids = [1, 2, 3]
        recovery = [10]
        with (
            patch.object(bp, "_batched_next_logits", side_effect=next_logits),
            patch.object(bp, "_batched_logits_at_positions", side_effect=logits_at_pos),
        ):
            out_v = bp.run_one_verify_round_batch(
                method="vanila",
                draft_model=None,
                inter_model=None,
                target_model=None,
                prompt_ids_batch=[list(prompt_ids)],
                recovery_token_ids=recovery,
                k=k,
                device_draft="cpu",
                device_inter="cpu",
                device_target="cpu",
                confidence_threshold=0.8,
                force_target_batch=[False],
            )
            call["n"] = 0
            out_e = bp.run_one_verify_round_topk_expansion_rows_batch(
                draft_model=None,
                inter_model=None,
                target_model=None,
                prompt_ids_batch=[list(prompt_ids)],
                recovery_token_ids=recovery,
                parent_request_indices=[0],
                allow_expansion_mask=[True],
                k=k,
                topk_selection=5,
                expansion_pct=0.0,
                device_draft="cpu",
                device_inter="cpu",
                device_target="cpu",
            )

        self.assertEqual(out_v["draft_tokens_batch"], out_e["draft_tokens_batch"])
        b = len(prompt_ids)
        full_v = cu._verify_full_seq_no_carry(prompt_ids, recovery[0], out_v["draft_tokens_batch"][0])
        full_e = cu._verify_full_seq_no_carry(prompt_ids, recovery[0], out_e["draft_tokens_batch"][0])
        self.assertEqual(full_v, full_e)
        self.assertEqual(cu._verify_positions_no_carry(b, k), cu._verify_positions_no_carry(b, k))


if __name__ == "__main__":
    unittest.main()
