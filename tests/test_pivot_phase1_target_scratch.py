import torch

from ssd.engine.helpers.pivot_tree_helpers import (
    build_rowwise_prefix_candidate_mask,
    can_use_target_scratch_phase1a,
    gather_logits_by_path,
)
from ssd.engine.pivot_types import ScratchOwner


def test_gather_logits_by_path_shape():
    path = [[0, 1, 2], [3, 4, 5]]
    logits = torch.randn(6, 11)
    out = gather_logits_by_path(logits, path)
    assert out.shape == (2, 3, 11)


def test_scratch_owner_merge_idempotent_same_object():
    o = ScratchOwner(target_block_ids=[1, 2], draft_block_ids=[])
    o.merge(o)
    assert o.target_block_ids == [1, 2]


def test_build_rowwise_prefix_candidate_mask_length():
    dev = torch.device("cpu")
    k1 = 3
    pos0_list = [8, 8]
    m = build_rowwise_prefix_candidate_mask(pos0_list, k1, device=dev)
    want = sum((p0 + k1) * k1 for p0 in pos0_list)
    assert m.numel() == want
    assert m.dtype == torch.uint8


def test_can_use_target_scratch_phase1a_requires_alignment():
    class _Aligned:
        block_size = 16
        num_cached_tokens = 32

    class _Partial:
        block_size = 16
        num_cached_tokens = 33

    assert can_use_target_scratch_phase1a(_Aligned()) is True
    assert can_use_target_scratch_phase1a(_Partial()) is False
