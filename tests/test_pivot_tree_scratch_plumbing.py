import torch

from ssd.engine.helpers.pivot_tree_helpers import build_tree_mask
from ssd.engine.pivot_types import ScratchOwner


class _DummyBM:
    def __init__(self):
        self.released = []

    def release_scratch_blocks(self, block_ids):
        self.released.append(list(block_ids))


def test_scratch_owner_release_unreleased_once():
    t_bm = _DummyBM()
    d_bm = _DummyBM()
    owner = ScratchOwner(target_block_ids=[1, 2], draft_block_ids=[3])
    owner.release_unreleased(t_bm, d_bm)
    owner.release_unreleased(t_bm, d_bm)
    assert t_bm.released == [[1, 2]]
    assert d_bm.released == [[3]]


def test_build_tree_mask_path_causal_shape():
    mask = build_tree_mask([[0, 1, 2], [3, 4, 5]], device=torch.device("cpu"))
    # flattened [Q, Q] with Q = B * (K+1)
    assert mask.numel() == 36
    m = mask.view(6, 6).bool()
    # Row 2 should see row-local prefix [0,1,2].
    assert m[2, 0]
    assert m[2, 1]
    assert m[2, 2]
    # Cross-row attention is disallowed in phase-0 mask builder.
    assert not m[2, 3]
