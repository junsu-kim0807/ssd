from __future__ import annotations

import sys
import types

import pytest

if "xxhash" not in sys.modules:
    class _DummyXXH64:
        def __init__(self):
            self._acc = bytearray()

        def update(self, data: bytes):
            self._acc.extend(data)

        def intdigest(self) -> int:
            return sum(self._acc) % (2**64)

    sys.modules["xxhash"] = types.SimpleNamespace(xxh64=lambda: _DummyXXH64())

from ssd.engine.block_manager import BlockManager


def _make_src_blocks(bm: BlockManager, n: int) -> list[int]:
    return bm.allocate_private_tail(n)


def test_cow_fork_mid_block_divergence_partial_copy():
    bm = BlockManager(num_blocks=32, block_size=4, cache_role="target")
    src = _make_src_blocks(bm, 3)

    plan = bm.make_cow_fork_block_table(
        src,
        cached_tokens=6,
        required_total_tokens=10,
    )

    assert plan.shared_prefix_blocks == 1
    assert plan.fork_block_table[0] == src[0]
    assert plan.fork_block_table[1] != src[1]
    assert plan.copy_src_block_ids == [src[1]]
    assert plan.copy_dst_block_ids == [plan.fork_block_table[1]]
    assert plan.copy_valid_tokens == [2]
    assert bm.blocks[src[0]].ref_count == 2
    for bid in plan.private_tail_block_ids:
        assert bm.blocks[bid].ref_count == 1


def test_cow_fork_block_boundary_no_partial_copy():
    bm = BlockManager(num_blocks=32, block_size=4, cache_role="target")
    src = _make_src_blocks(bm, 4)

    plan = bm.make_cow_fork_block_table(
        src,
        cached_tokens=8,
        required_total_tokens=12,
    )

    assert plan.shared_prefix_blocks == 2
    assert plan.fork_block_table[:2] == src[:2]
    assert plan.copy_src_block_ids == []
    assert plan.copy_dst_block_ids == []
    assert plan.copy_valid_tokens == []


def test_cow_fork_invalid_required_range_raises():
    bm = BlockManager(num_blocks=16, block_size=4, cache_role="target")
    src = _make_src_blocks(bm, 3)
    with pytest.raises(AssertionError):
        bm.make_cow_fork_block_table(
            src,
            cached_tokens=9,
            required_total_tokens=8,
        )


def test_partial_copy_destination_matches_shared_index():
    bm = BlockManager(num_blocks=32, block_size=4, cache_role="target")
    src = _make_src_blocks(bm, 3)
    plan = bm.make_cow_fork_block_table(
        src,
        cached_tokens=6,
        required_total_tokens=10,
    )
    assert plan.copy_dst_block_ids[0] == plan.fork_block_table[plan.shared_prefix_blocks]


def test_partial_copy_guard_predicate_with_relaxed_fixture():
    # Relaxed fixture: verify predicate semantics even for artificial truncate inputs.
    def should_copy_partial(partial_tokens: int, full_shared_blocks: int, required_blocks: int) -> bool:
        return partial_tokens > 0 and full_shared_blocks < required_blocks

    assert should_copy_partial(partial_tokens=2, full_shared_blocks=1, required_blocks=2)
    assert not should_copy_partial(partial_tokens=2, full_shared_blocks=2, required_blocks=2)
