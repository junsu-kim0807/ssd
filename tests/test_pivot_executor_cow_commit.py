from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch

if "xxhash" not in sys.modules:
    class _DummyXXH64:
        def __init__(self):
            self._acc = bytearray()

        def update(self, data: bytes):
            self._acc.extend(data)

        def intdigest(self) -> int:
            return sum(self._acc) % (2**64)

    sys.modules["xxhash"] = types.SimpleNamespace(xxh64=lambda: _DummyXXH64())

if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.SimpleNamespace(
        AutoConfig=object,
        AutoTokenizer=object,
    )
if "ssd.engine.scheduler" not in sys.modules:
    sys.modules["ssd.engine.scheduler"] = types.SimpleNamespace(Scheduler=object)
if "ssd.engine.model_runner" not in sys.modules:
    sys.modules["ssd.engine.model_runner"] = types.SimpleNamespace(ModelRunner=object)

from ssd.engine.block_manager import BlockManager
from ssd.engine.pivot_branch_planner import PivotHostPlan
from ssd.engine.pivot_executor_flat import PivotExecutorFlat
from ssd.engine.pivot_types import BranchForkState, PivotBranchBundle
from ssd.engine.sequence import Sequence


def _mk_bundle(
    *,
    parent_batch_size: int,
    parent_index_per_branch: list[int],
    branch_index_per_parent: list[int],
    branch_counts: list[int],
    root_token_ids: list[int],
    root_token_probs: list[float],
    expanded_seqs,
    branch_states,
) -> PivotBranchBundle:
    host_plan = PivotHostPlan(
        parent_index_per_branch=parent_index_per_branch,
        branch_index_per_parent=branch_index_per_parent,
        root_token_ids=root_token_ids,
        branch_counts=branch_counts,
        expand_mask=[c > 1 for c in branch_counts],
        criteria_scores=[0.0] * parent_batch_size,
        root_token_probs=root_token_probs,
    )
    return PivotBranchBundle(
        parent_batch_size=parent_batch_size,
        host_plan=host_plan,
        expanded_seqs=expanded_seqs,
        branch_states=branch_states,
    )


class _DummyRunner:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in these tests
        raise AssertionError("target_model_runner.call should not be reached")


def _mk_seq() -> Sequence:
    Sequence.block_size = 4
    return Sequence([1, 2, 3, 4])


def _mk_executor(t_bm: BlockManager, d_bm: BlockManager) -> PivotExecutorFlat:
    scheduler = SimpleNamespace(
        block_manager=t_bm,
        draft_block_manager=d_bm,
        intermediate_block_manager=None,
    )
    return PivotExecutorFlat(
        lookahead=2,
        device=torch.device("cpu"),
        target_model_runner=_DummyRunner(),
        scheduler=scheduler,
    )


def test_alternative_winner_commit_grafts_winner_and_frees_loser():
    t_bm = BlockManager(num_blocks=64, block_size=4, cache_role="target")
    d_bm = BlockManager(num_blocks=64, block_size=4, cache_role="draft")
    executor = _mk_executor(t_bm, d_bm)

    parent = _mk_seq()
    parent.block_table = t_bm.allocate_private_tail(3)
    parent.draft_block_table = d_bm.allocate_private_tail(3)
    parent.num_cached_tokens = 6
    parent.num_draft_cached_tokens = 6
    shared = 1

    winner_t = t_bm.make_fork_block_table(parent.block_table, required_total_blocks=3, shared_prefix_blocks=shared)
    loser_t = t_bm.make_fork_block_table(parent.block_table, required_total_blocks=3, shared_prefix_blocks=shared)
    winner_d = d_bm.make_fork_block_table(parent.draft_block_table, required_total_blocks=3, shared_prefix_blocks=shared)
    loser_d = d_bm.make_fork_block_table(parent.draft_block_table, required_total_blocks=3, shared_prefix_blocks=shared)
    winner_table_t, winner_tail_t = winner_t
    loser_table_t, loser_tail_t = loser_t
    winner_table_d, winner_tail_d = winner_d
    loser_table_d, loser_tail_d = loser_d

    # Branch creation refcount invariants.
    assert t_bm.blocks[parent.block_table[0]].ref_count == 3
    assert d_bm.blocks[parent.draft_block_table[0]].ref_count == 3
    for bid in winner_tail_t + loser_tail_t:
        assert t_bm.blocks[bid].ref_count == 1
    for bid in winner_tail_d + loser_tail_d:
        assert d_bm.blocks[bid].ref_count == 1

    winner_seq = parent.clone_spec()
    winner_seq.block_table = list(winner_table_t)
    winner_seq.draft_block_table = list(winner_table_d)
    loser_seq = parent.clone_spec()
    loser_seq.block_table = list(loser_table_t)
    loser_seq.draft_block_table = list(loser_table_d)

    bundle = _mk_bundle(
        parent_batch_size=1,
        parent_index_per_branch=[0, 0],
        branch_index_per_parent=[1, 2],
        branch_counts=[3],
        root_token_ids=[11, 12],
        root_token_probs=[0.6, 0.4],
        expanded_seqs=[winner_seq, loser_seq],
        branch_states=[
            BranchForkState(
                parent_seq_idx=0,
                branch_idx=1,
                root_token_id=11,
                root_confidence=0.6,
                target_shared_prefix_blocks=shared,
                draft_shared_prefix_blocks=shared,
                inter_shared_prefix_blocks=0,
                draft_private_tail_block_ids=list(winner_tail_d),
                target_private_tail_block_ids=list(winner_tail_t),
                inter_private_tail_block_ids=[],
                is_parent_inplace=False,
            ),
            BranchForkState(
                parent_seq_idx=0,
                branch_idx=2,
                root_token_id=12,
                root_confidence=0.4,
                target_shared_prefix_blocks=shared,
                draft_shared_prefix_blocks=shared,
                inter_shared_prefix_blocks=0,
                draft_private_tail_block_ids=list(loser_tail_d),
                target_private_tail_block_ids=list(loser_tail_t),
                inter_private_tail_block_ids=[],
                is_parent_inplace=False,
            ),
        ],
    )

    executor._commit_winner_and_release_forks([parent], bundle, winners_per_parent=[1])

    assert parent.block_table[:shared] == winner_table_t[:shared]
    assert parent.block_table[shared:] == winner_tail_t
    assert parent.draft_block_table[shared:] == winner_tail_d
    assert t_bm.blocks[parent.block_table[0]].ref_count == 1
    assert d_bm.blocks[parent.draft_block_table[0]].ref_count == 1
    for bid in winner_tail_t:
        assert t_bm.blocks[bid].ref_count == 1
    for bid in winner_tail_d:
        assert d_bm.blocks[bid].ref_count == 1
    for bid in loser_tail_t:
        assert bid in t_bm.free_block_ids
    for bid in loser_tail_d:
        assert bid in d_bm.free_block_ids


def test_branch0_winner_keeps_parent_and_releases_alternatives():
    t_bm = BlockManager(num_blocks=48, block_size=4, cache_role="target")
    d_bm = BlockManager(num_blocks=48, block_size=4, cache_role="draft")
    executor = _mk_executor(t_bm, d_bm)

    parent = _mk_seq()
    parent.block_table = t_bm.allocate_private_tail(3)
    parent.draft_block_table = d_bm.allocate_private_tail(3)
    before_parent_t = list(parent.block_table)
    before_parent_d = list(parent.draft_block_table)
    shared = 1

    alt_table_t, alt_tail_t = t_bm.make_fork_block_table(
        parent.block_table, required_total_blocks=3, shared_prefix_blocks=shared
    )
    alt_table_d, alt_tail_d = d_bm.make_fork_block_table(
        parent.draft_block_table, required_total_blocks=3, shared_prefix_blocks=shared
    )
    alt_seq = parent.clone_spec()
    alt_seq.block_table = list(alt_table_t)
    alt_seq.draft_block_table = list(alt_table_d)

    branch0_seq = parent
    bundle = _mk_bundle(
        parent_batch_size=1,
        parent_index_per_branch=[0, 0],
        branch_index_per_parent=[0, 1],
        branch_counts=[2],
        root_token_ids=[10, 11],
        root_token_probs=[0.8, 0.2],
        expanded_seqs=[branch0_seq, alt_seq],
        branch_states=[
            BranchForkState(
                parent_seq_idx=0,
                branch_idx=0,
                root_token_id=10,
                root_confidence=0.8,
                target_shared_prefix_blocks=len(parent.block_table),
                draft_shared_prefix_blocks=len(parent.draft_block_table),
                inter_shared_prefix_blocks=0,
                draft_private_tail_block_ids=[],
                target_private_tail_block_ids=[],
                inter_private_tail_block_ids=[],
                is_parent_inplace=True,
            ),
            BranchForkState(
                parent_seq_idx=0,
                branch_idx=1,
                root_token_id=11,
                root_confidence=0.2,
                target_shared_prefix_blocks=shared,
                draft_shared_prefix_blocks=shared,
                inter_shared_prefix_blocks=0,
                draft_private_tail_block_ids=list(alt_tail_d),
                target_private_tail_block_ids=list(alt_tail_t),
                inter_private_tail_block_ids=[],
                is_parent_inplace=False,
            ),
        ],
    )

    executor._commit_winner_and_release_forks([parent], bundle, winners_per_parent=[0])

    assert parent.block_table == before_parent_t
    assert parent.draft_block_table == before_parent_d
    assert t_bm.blocks[parent.block_table[0]].ref_count == 1
    assert d_bm.blocks[parent.draft_block_table[0]].ref_count == 1
    for bid in alt_tail_t:
        assert bid in t_bm.free_block_ids
    for bid in alt_tail_d:
        assert bid in d_bm.free_block_ids
