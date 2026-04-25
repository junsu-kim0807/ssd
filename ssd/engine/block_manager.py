from collections import deque
from dataclasses import dataclass
from typing import Literal
import xxhash
import numpy as np

from ssd.engine.sequence import Sequence

CacheRole = Literal["target", "draft", "intermediate"]


@dataclass
class CowForkPlan:
    fork_block_table: list[int]
    private_tail_block_ids: list[int]
    shared_prefix_blocks: int
    copy_src_block_ids: list[int]
    copy_dst_block_ids: list[int]
    copy_valid_tokens: list[int]


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        is_draft: bool | None = None,
        cache_role: CacheRole = "target",
        speculate_k: int = -1,
        max_model_len: int = -1,
        verbose: bool = False,
    ):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        if is_draft is not None:
            cache_role = "draft" if is_draft else "target"
        self.cache_role: CacheRole = cache_role
        self.is_draft: bool = cache_role == "draft"
        self.speculate_k: int = speculate_k
        self.verbose: bool = verbose
        self.max_model_len: int = max_model_len

    def _block_table(self, seq: Sequence) -> list[int]:
        if self.cache_role == "draft":
            return seq.draft_block_table
        if self.cache_role == "intermediate":
            return seq.inter_block_table
        return seq.block_table

    def _num_cached_tokens(self, seq: Sequence) -> int:
        if self.cache_role == "draft":
            return seq.num_draft_cached_tokens
        if self.cache_role == "intermediate":
            return seq.num_inter_cached_tokens
        return seq.num_cached_tokens

    def _effective_tokens_for_capacity(self, seq: Sequence) -> int:
        """Token extent for block headroom (may exceed ``len(seq)`` for HV provisional tails)."""
        if self.cache_role == "draft":
            # Draft KV covers logical positions through ``num_draft_cached_tokens - 1``; the next
            # speculate forward writes at that frontier (see ``prepare_decode_tensors_from_seqs``).
            return max(seq.num_tokens, seq.num_draft_cached_tokens + 1)
        if self.cache_role == "intermediate":
            return max(seq.num_tokens, seq.num_inter_cached_tokens)
        return seq.num_tokens

    def _bump_cached_tokens(self, seq: Sequence, delta: int) -> None:
        if self.cache_role == "draft":
            seq.num_draft_cached_tokens += delta
        elif self.cache_role == "intermediate":
            seq.num_inter_cached_tokens += delta
        else:
            seq.num_cached_tokens += delta

        
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _allocate_n_blocks(self, n: int) -> list[Block]:
        if len(self.free_block_ids) < n:
            raise RuntimeError(f"Insufficient free blocks: need {n}, have {len(self.free_block_ids)}")
        
        # Extract n block IDs in one operation
        block_ids = [self.free_block_ids.popleft() for _ in range(n)]

        # Reset all blocks and update tracking sets
        blocks = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            assert block.ref_count == 0
            block.reset()
            self.used_block_ids.add(block_id)
            blocks.append(block)
        
        return blocks

    def _deallocate_n_blocks(self, block_ids: list[int]): # we need to separate where we do ref_count -=1 for forked things from the act of deallocation itself 
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1 # added this -- keeping the assert ensures we only have our fork clones pointing to each seq
            if block.ref_count == 0: 
                self._deallocate_block(block_id)

    def fork_shared_prefix(self, src_block_table: list[int], prefix_blocks: int) -> list[int]:
        """Clone ``src`` prefix by sharing ownership (ref_count += 1)."""
        if prefix_blocks <= 0:
            return []
        prefix_blocks = min(prefix_blocks, len(src_block_table))
        out = list(src_block_table[:prefix_blocks])
        for block_id in out:
            self.blocks[block_id].ref_count += 1
        return out

    def allocate_private_tail(self, num_needed_blocks: int) -> list[int]:
        if num_needed_blocks <= 0:
            return []
        blocks = self._allocate_n_blocks(num_needed_blocks)
        return [b.block_id for b in blocks]

    def make_fork_block_table(
        self,
        src_block_table: list[int],
        required_total_blocks: int,
        shared_prefix_blocks: int,
    ) -> tuple[list[int], list[int]]:
        """Build forked table as shared prefix + private tail."""
        required_total_blocks = max(0, int(required_total_blocks))
        shared_prefix_blocks = max(0, min(int(shared_prefix_blocks), len(src_block_table), required_total_blocks))
        fork_prefix = self.fork_shared_prefix(src_block_table, shared_prefix_blocks)
        tail_needed = required_total_blocks - shared_prefix_blocks
        private_tail_ids = self.allocate_private_tail(tail_needed)
        return fork_prefix + private_tail_ids, private_tail_ids

    def make_cow_fork_block_table(
        self,
        src_block_table: list[int],
        *,
        cached_tokens: int,
        required_total_tokens: int,
    ) -> CowForkPlan:
        assert cached_tokens >= 0
        assert required_total_tokens >= cached_tokens, (
            "COW fork requires required_total_tokens >= cached_tokens, "
            f"got required={required_total_tokens}, cached={cached_tokens}"
        )

        block_size = self.block_size
        required_blocks = (required_total_tokens + block_size - 1) // block_size
        full_shared_blocks = min(
            cached_tokens // block_size,
            len(src_block_table),
            required_blocks,
        )
        partial_tokens = cached_tokens % block_size

        shared_prefix = self.fork_shared_prefix(src_block_table, full_shared_blocks)
        private_needed = required_blocks - full_shared_blocks
        private_tail = self.allocate_private_tail(private_needed)
        fork_table = shared_prefix + private_tail

        copy_src: list[int] = []
        copy_dst: list[int] = []
        copy_valid: list[int] = []
        if partial_tokens > 0 and full_shared_blocks < required_blocks:
            assert private_tail, "partial COW requires a private block"
            assert full_shared_blocks < len(src_block_table)
            assert full_shared_blocks < len(fork_table)
            copy_src = [int(src_block_table[full_shared_blocks])]
            copy_dst = [int(private_tail[0])]
            copy_valid = [int(partial_tokens)]

        return CowForkPlan(
            fork_block_table=fork_table,
            private_tail_block_ids=private_tail,
            shared_prefix_blocks=full_shared_blocks,
            copy_src_block_ids=copy_src,
            copy_dst_block_ids=copy_dst,
            copy_valid_tokens=copy_valid,
        )

    def release_fork(
        self,
        fork_block_table: list[int],
        private_tail_ids: list[int],
        shared_prefix_blocks: int,
    ) -> None:
        """Release forked ownership (shared prefix + private tail)."""
        shared_prefix_blocks = max(0, min(shared_prefix_blocks, len(fork_block_table)))
        for block_id in fork_block_table[:shared_prefix_blocks]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        self._deallocate_n_blocks(private_tail_ids)


    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        block_table = self._block_table(seq)
        assert not block_table
        h = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:  # cache hit
                self._bump_cached_tokens(seq, self.block_size)
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            block_table.append(block_id)


    def deallocate(self, seq: Sequence):
        block_table = self._block_table(seq)
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        if self.cache_role == "draft":
            seq.num_draft_cached_tokens = 0
        elif self.cache_role == "intermediate":
            seq.num_inter_cached_tokens = 0
        else:
            seq.num_cached_tokens = 0

        block_table.clear()

    def can_append(self, seq: Sequence, lookahead_num_tokens: int = 1) -> bool:
        block_table = self._block_table(seq)
        eff_tokens = self._effective_tokens_for_capacity(seq)

        # Check if sequence length + lookahead would exceed max model length
        if eff_tokens + lookahead_num_tokens > self.max_model_len:
            print(f'[block_manager] WARNING: Sequence length + lookahead would exceed max model length', flush=True)
            return False

        # How many blocks do we need in total to cover current tokens + lookahead?
        target_blocks = (eff_tokens + lookahead_num_tokens +
                         self.block_size - 1) // self.block_size
        current_blocks = len(block_table)

        if target_blocks > current_blocks:
            needed = target_blocks - current_blocks
            ok = len(self.free_block_ids) >= needed
        else:
            ok = True  # Current blocks are sufficient
        if self.verbose:
            print(
                "[HV_BLOCK_DEBUG:bms_can_append] "
                f"role={self.cache_role} "
                f"seq_id={seq.seq_id} "
                f"num_tokens={seq.num_tokens} "
                f"num_cached_tokens={seq.num_cached_tokens} "
                f"num_draft_cached_tokens={seq.num_draft_cached_tokens} "
                f"num_inter_cached_tokens={getattr(seq, 'num_inter_cached_tokens', None)} "
                f"hv_num_provisional_tokens={getattr(seq, 'hv_num_provisional_tokens', 0)} "
                f"eff_tokens={eff_tokens} "
                f"lookahead_num_tokens={lookahead_num_tokens} "
                f"current_blocks={current_blocks} "
                f"target_blocks={target_blocks} "
                f"free_blocks={len(self.free_block_ids)} "
                f"can_append={ok}",
                flush=True,
            )
        return ok

    def may_append(self, seq: Sequence, lookahead_num_tokens: int = 1):
        block_table = self._block_table(seq)
        eff_tokens = self._effective_tokens_for_capacity(seq)

        # How many blocks do we need in total to cover current tokens + lookahead?
        target_blocks = (eff_tokens + lookahead_num_tokens +
                         self.block_size - 1) // self.block_size
        current_blocks = len(block_table)
        if self.verbose:
            print(
                "[HV_BLOCK_DEBUG:bms_may_append_before] "
                f"role={self.cache_role} "
                f"seq_id={seq.seq_id} "
                f"eff_tokens={eff_tokens} "
                f"lookahead_num_tokens={lookahead_num_tokens} "
                f"current_blocks={current_blocks} "
                f"target_blocks={target_blocks}",
                flush=True,
            )

        if target_blocks > current_blocks:
            needed = target_blocks - current_blocks
            new_blocks = self._allocate_n_blocks(needed)
            for block in new_blocks:
                block_table.append(block.block_id)
        if self.verbose:
            print(
                "[HV_BLOCK_DEBUG:bms_may_append_after] "
                f"role={self.cache_role} "
                f"seq_id={seq.seq_id} "
                f"current_blocks={len(block_table)} "
                f"block_table={block_table}",
                flush=True,
            )

