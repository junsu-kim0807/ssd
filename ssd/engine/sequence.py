from copy import copy
from enum import Enum, auto
from itertools import count

from ssd.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class _TokensView:
    """Frozen-prefix + private-tail token view for speculative branch clones.

    Shares ``_base`` (parent's ``token_ids`` list) by reference but freezes
    ``_base_len`` at clone time. Reads in [0, _base_len) fall through to
    ``_base``; reads in [_base_len, _base_len + len(_tail)) hit the private
    tail. All mutations (append/extend/del) only affect ``_tail``.

    Why ``_base_len`` matters: the parent sequence may continue to ``append_token``
    (e.g. branch-0 in-place adds its root to the parent). Without freezing the
    base length, the branch view would observe parent's later mutations and
    return a corrupted prefix.
    """

    __slots__ = ("_base", "_base_len", "_tail")

    def __init__(self, base: list[int], base_len: int, tail: list[int] | None = None):
        self._base = base
        self._base_len = int(base_len)
        self._tail = [] if tail is None else tail

    def __len__(self) -> int:
        return self._base_len + len(self._tail)

    def __bool__(self) -> bool:
        return self._base_len + len(self._tail) > 0

    def __getitem__(self, key):
        bl = self._base_len
        if isinstance(key, int):
            n = bl + len(self._tail)
            if key < 0:
                key += n
            if key < 0 or key >= n:
                raise IndexError(key)
            return self._base[key] if key < bl else self._tail[key - bl]
        if isinstance(key, slice):
            n = bl + len(self._tail)
            start, stop, step = key.indices(n)
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            if start >= bl:
                return self._tail[start - bl : stop - bl]
            if stop <= bl:
                return self._base[start:stop]
            # Mixed: prefix portion + tail portion. Rare on hot path.
            return self._base[start:bl] + self._tail[: stop - bl]
        raise TypeError(f"_TokensView indices must be int or slice, got {type(key).__name__}")

    def append(self, x: int) -> None:
        self._tail.append(int(x))

    def extend(self, xs) -> None:
        self._tail.extend(int(x) for x in xs)

    def __delitem__(self, key) -> None:
        bl = self._base_len
        if isinstance(key, slice):
            start, stop, step = key.indices(bl + len(self._tail))
            if step == 1 and start >= bl:
                # Hot path: ``del seq.token_ids[orig_len:]`` rollback only
                # touches the private tail.
                del self._tail[start - bl : stop - bl]
                return
        # Rare fallback: promote to a flat list and apply the deletion there.
        merged = list(self)
        del merged[key]
        self._base = merged
        self._base_len = len(merged)
        self._tail = []

    def __iter__(self):
        for i in range(self._base_len):
            yield self._base[i]
        yield from self._tail

    def __eq__(self, other) -> bool:
        if isinstance(other, _TokensView):
            return list(self) == list(other)
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented

    def materialize(self) -> list[int]:
        """Promote to a flat ``list[int]`` (e.g. for pickling or APIs that
        require a real list)."""
        return self._base[: self._base_len] + list(self._tail)


_BRANCH_BLOCK_TABLE_ATTRS = frozenset({"block_table", "draft_block_table", "inter_block_table"})


class Sequence:
    counter = count()
    # Sequence.block_size set first thing in ModelRunner init

    _ATTRIBUTES = [
        'seq_id', 'status', 'token_ids', 'last_token', 'num_tokens',
        'num_prompt_tokens', 'num_cached_tokens', 'block_table',
        'last_spec_step_accepted_len', 'draft_block_table',
        'num_draft_cached_tokens', 'temperature', 'draft_temperature', 'max_new_tokens',
        'ignore_eos', 'recovery_token_id', 'last_target_hidden_state',
        'extend_eagle_acts', 'extend_token_ids', 'extend_count',
        # hierarchical verification (HV)
        'hv_round_idx', 'hv_provisional_token_ids', 'hv_provisional_recovery_token_id',
        'hv_num_provisional_tokens', 'inter_block_table', 'num_inter_cached_tokens',
        'intermediate_last_spec_step_accepted_len', 'target_last_spec_step_accepted_len',
    ]

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.last_spec_step_accepted_len = -1 # -1 on first req to force cache miss
        
        self.draft_block_table = [] 
        self.num_draft_cached_tokens = 0

        self.temperature = sampling_params.temperature
        self.draft_temperature = sampling_params.draft_temperature
        self.max_new_tokens = sampling_params.max_new_tokens
        self.ignore_eos = sampling_params.ignore_eos

        self.recovery_token_id = None
        self.last_target_hidden_state = None

        self.extend_eagle_acts = None
        self.extend_token_ids = None
        self.extend_count = 0

        # HV: ``hv_round_idx < r`` => intermediate, ``hv_round_idx == r`` => target (see VerifierHierarchical)
        self.hv_round_idx = 0
        self.hv_provisional_token_ids: list[int] = []
        self.hv_provisional_recovery_token_id: int | None = None
        self.hv_num_provisional_tokens = 0
        self.inter_block_table: list[int] = []
        self.num_inter_cached_tokens = 0
        self.intermediate_last_spec_step_accepted_len = -1
        self.target_last_spec_step_accepted_len = -1

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return (self.num_cached_tokens + self.block_size - 1) // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property 
    def num_draft_cached_blocks(self):
        return (self.num_draft_cached_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_cached_blocks - 1) * self.block_size
    
    @property
    def last_block_num_tokens_draft(self):
        return self.num_tokens - (self.num_draft_cached_blocks - 1) * self.block_size

    def draft_context_len(self) -> int:
        """Tokens covered by draft KV: committed prefix minus one, plus HV provisional body."""
        return self.num_tokens - 1 + self.hv_num_provisional_tokens

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def clone_spec(self):
        """Returns a new sequence with identical state for speculative decoding."""
        cloned = Sequence.__new__(Sequence)
        for attr in self._ATTRIBUTES:
            value = getattr(self, attr)
            setattr(cloned, attr, copy(value))
        return cloned

    def clone_spec_for_branch(self):
        """Clone for a pivot branch that only reads/mutates the appended tail.

        Avoids the O(seq_len) ``token_ids`` list copy by wrapping the parent's
        list in a ``_TokensView`` whose base length is frozen at clone time.
        Subsequent mutations on either side (parent's branch-0 in-place append,
        the branch's tail draft appends, verify-fail rollback ``del [orig_len:]``)
        are isolated.

        ``block_table`` / ``draft_block_table`` / ``inter_block_table`` are
        initialized empty rather than copied: pivot branch construction
        overwrites all three immediately with COW fork plans, so the
        per-clone O(num_blocks) copy would be pure waste.
        """
        cloned = Sequence.__new__(Sequence)
        for attr in self._ATTRIBUTES:
            value = getattr(self, attr)
            if attr == "token_ids":
                if isinstance(value, _TokensView):
                    # Re-cloning a branch (rare). Materialize once so the new
                    # view's ``_base`` length is consistent with ``base_len``.
                    base = value.materialize()
                    base_len = len(base)
                else:
                    base = value
                    base_len = len(value)
                cloned.token_ids = _TokensView(base, base_len)
            elif attr in _BRANCH_BLOCK_TABLE_ATTRS:
                # Pivot Pass 1 overwrites these with COW fork tables; skip the
                # redundant copy. (When intermediate runner is absent, parent's
                # ``inter_block_table`` is also ``[]``, so the result is identical.)
                setattr(cloned, attr, [])
            else:
                setattr(cloned, attr, copy(value))
        return cloned

    def __getstate__(self):
        state = {}
        for attr in self._ATTRIBUTES:
            value = getattr(self, attr)
            if attr == "token_ids" and isinstance(value, _TokensView):
                # Materialize to a real list before pickling so workers don't
                # need to import _TokensView.
                value = value.materialize()
            state[attr] = value
        return state

    def __setstate__(self, state):
        for attr in self._ATTRIBUTES:
            setattr(self, attr, state.get(attr))
