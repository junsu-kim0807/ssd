from copy import copy
from enum import Enum, auto
from itertools import count

from ssd.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


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
    
    def __getstate__(self):
        state = {}
        for attr in self._ATTRIBUTES:
            state[attr] = getattr(self, attr)
        return state

    def __setstate__(self, state):
        for attr in self._ATTRIBUTES:
            setattr(self, attr, state.get(attr))
