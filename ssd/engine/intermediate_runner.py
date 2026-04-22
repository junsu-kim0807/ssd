"""Colocated intermediate model runner for hierarchical verification (sync spec)."""

from ssd.config import Config
from ssd.engine.intermediate_shard_config import make_intermediate_shard_config
from ssd.engine.model_runner import ModelRunner


class IntermediateRunner(ModelRunner):
    """Target-sized API but KV lives on ``Sequence.inter_block_table`` / ``num_inter_cached_tokens``."""

    @classmethod
    def create_intermediate_config(cls, cfg: Config) -> Config:
        return make_intermediate_shard_config(cfg)

    def __init__(self, cfg: Config):
        icfg = self.create_intermediate_config(cfg)
        super().__init__(
            icfg, rank=0, event=None, is_draft=False, num_tp_gpus=1, is_intermediate=True)
