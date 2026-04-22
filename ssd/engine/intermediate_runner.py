"""Colocated intermediate model runner for hierarchical verification (sync spec)."""

import dataclasses

from ssd.config import Config
from ssd.engine.model_runner import ModelRunner


class IntermediateRunner(ModelRunner):
    """Target-sized API but KV lives on ``Sequence.inter_block_table`` / ``num_inter_cached_tokens``."""

    @classmethod
    def create_intermediate_config(cls, cfg: Config) -> Config:
        path = cfg.intermediate or cfg.draft
        util = min(0.45, max(0.05, cfg.gpu_memory_utilization * 0.4))
        # Sync hierarchical: keep ``num_gpus`` aligned with the target Config (node layout / parity).
        # The intermediate ModelRunner still uses ``num_tp_gpus=1`` on rank 0 (no extra TP workers).
        return dataclasses.replace(
            cfg,
            model=path,
            hf_config=cfg.intermediate_hf_config,
            gpu_memory_utilization=util,
            enforce_eager=True,
            speculate=False,
            draft_async=False,
            num_gpus=cfg.num_gpus,
        )

    def __init__(self, cfg: Config):
        icfg = self.create_intermediate_config(cfg)
        super().__init__(
            icfg, rank=0, event=None, is_draft=False, num_tp_gpus=1, is_intermediate=True)
