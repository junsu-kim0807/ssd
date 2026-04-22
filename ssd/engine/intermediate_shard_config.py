"""Shared config builder for hierarchical intermediate (TP shards on all target GPUs)."""

from __future__ import annotations

import dataclasses

from ssd.config import Config


def make_intermediate_shard_config(cfg: Config) -> Config:
    """Config slice for the intermediate model (same TP world size as target)."""
    path = cfg.intermediate or cfg.draft
    util = min(0.45, max(0.05, cfg.gpu_memory_utilization * 0.4))
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
