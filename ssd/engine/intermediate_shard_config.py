"""Shared config builder for hierarchical intermediate (TP shards on all target GPUs)."""

from __future__ import annotations

import dataclasses

from ssd.config import Config


def make_intermediate_shard_config(cfg: Config) -> Config:
    """Config slice for the intermediate model (same TP world size as target).

    ``speculate`` must stay **True** with the same ``speculate_k`` as the target so
    ``Attention`` takes the multi-query ``flash_attn_with_kvcache`` path when
    ``run_intermediate_verify_suffix`` sets ``cu_seqlens_q`` (variable tokens per seq:
    optional ``[nic:c0)`` gap plus ``K+1`` scored tail).
    ``speculate=False`` incorrectly routes that forward through single-query decode
    and breaks with e.g. ``batch_size must be equal to batch_size_k``.

    ``enforce_eager`` follows the top-level engine config (same as target/draft).
    On TP-colocated intermediate, ``ModelRunner.run_intermediate_verify_suffix`` may
    still force eager for that forward because verify CUDA graphs are not captured
    for ``intermediate_model`` on the target runner.
    """
    path = cfg.intermediate or cfg.draft
    util = min(0.45, max(0.05, cfg.gpu_memory_utilization * 0.4))
    return dataclasses.replace(
        cfg,
        model=path,
        hf_config=cfg.intermediate_hf_config,
        gpu_memory_utilization=util,
        enforce_eager=cfg.enforce_eager,
        speculate=True,
        speculate_k=cfg.speculate_k,
        draft_async=False,
        num_gpus=cfg.num_gpus,
    )
