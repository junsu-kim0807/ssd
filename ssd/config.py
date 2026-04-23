import os
from dataclasses import dataclass
from typing import Literal
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT


def _decoder_cfg(cfg):
    return getattr(cfg, "text_config", cfg)


def _cfg_attr(cfg, name, default=None):
    dec = _decoder_cfg(cfg)
    return getattr(dec, name, getattr(cfg, name, default))


@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    # Bench defaults to 0.55 when ``spec_policy=hierarchical`` (sync: target shard 0 + draft + intermediate on GPU 0).
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 
    spec_policy: str = "default"
    spec_hive: bool = False
    interval: int = 0
    threshold: float = 0.8
    expansion_pct: float = 1.0

    # hierarchical verification (sync spec, single verify per step)
    intermediate: str = ""  # HF model dir; empty => use same path as draft
    intermediate_hf_config: AutoConfig | None = None
    target_verify_interval: int = 1  # r: hv_round_idx 0..r-1 intermediate; hv_round_idx == r => target verify

    # HV verify CUDAGraph (ignored unless spec_policy=hierarchical; still gated by enforce_eager)
    enable_intermediate_verify_cudagraph: bool = True
    enable_intermediate_gap_bucket_cudagraph: bool = True
    intermediate_verify_gap_buckets: list[int] | None = None
    enable_target_verify_varlen_cudagraph: bool = True
    target_verify_varlen_buckets: list[int] | None = None

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    # Profiling (disabled when profiler_output_dir is empty / None)
    profiler_mode: Literal[
        "cost_breakdown", "metadata", "cost_metadata", "kernel_breakdown"
    ] = "cost_metadata"
    profiler_output_dir: str | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        self.max_model_len = min(
            self.max_model_len,
            _cfg_attr(self.hf_config, "max_position_embeddings", self.max_model_len),
        )
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft, trust_remote_code=True)
            self.max_model_len = min(
                self.max_model_len,
                _cfg_attr(self.draft_hf_config, "max_position_embeddings", self.max_model_len),
            )
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
            if self.spec_policy not in {"default", "pivot", "hierarchical"}:
                raise ValueError(
                    f"Unsupported spec_policy={self.spec_policy}. "
                    "Use 'default', 'pivot', or 'hierarchical'.")
            if self.interval < 0:
                raise ValueError("interval must be >= 0")
            if not (0.0 <= self.threshold <= 1.0):
                raise ValueError("threshold must be in [0, 1]")
            if self.expansion_pct <= 0.0:
                raise ValueError("expansion_pct must be > 0")
            if self.spec_policy == "pivot":
                assert self.draft_async, "pivot policy currently requires draft_async=True"
                assert self.spec_hive, "pivot policy currently requires spec_hive=True"
            if self.spec_policy == "hierarchical":
                assert not self.draft_async, "hierarchical policy requires draft_async=False (sync spec)"
                assert not self.use_eagle, "hierarchical policy does not support EAGLE yet"
                if self.target_verify_interval < 1:
                    raise ValueError("target_verify_interval must be >= 1 for hierarchical")
                im = self.intermediate or self.draft
                self.intermediate_hf_config = AutoConfig.from_pretrained(im, trust_remote_code=True)
                self.max_model_len = min(
                    self.max_model_len,
                    _cfg_attr(
                        self.intermediate_hf_config,
                        "max_position_embeddings",
                        self.max_model_len,
                    ),
                )
                K = self.speculate_k
                r = self.target_verify_interval
                hv_target_upper = (r + 1) * (K + 1)
                if self.intermediate_verify_gap_buckets is None:
                    self.intermediate_verify_gap_buckets = list(range(K + 2, 2 * K + 3))
                if self.target_verify_varlen_buckets is None:
                    self.target_verify_varlen_buckets = list(range(K + 2, hv_target_upper + 1))

        if self.profiler_output_dir and str(self.profiler_output_dir).strip():
            if self.profiler_mode not in (
                "cost_breakdown",
                "metadata",
                "cost_metadata",
                "kernel_breakdown",
            ):
                raise ValueError(f"Unsupported profiler_mode={self.profiler_mode!r}")
            if self.profiler_mode == "kernel_breakdown":
                if not self.enforce_eager:
                    print(
                        "[Config] profiler_mode=kernel_breakdown: forcing enforce_eager=True "
                        "for interpretable kernel traces.",
                        flush=True,
                    )
                    self.enforce_eager = True

        if self.use_eagle:
            if self.eagle_layers is None:
                L = _cfg_attr(self.hf_config, "num_hidden_layers")
                assert L is not None, "ERROR in Config: num_hidden_layers missing on hf_config"
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_dec = _decoder_cfg(self.hf_config)
                draft_dec = _decoder_cfg(self.draft_hf_config)
                target_rope_theta = getattr(target_dec, "rope_theta", 500000.0)
                draft_rope_theta = getattr(draft_dec, "rope_theta", 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    setattr(draft_dec, "rope_theta", target_rope_theta)
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft decoder max_position_embeddings for RoPE.
                target_max_pos = _cfg_attr(self.hf_config, "max_position_embeddings", 8192)
                draft_max_pos = _cfg_attr(self.draft_hf_config, "max_position_embeddings", 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    setattr(draft_dec, "max_position_embeddings", target_max_pos)
        
        assert self.max_num_batched_tokens >= self.max_model_len
