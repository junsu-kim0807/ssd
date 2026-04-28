"""QuantBackend / QuantSpec — describes how an intermediate checkpoint is quantized.

Phase 0 only stores the description; no kernels or layers consume it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QuantBackend(str, Enum):
    NONE = "none"
    MODELOPT_FP8_PER_TENSOR = "modelopt_fp8_per_tensor"
    COMPRESSED_TENSORS_FP8_BLOCK = "compressed_tensors_fp8_block"
    MODELOPT_NVFP4 = "modelopt_nvfp4"
    GENERIC_FP4 = "generic_fp4"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class QuantSpec:
    """Resolved quantization description for a single checkpoint."""

    backend: QuantBackend
    weight_dtype: str            # "fp8_e4m3" | "fp4_e2m1" | ...
    scale_layout: str            # "per_tensor" | "per_channel" | "block_128x128" | "block_1x16"
    activation_quant: str        # "static_per_tensor" | "dynamic_per_token" | "dynamic_per_block_16" | "none"
    kernel_type: str             # "scaled_mm" | "block_fp8_triton" | "nvfp4_cutlass" | "weight_only_dequant"
    checkpoint_naming: str       # "modelopt" | "compressed_tensors" | "generic"
    quantize_linear_modules: bool = True
    force_dense_module_patterns: tuple[str, ...] = field(
        default=(
            "lm_head",
            "embed_tokens",
            "input_layernorm",
            "post_attention_layernorm",
            "norm",
        )
    )
    # Raw HF quantization_config dict (kept for diagnostics / future stages).
    raw: dict | None = None
    # Modules the checkpoint itself flagged as not quantized (verbatim from
    # ``ignored_layers`` / ``excluded_layers``); merged into force-dense at runtime.
    ignored_layers: tuple[str, ...] = ()

    @property
    def is_quantized(self) -> bool:
        return self.backend != QuantBackend.NONE
