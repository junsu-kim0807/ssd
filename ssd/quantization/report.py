"""Phase 0 diagnostics: print what was detected, fail-fast for unsupported cases.

No quantized kernels exist yet, so any successful detection (i.e. ``spec is not
None``) terminates intermediate model setup with ``NotImplementedError``. Phase
1 onward will replace ``gate_intermediate_quant_spec`` with backend-specific
dispatch.
"""

from __future__ import annotations

import os
from glob import glob

from ssd.quantization.spec import QuantBackend, QuantSpec


_PRINTED_PATHS: set[str] = set()


def _sample_safetensor_keys(model_path: str | None, max_files: int = 1, max_keys: int = 64) -> list[str]:
    if not model_path or not os.path.isdir(model_path):
        return []
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        return []
    try:
        from safetensors import safe_open
    except Exception:
        return []
    sample: list[str] = []
    for file in files[:max_files]:
        try:
            with safe_open(file, "pt", "cpu") as f:
                for k in f.keys():
                    sample.append(k)
                    if len(sample) >= max_keys:
                        return sample
        except Exception:
            continue
    return sample


def log_intermediate_quant_spec(model_path: str, spec: QuantSpec | None) -> None:
    """One-shot diagnostic print per ``model_path`` for the intermediate runner."""
    if model_path in _PRINTED_PATHS:
        return
    _PRINTED_PATHS.add(model_path)

    print(f"[intermediate.quant] path={model_path}", flush=True)
    if spec is None:
        print("[intermediate.quant] detected backend=NONE (dense bf16/fp16)", flush=True)
        return

    print(f"[intermediate.quant] quantization_config(raw)={spec.raw!r}", flush=True)
    print(f"[intermediate.quant] detected backend={spec.backend.value}", flush=True)
    print(
        f"[intermediate.quant] weight_dtype={spec.weight_dtype} "
        f"scale_layout={spec.scale_layout}",
        flush=True,
    )
    print(
        f"[intermediate.quant] activation_quant={spec.activation_quant} "
        f"kernel_type={spec.kernel_type} naming={spec.checkpoint_naming}",
        flush=True,
    )
    print(f"[intermediate.quant] ignored_layers={list(spec.ignored_layers)}", flush=True)
    print(
        f"[intermediate.quant] force_dense_module_patterns="
        f"{list(spec.force_dense_module_patterns)}",
        flush=True,
    )

    sample = _sample_safetensor_keys(model_path)
    if sample:
        weight_sample = [k for k in sample if k.endswith(".weight")][:4]
        scale_sample = [k for k in sample if "scale" in k][:4]
        print(f"[intermediate.quant] sample weights: {weight_sample}", flush=True)
        print(f"[intermediate.quant] sample scales:  {scale_sample}", flush=True)


def gate_intermediate_quant_spec(spec: QuantSpec | None) -> None:
    """Phase 0 gate: refuse to load any quantized intermediate checkpoint."""
    if spec is None or not spec.is_quantized:
        return
    raise NotImplementedError(
        f"Intermediate checkpoint is quantized (backend={spec.backend.value}). "
        "Phase 0 only performs detection — quantized kernels/loaders are not "
        "implemented yet. Use a bf16/fp16 intermediate for now, or wait for "
        "the corresponding phase (Phase 1: MODELOPT_FP8_PER_TENSOR; Phase 3: "
        "COMPRESSED_TENSORS_FP8_BLOCK; Phase 4: MODELOPT_NVFP4 / GENERIC_FP4)."
    )


__all__ = ["log_intermediate_quant_spec", "gate_intermediate_quant_spec", "QuantBackend"]
