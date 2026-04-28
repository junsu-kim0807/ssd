"""Phase 0 — recognize quantized intermediate checkpoints from HF config / safetensors keys.

Returns a :class:`QuantSpec` describing what *would* be needed to consume the
checkpoint; this stage performs no model loading, no kernel selection, and no
parameter creation. Unrecognized quantization metadata is reported via
``ValueError`` so we fail-fast rather than silently treating a quantized
checkpoint as bf16.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any

from ssd.quantization.spec import QuantBackend, QuantSpec


def _decoder_cfg(cfg: Any) -> Any:
    return getattr(cfg, "text_config", cfg)


def _quant_dict(hf_config: Any) -> dict | None:
    """Extract ``quantization_config`` from an HF config (dict or attribute style)."""
    if hf_config is None:
        return None
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        qc = getattr(_decoder_cfg(hf_config), "quantization_config", None)
    if qc is None:
        return None
    if isinstance(qc, dict):
        return dict(qc)
    if hasattr(qc, "to_dict"):
        try:
            return dict(qc.to_dict())
        except Exception:
            pass
    out = {}
    for k in dir(qc):
        if k.startswith("_"):
            continue
        try:
            v = getattr(qc, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out or None


def _peek_safetensor_keys(model_path: str | None, max_files: int = 1, max_keys: int = 4096) -> list[str]:
    """Return a sample of weight names from the first safetensors shard, if available."""
    if not model_path or not os.path.isdir(model_path):
        return []
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        return []
    keys: list[str] = []
    try:
        from safetensors import safe_open  # local import; safetensors is already a dep
    except Exception:
        return []
    for file in files[:max_files]:
        try:
            with safe_open(file, "pt", "cpu") as f:
                for k in f.keys():
                    keys.append(k)
                    if len(keys) >= max_keys:
                        return keys
        except Exception:
            continue
    return keys


def _has_suffix(keys: list[str], suffix: str) -> bool:
    return any(k.endswith(suffix) for k in keys)


def _normalize_ignored(qd: dict) -> tuple[str, ...]:
    raw = qd.get("ignored_layers") or qd.get("excluded_layers") or qd.get("ignore") or []
    if isinstance(raw, str):
        raw = [raw]
    try:
        return tuple(str(x) for x in raw)
    except Exception:
        return ()


def _is_fp4_like(qd: dict, keys: list[str]) -> bool:
    text = " ".join(str(qd.get(k, "")) for k in (
        "quant_algo", "quant_method", "weight_dtype", "weight_quant",
        "fmt", "format", "dtype", "data_type",
    )).lower()
    if "fp4" in text or "nvfp4" in text or "e2m1" in text:
        return True
    if _has_suffix(keys, "weight_scale_2") or _has_suffix(keys, ".weight_scale_2"):
        return True
    return False


def _is_nvfp4(qd: dict, keys: list[str]) -> bool:
    text = " ".join(str(qd.get(k, "")) for k in (
        "quant_algo", "quant_method", "weight_dtype", "weight_quant",
        "fmt", "format", "dtype", "data_type",
    )).lower()
    if "nvfp4" in text:
        return True
    if "modelopt" in str(qd.get("quant_method", "")).lower() and _is_fp4_like(qd, keys):
        return True
    return False


def _is_fp8(qd: dict, keys: list[str]) -> bool:
    text = " ".join(str(qd.get(k, "")) for k in (
        "quant_algo", "quant_method", "weight_dtype", "weight_quant",
        "fmt", "format", "dtype", "data_type",
    )).lower()
    if "fp8" in text or "e4m3" in text or "e5m2" in text:
        return True
    if _has_suffix(keys, ".weight_scale") and (
        _has_suffix(keys, ".input_scale") or _has_suffix(keys, ".weight_scale_inv")
    ):
        return True
    return False


def _block_size(qd: dict) -> tuple[int, int] | None:
    bs = qd.get("weight_block_size") or qd.get("block_size") or qd.get("group_size")
    if bs is None:
        return None
    if isinstance(bs, int):
        return (bs, bs)
    try:
        a, b = int(bs[0]), int(bs[1])
        return (a, b)
    except Exception:
        return None


def _is_block_fp8(qd: dict, keys: list[str]) -> bool:
    if _block_size(qd) is not None:
        return True
    qm = str(qd.get("quant_method", "")).lower()
    if "compressed" in qm or qd.get("format") == "float-quantized":
        return True
    if _has_suffix(keys, ".weight_scale_inv"):
        return True
    return False


def detect_quant_spec(hf_config: Any, model_path: str | None = None) -> QuantSpec | None:
    """Identify the quantization backend declared by ``hf_config``.

    Returns ``None`` for a plain bf16/fp16 checkpoint. Raises ``ValueError`` if
    a ``quantization_config`` is present but not understood — Phase 0 prefers
    fail-fast over silently mis-loading a quantized checkpoint.
    """
    qd = _quant_dict(hf_config)
    if not qd:
        return None

    keys = _peek_safetensor_keys(model_path)
    ignored = _normalize_ignored(qd)
    qm = str(qd.get("quant_method", "")).lower()
    is_fp8 = _is_fp8(qd, keys)
    is_fp4 = _is_fp4_like(qd, keys)

    if is_fp4 and _is_nvfp4(qd, keys):
        return QuantSpec(
            backend=QuantBackend.MODELOPT_NVFP4,
            weight_dtype="fp4_e2m1",
            scale_layout="block_1x16",
            activation_quant="dynamic_per_block_16",
            kernel_type="nvfp4_cutlass",
            checkpoint_naming="modelopt",
            raw=qd,
            ignored_layers=ignored,
        )

    if is_fp4:
        return QuantSpec(
            backend=QuantBackend.GENERIC_FP4,
            weight_dtype="fp4_e2m1",
            scale_layout="unknown",
            activation_quant="unknown",
            kernel_type="weight_only_dequant",
            checkpoint_naming="generic",
            raw=qd,
            ignored_layers=ignored,
        )

    if is_fp8:
        if _is_block_fp8(qd, keys):
            block = _block_size(qd) or (128, 128)
            act = str(qd.get("activation_scheme", "dynamic")).lower()
            return QuantSpec(
                backend=QuantBackend.COMPRESSED_TENSORS_FP8_BLOCK,
                weight_dtype="fp8_e4m3",
                scale_layout=f"block_{block[0]}x{block[1]}",
                activation_quant=(
                    "dynamic_per_token" if act == "dynamic" else "static_per_tensor"
                ),
                kernel_type="block_fp8_triton",
                checkpoint_naming=(
                    "compressed_tensors" if "compressed" in qm else "modelopt"
                ),
                raw=qd,
                ignored_layers=ignored,
            )
        act = str(qd.get("activation_scheme", "static")).lower()
        return QuantSpec(
            backend=QuantBackend.MODELOPT_FP8_PER_TENSOR,
            weight_dtype="fp8_e4m3",
            scale_layout="per_tensor",
            activation_quant=(
                "static_per_tensor" if act == "static" else "dynamic_per_token"
            ),
            kernel_type="scaled_mm",
            checkpoint_naming="modelopt",
            raw=qd,
            ignored_layers=ignored,
        )

    raise ValueError(
        "Unrecognized intermediate quantization_config — refusing to load a "
        f"quantized checkpoint silently. raw={qd!r} sample_keys={keys[:8]!r}"
    )
