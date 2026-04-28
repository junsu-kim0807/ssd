"""Phase 0: capability detection for quantized intermediate checkpoints.

This package currently only inspects HF configs / safetensors keys to identify
the quantization backend; no quantized kernels or layers are wired in yet.
"""

from ssd.quantization.spec import QuantBackend, QuantSpec
from ssd.quantization.detect import detect_quant_spec
from ssd.quantization.report import log_intermediate_quant_spec, gate_intermediate_quant_spec

__all__ = [
    "QuantBackend",
    "QuantSpec",
    "detect_quant_spec",
    "log_intermediate_quant_spec",
    "gate_intermediate_quant_spec",
]
