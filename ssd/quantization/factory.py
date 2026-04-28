"""Factory functions for parallel Linear layers, with optional quantization.

Phase 1 supports MODELOPT_FP8_PER_TENSOR; other backends raise. The dense path
(``quant_spec is None`` or a force-dense module) returns the existing
:mod:`ssd.layers.linear` classes so non-quantized runs are byte-identical.
"""

from __future__ import annotations

import fnmatch

import torch.distributed as dist
from torch import nn

from ssd.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
from ssd.quantization.spec import QuantBackend, QuantSpec


def _is_force_dense(module_path: str, spec: QuantSpec | None) -> bool:
    if spec is None or not spec.is_quantized:
        return True
    if not spec.quantize_linear_modules:
        return True
    patterns = list(spec.force_dense_module_patterns) + list(spec.ignored_layers)
    for pat in patterns:
        if not pat:
            continue
        if any(c in pat for c in "*?["):
            if fnmatch.fnmatch(module_path, pat):
                return True
        elif pat in module_path:
            return True
    return False


def _unsupported(spec: QuantSpec, kind: str) -> "NoReturn":
    raise NotImplementedError(
        f"{kind}: backend {spec.backend.value} is not implemented in Phase 1 "
        "(supported: modelopt_fp8_per_tensor)."
    )


def make_qkv_linear(
    hidden_size: int,
    head_size: int,
    total_num_heads: int,
    total_num_kv_heads: int | None = None,
    bias: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_size: int = 1,
    quant_spec: QuantSpec | None = None,
    module_path: str = "",
) -> nn.Module:
    if _is_force_dense(module_path, quant_spec):
        return QKVParallelLinear(
            hidden_size, head_size, total_num_heads, total_num_kv_heads,
            bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    if quant_spec.backend == QuantBackend.MODELOPT_FP8_PER_TENSOR:
        from ssd.quantization.fp8_per_tensor import FP8PerTensorQKVLinear
        return FP8PerTensorQKVLinear(
            hidden_size, head_size, total_num_heads, total_num_kv_heads,
            bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    _unsupported(quant_spec, "make_qkv_linear")


def make_merged_column_linear(
    input_size: int,
    output_sizes: list[int],
    bias: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_size: int = 1,
    quant_spec: QuantSpec | None = None,
    module_path: str = "",
) -> nn.Module:
    if _is_force_dense(module_path, quant_spec):
        return MergedColumnParallelLinear(
            input_size, output_sizes, bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    if quant_spec.backend == QuantBackend.MODELOPT_FP8_PER_TENSOR:
        from ssd.quantization.fp8_per_tensor import FP8PerTensorMergedColumnLinear
        return FP8PerTensorMergedColumnLinear(
            input_size, output_sizes, bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    _unsupported(quant_spec, "make_merged_column_linear")


def make_row_linear(
    input_size: int,
    output_size: int,
    bias: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_size: int = 1,
    quant_spec: QuantSpec | None = None,
    module_path: str = "",
) -> nn.Module:
    if _is_force_dense(module_path, quant_spec):
        return RowParallelLinear(
            input_size, output_size, bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    if quant_spec.backend == QuantBackend.MODELOPT_FP8_PER_TENSOR:
        from ssd.quantization.fp8_per_tensor import FP8PerTensorRowLinear
        return FP8PerTensorRowLinear(
            input_size, output_size, bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    _unsupported(quant_spec, "make_row_linear")


def make_column_linear(
    input_size: int,
    output_size: int,
    bias: bool = False,
    tp_group: dist.ProcessGroup | None = None,
    tp_size: int = 1,
    quant_spec: QuantSpec | None = None,
    module_path: str = "",
) -> nn.Module:
    if _is_force_dense(module_path, quant_spec):
        return ColumnParallelLinear(
            input_size, output_size, bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    if quant_spec.backend == QuantBackend.MODELOPT_FP8_PER_TENSOR:
        from ssd.quantization.fp8_per_tensor import FP8PerTensorColumnLinear
        return FP8PerTensorColumnLinear(
            input_size, output_size, bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
    _unsupported(quant_spec, "make_column_linear")


__all__ = [
    "make_qkv_linear",
    "make_merged_column_linear",
    "make_row_linear",
    "make_column_linear",
]
