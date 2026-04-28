"""FP8 per-tensor Linear (ModelOpt-style static quantization).

Phase 1 MVP: separate q/k/v and gate/up sub-linears (no scale merging).
Activation uses static per-tensor ``input_scale``; weights use per-tensor
``weight_scale``. Forward path is :func:`torch._scaled_mm`; a debug
fallback (``SSD_QUANT_DEBUG_DEQUANT=1``) dequantizes weights to bf16 and
runs :func:`F.linear` so loader/scale/kernel issues can be isolated.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ssd.layers.linear import LinearBase, divide


_FP8_E4M3_MAX = 448.0


def _debug_dequant_enabled() -> bool:
    return os.environ.get("SSD_QUANT_DEBUG_DEQUANT", "0") == "1"


def _to_fp8_view(loaded: torch.Tensor) -> torch.Tensor:
    """Coerce a loaded weight tensor to ``float8_e4m3fn`` without modifying bytes."""
    if loaded.dtype == torch.float8_e4m3fn:
        return loaded
    if loaded.dtype in (torch.int8, torch.uint8):
        return loaded.view(torch.float8_e4m3fn)
    # Last-resort cast (e.g. checkpoint stored as bf16/fp32). Lossy but defensive.
    return loaded.to(torch.float8_e4m3fn)


def _quantize_input_static(x_2d: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Per-tensor static activation quantization to FP8 E4M3."""
    scale = input_scale.to(x_2d.device, dtype=torch.float32).reshape(())
    x = x_2d.to(torch.float32) / scale
    x = x.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return x.to(torch.float8_e4m3fn)


def _fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,           # (out, in), float8_e4m3fn
    weight_scale: torch.Tensor,     # scalar fp32
    input_scale: torch.Tensor,      # scalar fp32
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``y = x @ weight.T`` with FP8 per-tensor scales (or dequant fallback)."""
    if _debug_dequant_enabled():
        w_dq = weight.to(torch.float32) * weight_scale.to(torch.float32).reshape(())
        return F.linear(x, w_dq.to(x.dtype), bias)

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    x_q = _quantize_input_static(x_2d, input_scale)
    s_a = input_scale.to(torch.float32).reshape(())
    s_b = weight_scale.to(torch.float32).reshape(())
    out = torch._scaled_mm(
        x_q,
        weight.t(),
        scale_a=s_a,
        scale_b=s_b,
        bias=bias,
        out_dtype=out_dtype,
    )
    if isinstance(out, tuple):  # older _scaled_mm signature returned (out, amax)
        out = out[0]
    return out.reshape(*orig_shape[:-1], out.shape[-1])


# ---------------------------------------------------------------------------
# Single-projection FP8 linears
# ---------------------------------------------------------------------------


class FP8PerTensorColumnLinear(LinearBase):
    """Column-parallel FP8 Linear (no merge across logical projections)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__(input_size, output_size, tp_dim=0, tp_group=tp_group, tp_size=tp_size)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.weight_scale.weight_loader = self._scalar_loader
        self.input_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.input_scale.weight_loader = self._scalar_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self._bias_loader
        else:
            self.register_parameter("bias", None)

    def _weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        loaded = _to_fp8_view(loaded)
        shard_size = param.data.size(0)
        start = self.tp_rank * shard_size
        loaded = loaded.narrow(0, start, shard_size)
        param.data.copy_(loaded)

    def _scalar_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.reshape(-1)[:1].to(param.dtype))

    def _bias_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        shard_size = param.data.size(0)
        start = self.tp_rank * shard_size
        param.data.copy_(loaded.narrow(0, start, shard_size).to(param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _fp8_linear(x, self.weight, self.weight_scale, self.input_scale, self.bias, x.dtype)


class FP8PerTensorRowLinear(LinearBase):
    """Row-parallel FP8 Linear with all-reduce on output."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__(input_size, output_size, tp_dim=1, tp_group=tp_group, tp_size=tp_size)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size_per_partition,
                        dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight.weight_loader = self._weight_loader
        self.weight_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.weight_scale.weight_loader = self._scalar_loader
        self.input_scale = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        self.input_scale.weight_loader = self._scalar_loader

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self._bias_loader
        else:
            self.register_parameter("bias", None)

    def _weight_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        loaded = _to_fp8_view(loaded)
        shard_size = param.data.size(1)
        start = self.tp_rank * shard_size
        loaded = loaded.narrow(1, start, shard_size)
        param.data.copy_(loaded)

    def _scalar_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.reshape(-1)[:1].to(param.dtype))

    def _bias_loader(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        param.data.copy_(loaded.to(param.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bias if self.tp_rank == 0 else None
        y = _fp8_linear(x, self.weight, self.weight_scale, self.input_scale, b, x.dtype)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self.tp_group)
        return y


# ---------------------------------------------------------------------------
# Separate QKV / Merged-Column wrappers (no scale merging in MVP)
# ---------------------------------------------------------------------------


class FP8PerTensorQKVLinear(nn.Module):
    """Wrapper exposing q_proj / k_proj / v_proj as separate FP8 column linears.

    HF checkpoint keys (``self_attn.q_proj.weight``, ...) map 1:1 to the
    submodule attribute names so the loader needs no remap.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        total_num_kv_heads = total_num_kv_heads if total_num_kv_heads is not None else total_num_heads

        self.q_proj = FP8PerTensorColumnLinear(
            hidden_size, total_num_heads * head_size, bias=bias,
            tp_group=tp_group, tp_size=tp_size,
        )
        self.k_proj = FP8PerTensorColumnLinear(
            hidden_size, total_num_kv_heads * head_size, bias=bias,
            tp_group=tp_group, tp_size=tp_size,
        )
        self.v_proj = FP8PerTensorColumnLinear(
            hidden_size, total_num_kv_heads * head_size, bias=bias,
            tp_group=tp_group, tp_size=tp_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)


class FP8PerTensorMergedColumnLinear(nn.Module):
    """Wrapper exposing gate_proj / up_proj as separate FP8 column linears."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        if len(output_sizes) != 2:
            raise NotImplementedError(
                f"FP8PerTensorMergedColumnLinear: only 2 outputs (gate, up) supported, "
                f"got {len(output_sizes)}"
            )
        self.gate_proj = FP8PerTensorColumnLinear(
            input_size, output_sizes[0], bias=bias, tp_group=tp_group, tp_size=tp_size,
        )
        self.up_proj = FP8PerTensorColumnLinear(
            input_size, output_sizes[1], bias=bias, tp_group=tp_group, tp_size=tp_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)


__all__ = [
    "FP8PerTensorColumnLinear",
    "FP8PerTensorRowLinear",
    "FP8PerTensorQKVLinear",
    "FP8PerTensorMergedColumnLinear",
]
