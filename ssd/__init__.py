"""SSD package root: lazy exports so ``import ssd.utils.*`` works without optional heavy deps."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "LLM",
    "SamplingParams",
    "Sequence",
    "SequenceStatus",
    "ModelRunner",
    "Config",
    "infer_model_family",
    "Attention",
    "set_context",
    "get_context",
    "reset_context",
    "verify",
    "make_glue_decode_input_ids",
    "get_forked_recovery_tokens_from_logits",
    "apply_sampler_x_rescaling",
    "get_custom_mask",
    "run_verify_cudagraph",
    "run_decode_cudagraph",
    "capture_cudagraph",
    "capture_verify_cudagraph",
    "run_fi_tree_decode_cudagraph",
    "prepare_decode_tensors_from_seqs",
    "prepare_block_tables_from_seqs",
    "prepare_prefill_tensors_from_seqs",
    "prepare_prefill_payload",
)

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "LLM": ("ssd.llm", "LLM"),
    "SamplingParams": ("ssd.sampling_params", "SamplingParams"),
    "Sequence": ("ssd.engine.sequence", "Sequence"),
    "SequenceStatus": ("ssd.engine.sequence", "SequenceStatus"),
    "ModelRunner": ("ssd.engine.model_runner", "ModelRunner"),
    "Config": ("ssd.config", "Config"),
    "infer_model_family": ("ssd.utils.misc", "infer_model_family"),
    "Attention": ("ssd.layers.attention", "Attention"),
    "set_context": ("ssd.utils.context", "set_context"),
    "get_context": ("ssd.utils.context", "get_context"),
    "reset_context": ("ssd.utils.context", "reset_context"),
    "verify": ("ssd.utils.verify", "verify"),
    "make_glue_decode_input_ids": ("ssd.utils.async_helpers.async_spec_helpers", "make_glue_decode_input_ids"),
    "get_forked_recovery_tokens_from_logits": (
        "ssd.utils.async_helpers.async_spec_helpers",
        "get_forked_recovery_tokens_from_logits",
    ),
    "apply_sampler_x_rescaling": ("ssd.utils.async_helpers.async_spec_helpers", "apply_sampler_x_rescaling"),
    "get_custom_mask": ("ssd.engine.helpers.mask_helpers", "get_custom_mask"),
    "run_verify_cudagraph": ("ssd.engine.helpers.cudagraph_helpers", "run_verify_cudagraph"),
    "run_decode_cudagraph": ("ssd.engine.helpers.cudagraph_helpers", "run_decode_cudagraph"),
    "capture_cudagraph": ("ssd.engine.helpers.cudagraph_helpers", "capture_cudagraph"),
    "capture_verify_cudagraph": ("ssd.engine.helpers.cudagraph_helpers", "capture_verify_cudagraph"),
    "run_fi_tree_decode_cudagraph": ("ssd.engine.helpers.cudagraph_helpers", "run_fi_tree_decode_cudagraph"),
    "prepare_decode_tensors_from_seqs": ("ssd.engine.helpers.runner_helpers", "prepare_decode_tensors_from_seqs"),
    "prepare_block_tables_from_seqs": ("ssd.engine.helpers.runner_helpers", "prepare_block_tables_from_seqs"),
    "prepare_prefill_tensors_from_seqs": ("ssd.engine.helpers.runner_helpers", "prepare_prefill_tensors_from_seqs"),
    "prepare_prefill_payload": ("ssd.engine.helpers.runner_helpers", "prepare_prefill_payload"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = _EXPORT_MAP[name]
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
