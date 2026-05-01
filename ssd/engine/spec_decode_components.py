from __future__ import annotations

import math
from dataclasses import dataclass

from ssd.config import Config
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase
from ssd.engine.pivot_branch_planner import PivotExpansionConfig
from ssd.engine.pivot_executor_flat import PivotExecutorFlat
from ssd.engine.pivot_precollapse_speculator_sync import PivotPrecollapseSpeculatorSync
from ssd.engine.pivot_precollapse_verifier import PivotPrecollapseVerifier
from ssd.engine.pivot_speculator_sync import PivotRootSpeculatorSync
from ssd.engine.pivot_tree_executor import PivotTreeScratchExecutor
from ssd.engine.pivot_tree_speculator import PivotTreeScratchSpeculator
from ssd.engine.spec_policy_traits import is_pivot_legacy, uses_hierarchical_verify
from ssd.engine.speculator_sync import SpeculatorSync
from ssd.engine.verifier import Verifier
from ssd.engine.verifier_hierarchical import VerifierHierarchical
from ssd.engine.verifier_pivot import VerifierPivot


@dataclass
class SpecDecodeComponents:
    speculator: SpeculatorBase
    verifier: VerifierBase


def _pivot_expansion_threshold_domain_label(criteria: str) -> str:
    return (
        "full_softmax_p_top1_minus_p_top2"
        if criteria == "softmax_residual"
        else "binary_top1_vs_top2_proxy"
    )


def _pivot_expansion_pct_caps_rows(policy: str) -> bool:
    """Policies that use ``pivot_expansion_pct`` as a hard cap on expandable parents."""
    return policy in {"dynamic", "dynamic_expansion"}


def build_spec_components(
    config: Config,
    *,
    scheduler,
    draft_runner,
    model_runner,
    intermediate_runner,
    tokenizer,
    metrics: dict,
    enable_profile_trace: bool,
) -> SpecDecodeComponents:
    if config.draft_async:
        raise ValueError("build_spec_components only supports sync spec paths")

    if (
        config.spec_policy == "pivot_tree_scratch"
        and config.pivot_expansion_policy == "dynamic_expansion"
    ):
        raise ValueError(
            "dynamic_expansion is only supported for spec_policy pivot and pivot_precollapse "
            "(not pivot_tree_scratch)"
        )

    if config.spec_policy == "pivot_hierarchical":
        # Mirror the gate enforced in ``LLMEngine.create_inference_step``: the
        # ``pivot_hierarchical`` policy is not implemented yet (it requires
        # ``PivotHierarchicalFusedStep`` with branch-local HV state). Failing here
        # too keeps the engine wiring and the component wiring in agreement.
        raise NotImplementedError(
            "pivot_hierarchical requires PivotHierarchicalFusedStep with branch-local HV state"
        )

    if config.spec_policy == "pivot":
        max_expand_rows = config.max_num_seqs * max(1, int(config.pivot_topk))
        if _pivot_expansion_pct_caps_rows(config.pivot_expansion_policy) and float(
            config.pivot_expansion_pct
        ) > 0.0:
            # Reuse ``pivot_expansion_pct`` as a hard expansion cap on parents.
            # Worst-case added rows per expanded parent are ``topk - 1`` (branch 0 already exists).
            max_expand_reqs = int(math.floor(config.max_num_seqs * float(config.pivot_expansion_pct)))
            max_expand_rows = config.max_num_seqs + max(0, max_expand_reqs) * max(
                0, int(config.pivot_topk) - 1
            )
        print(
            {
                "spec_policy": config.spec_policy,
                "pivot_expansion_policy": config.pivot_expansion_policy,
                "pivot_expansion_pct": config.pivot_expansion_pct,
                "pivot_expansion_threshold": config.pivot_expansion_threshold,
                "pivot_expansion_criteria": config.pivot_expansion_criteria,
                "pivot_expansion_slope_thresholds": config.pivot_expansion_slope_thresholds,
                "pivot_expansion_slope_branch_counts": config.pivot_expansion_slope_branch_counts,
                "pivot_expansion_threshold_domain": _pivot_expansion_threshold_domain_label(
                    config.pivot_expansion_criteria
                ),
                "pivot_topk": config.pivot_topk,
                "max_expand_rows": max_expand_rows,
            },
            flush=True,
        )
        speculator = PivotRootSpeculatorSync(
            lookahead=config.speculate_k,
            device=config.device,
            draft_model_runner=draft_runner,
            target_model_runner=model_runner,
            intermediate_runner=intermediate_runner,
            scheduler=scheduler,
            expansion_cfg=PivotExpansionConfig(
                policy=config.pivot_expansion_policy,
                criteria=config.pivot_expansion_criteria,
                expansion_pct=config.pivot_expansion_pct,
                threshold=config.pivot_expansion_threshold,
                topk=config.pivot_topk,
                slope_thresholds=tuple(config.pivot_expansion_slope_thresholds),
                slope_branch_counts=tuple(config.pivot_expansion_slope_branch_counts),
            ),
            max_expand_rows=max_expand_rows,
            enable_profile_trace=enable_profile_trace,
        )
        verifier = PivotExecutorFlat(
            lookahead=config.speculate_k,
            device=config.device,
            target_model_runner=model_runner,
            scheduler=scheduler,
            metrics=metrics,
            enable_profile_trace=enable_profile_trace,
        )
        return SpecDecodeComponents(speculator=speculator, verifier=verifier)

    if config.spec_policy == "pivot_precollapse":
        max_expand_rows = config.max_num_seqs * max(1, int(config.pivot_topk))
        if _pivot_expansion_pct_caps_rows(config.pivot_expansion_policy) and float(
            config.pivot_expansion_pct
        ) > 0.0:
            max_expand_reqs = int(math.floor(config.max_num_seqs * float(config.pivot_expansion_pct)))
            max_expand_rows = config.max_num_seqs + max(0, max_expand_reqs) * max(
                0, int(config.pivot_topk) - 1
            )
        print(
            {
                "spec_policy": config.spec_policy,
                "pivot_expansion_policy": config.pivot_expansion_policy,
                "pivot_expansion_pct": config.pivot_expansion_pct,
                "pivot_expansion_threshold": config.pivot_expansion_threshold,
                "pivot_expansion_criteria": config.pivot_expansion_criteria,
                "pivot_expansion_slope_thresholds": config.pivot_expansion_slope_thresholds,
                "pivot_expansion_slope_branch_counts": config.pivot_expansion_slope_branch_counts,
                "pivot_expansion_threshold_domain": _pivot_expansion_threshold_domain_label(
                    config.pivot_expansion_criteria
                ),
                "pivot_precollapse_score_method": config.pivot_precollapse_score_method,
                "pivot_topk": config.pivot_topk,
                "max_expand_rows": max_expand_rows,
            },
            flush=True,
        )
        speculator = PivotPrecollapseSpeculatorSync(
            lookahead=config.speculate_k,
            device=config.device,
            draft_model_runner=draft_runner,
            target_model_runner=model_runner,
            intermediate_runner=intermediate_runner,
            scheduler=scheduler,
            expansion_cfg=PivotExpansionConfig(
                policy=config.pivot_expansion_policy,
                criteria=config.pivot_expansion_criteria,
                expansion_pct=config.pivot_expansion_pct,
                threshold=config.pivot_expansion_threshold,
                topk=config.pivot_topk,
                slope_thresholds=tuple(config.pivot_expansion_slope_thresholds),
                slope_branch_counts=tuple(config.pivot_expansion_slope_branch_counts),
            ),
            max_expand_rows=max_expand_rows,
            enable_profile_trace=enable_profile_trace,
            score_method=config.pivot_precollapse_score_method,
            score_temperature_aware=config.pivot_precollapse_score_temperature_aware,
        )
        verifier = PivotPrecollapseVerifier(
            lookahead=config.speculate_k,
            device=config.device,
            target_model_runner=model_runner,
            sampler_x=config.sampler_x,
            async_fan_out=config.async_fan_out,
            jit_speculate=config.jit_speculate,
            tokenizer=tokenizer,
            metrics=metrics,
            enable_profile_trace=enable_profile_trace,
        )
        return SpecDecodeComponents(speculator=speculator, verifier=verifier)

    if config.spec_policy == "pivot_tree_scratch":
        max_expand_rows = config.max_num_seqs * max(1, int(config.pivot_topk))
        if _pivot_expansion_pct_caps_rows(config.pivot_expansion_policy) and float(
            config.pivot_expansion_pct
        ) > 0.0:
            max_expand_reqs = int(math.floor(config.max_num_seqs * float(config.pivot_expansion_pct)))
            max_expand_rows = config.max_num_seqs + max(0, max_expand_reqs) * max(
                0, int(config.pivot_topk) - 1
            )
        speculator = PivotTreeScratchSpeculator(
            lookahead=config.speculate_k,
            device=config.device,
            draft_model_runner=draft_runner,
            target_model_runner=model_runner,
            intermediate_runner=intermediate_runner,
            scheduler=scheduler,
            expansion_cfg=PivotExpansionConfig(
                policy=config.pivot_expansion_policy,
                criteria=config.pivot_expansion_criteria,
                expansion_pct=config.pivot_expansion_pct,
                threshold=config.pivot_expansion_threshold,
                topk=config.pivot_topk,
                slope_thresholds=tuple(config.pivot_expansion_slope_thresholds),
                slope_branch_counts=tuple(config.pivot_expansion_slope_branch_counts),
            ),
            max_expand_rows=max_expand_rows,
            enable_profile_trace=enable_profile_trace,
            metrics=metrics,
        )
        verifier = PivotTreeScratchExecutor(
            lookahead=config.speculate_k,
            device=config.device,
            target_model_runner=model_runner,
            draft_model_runner=draft_runner,
            scheduler=scheduler,
            metrics=metrics,
            enable_profile_trace=enable_profile_trace,
        )
        return SpecDecodeComponents(speculator=speculator, verifier=verifier)

    speculator = SpeculatorSync(
        lookahead=config.speculate_k,
        device=config.device,
        draft_model_runner=draft_runner,
    )

    if uses_hierarchical_verify(config.spec_policy):
        # ``pivot_hierarchical`` is rejected above; the only HV policy that reaches
        # here is plain ``hierarchical`` which uses the inner verifier directly.
        verifier = VerifierHierarchical(
            lookahead=config.speculate_k,
            device=config.device,
            target_model_runner=model_runner,
            intermediate_runner=intermediate_runner,
            target_verify_interval=config.target_verify_interval,
            sampler_x=config.sampler_x,
            async_fan_out=config.async_fan_out,
            jit_speculate=config.jit_speculate,
            tokenizer=tokenizer,
            metrics=metrics,
            enable_profile_trace=enable_profile_trace,
        )
        return SpecDecodeComponents(speculator=speculator, verifier=verifier)

    if is_pivot_legacy(config.spec_policy):
        verifier = VerifierPivot(
            lookahead=config.speculate_k,
            device=config.device,
            target_model_runner=model_runner,
            sampler_x=config.sampler_x,
            async_fan_out=config.async_fan_out,
            jit_speculate=config.jit_speculate,
            tokenizer=tokenizer,
            metrics=metrics,
            interval=config.interval,
            threshold=config.threshold,
            expansion_pct=config.expansion_pct,
            enable_profile_trace=enable_profile_trace,
        )
        return SpecDecodeComponents(speculator=speculator, verifier=verifier)

    verifier = Verifier(
        lookahead=config.speculate_k,
        device=config.device,
        target_model_runner=model_runner,
        sampler_x=config.sampler_x,
        async_fan_out=config.async_fan_out,
        jit_speculate=config.jit_speculate,
        tokenizer=tokenizer,
        metrics=metrics,
        enable_profile_trace=enable_profile_trace,
    )
    return SpecDecodeComponents(speculator=speculator, verifier=verifier)
