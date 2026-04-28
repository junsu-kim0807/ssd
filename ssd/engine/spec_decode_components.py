from __future__ import annotations

import math
from dataclasses import dataclass

from ssd.config import Config
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase
from ssd.engine.pivot_branch_planner import PivotExpansionConfig
from ssd.engine.pivot_executor_flat import PivotExecutorFlat
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
        if config.pivot_expansion_policy == "dynamic" and float(config.pivot_expansion_pct) > 0.0:
            # For dynamic policy, reuse ``pivot_expansion_pct`` as a hard expansion cap.
            # Added rows per expanded request are ``topk - 1`` (branch 0 already exists).
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

    if config.spec_policy == "pivot_tree_scratch":
        max_expand_rows = config.max_num_seqs * max(1, int(config.pivot_topk))
        if config.pivot_expansion_policy == "dynamic" and float(config.pivot_expansion_pct) > 0.0:
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
            ),
            max_expand_rows=max_expand_rows,
            enable_profile_trace=enable_profile_trace,
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
