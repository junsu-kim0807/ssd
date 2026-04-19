#!/usr/bin/env python3
"""
Generate Slurm job scripts for profile/batch_profile.py sweeps.

Supports **vanila + --hispec** (with --interval >= 1) alongside topk_expansion and other methods.
Use --methods to restrict (e.g. ``--methods vanila`` for vanila-only).
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_DIR = "/project/def-pnair/junsu/kv_cache/ssd/profile"
DEFAULT_VENV_DIR = "/project/def-pnair/junsu/kv_cache/.venv"
DEFAULT_DATA_DIR = "/project/def-pnair/junsu/kv_cache/ssd/data"
DEFAULT_PROFILE_SCRIPT = "batch_profile.py"
DEFAULT_JOB_ROOT = "/project/6045356/junsu/kv_cache/ssd/profile/scripts/jobs/spec_decode/pivot"
DEFAULT_OUT_LOG_ROOT = "/project/6045356/junsu/kv_cache/ssd/profile/logs/spec_decode/pivot"
DEFAULT_ERR_LOG_ROOT = "/project/6045356/junsu/kv_cache/ssd/profile/scripts/logs/spec_decode/pivot"
DEFAULT_RESULT_ROOT = "/home/junsuk87/scratch/ssd/profile/results/spec_decode/pivot"

DEFAULT_DATASETS = "alpaca,gsm8k,humaneval,qa"
DEFAULT_KS = "4,5,6"
DEFAULT_BATCH_SIZES = "16"
# vanila + hispec is valid (batch_profile); include alongside topk_expansion by default.
DEFAULT_METHODS = "vanila,topk_expansion"
DEFAULT_HISPEC = True
DEFAULT_INTERVAL = 1

DEFAULT_PAIRS = [
    "meta-llama/Llama-3.2-1B-Instruct|meta-llama/Llama-3.1-8B-Instruct|meta-llama/Llama-3.3-70B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct|meta-llama/Llama-3.2-3B-Instruct|meta-llama/Llama-3.3-70B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct|meta-llama/Llama-3.2-1B-Instruct|meta-llama/Llama-3.3-70B-Instruct",
    # "double7/vicuna-68m|lmsys/vicuna-7b-v1.3|lmsys/vicuna-13b-v1.3",
    # "double7/vicuna-160m|lmsys/vicuna-7b-v1.3|lmsys/vicuna-13b-v1.3",
    # "lmsys/vicuna-7b-v1.3|double7/vicuna-160m|lmsys/vicuna-13b-v1.3",
]


@dataclass(frozen=True)
class ModelTriplet:
    draft: str
    intermediate: str
    target: str


@dataclass(frozen=True)
class ResourceSpec:
    tp_size: int
    cpus_per_task: int
    gres: str
    mem_per_gpu: str
    time_limit: str


DATASET_TO_MAX_NEW_TOKENS_ARG = {
    "alpaca": "--alpaca-max-new-tokens",
    "gsm8k": "--gsm8k-max-new-tokens",
    "humaneval": "--humaneval-max-new-tokens",
    "mt_bench": "--mt-bench-max-new-tokens",
    "qa": "--qa-max-new-tokens",
}


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_method(method: str) -> str:
    m = method.strip().lower()
    if m == "vanilla":
        m = "vanila"
    if m not in {"vanila", "bump", "morphable", "topk_expansion"}:
        raise ValueError(f"Unsupported method: {method}")
    return m


def format_float_tag(value: float) -> str:
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return s.replace(".", "p")


def build_config_tag(
    *,
    method: str,
    hispec: bool,
    interval: int,
    topk_selection: int,
    expansion_pct: float,
) -> str:
    parts: list[str] = []
    if method == "vanila" and not hispec:
        parts.append("vanila_plain")
    if hispec:
        parts.append(f"hispec_int{interval}")
    if method == "topk_expansion":
        parts.append(f"topksel{topk_selection}")
        parts.append(f"exppct{format_float_tag(expansion_pct)}")
    return "_".join(parts)


def method_allowed_for_pair(pair: ModelTriplet, method: str) -> bool:
    restricted_draft_only_for_simple_methods = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "lmsys/vicuna-7b-v1.3",
    }
    if pair.draft in restricted_draft_only_for_simple_methods and method in {"bump", "morphable"}:
        return False
    return True


def sanitize_path_component(name: str) -> str:
    name = str(name).strip().replace("\\", "/").rstrip("/")
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or "unknown_model"


def build_model_run_dir_name(draft: str, intermediate: str, target: str) -> str:
    return (
        f"draft__{sanitize_path_component(draft)}"
        f"__intermediate__{sanitize_path_component(intermediate)}"
        f"__target__{sanitize_path_component(target)}"
    )


def short_model_tag(model_name: str) -> str:
    low = model_name.lower().strip().replace("_", "-")
    if "/" in low:
        low = low.split("/")[-1]
    low = low.replace("instruct", "").replace("chat", "").replace("-it", "").strip("-")

    llama_match = re.search(r"llama-?(?P<maj>\d)(?:\.(?P<minor>\d))?-(?P<size>\d+)(?P<unit>[bm])", low)
    if llama_match:
        maj = llama_match.group("maj")
        minor = llama_match.group("minor") or "0"
        size = llama_match.group("size")
        unit = llama_match.group("unit")
        return f"llama{maj}{minor}_{size}{unit}"

    qwen_match = re.search(r"qwen(?:-?\d+(?:\.\d+)?)?-(?P<size>\d+(?:\.\d+)?)(?P<unit>[bm])", low)
    if qwen_match:
        size = qwen_match.group("size").replace(".", "p")
        unit = qwen_match.group("unit")
        return f"qwen_{size}{unit}"

    vicuna_match = re.search(r"vicuna-(?P<size>\d+(?:\.\d+)?)(?P<unit>[bm])", low)
    if vicuna_match:
        size = vicuna_match.group("size").replace(".", "p")
        unit = vicuna_match.group("unit")
        return f"vicuna_{size}{unit}"

    generic = sanitize_path_component(model_name).lower()
    generic = generic.replace(".", "")
    generic = re.sub(r"__+", "_", generic)
    return generic


def pair_slug(pair: ModelTriplet) -> str:
    return f"{short_model_tag(pair.draft)}_{short_model_tag(pair.intermediate)}_{short_model_tag(pair.target)}"


def parse_pair(raw: str) -> ModelTriplet:
    parts = [x.strip() for x in raw.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid pair '{raw}'. Expected format: draft|intermediate|target")
    return ModelTriplet(draft=parts[0], intermediate=parts[1], target=parts[2])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_max_tokens_flag(dataset: str, max_new_tokens: int | None) -> str:
    if max_new_tokens is None:
        return ""
    arg = DATASET_TO_MAX_NEW_TOKENS_ARG.get(dataset)
    if arg is None:
        return ""
    return f"  {arg} {max_new_tokens} \\\n"


def choose_dataset_time_limit(dataset: str, mt_bench_time_limit: str, other_dataset_time_limit: str) -> str:
    if dataset == "mt_bench":
        return mt_bench_time_limit
    return other_dataset_time_limit


def shell_quote_single(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def model_family(pair: ModelTriplet) -> str:
    joined = " ".join([pair.draft, pair.intermediate, pair.target]).lower()
    if "vicuna" in joined:
        return "vicuna"
    if "llama" in joined:
        return "llama"
    return "generic"


def choose_resources(
    *,
    pair: ModelTriplet,
    llama_tp_size: int,
    llama_cpus_per_task: int,
    llama_gres: str,
    llama_mem_per_gpu: str,
    llama_time_limit: str,
    vicuna_tp_size: int,
    vicuna_cpus_per_task: int,
    vicuna_gres: str,
    vicuna_mem_per_gpu: str,
    vicuna_time_limit: str,
    generic_tp_size: int,
    generic_cpus_per_task: int,
    generic_gres: str,
    generic_mem_per_gpu: str,
    generic_time_limit: str,
) -> ResourceSpec:
    family = model_family(pair)
    if family == "vicuna":
        return ResourceSpec(
            tp_size=vicuna_tp_size,
            cpus_per_task=vicuna_cpus_per_task,
            gres=vicuna_gres,
            mem_per_gpu=vicuna_mem_per_gpu,
            time_limit=vicuna_time_limit,
        )
    if family == "llama":
        return ResourceSpec(
            tp_size=llama_tp_size,
            cpus_per_task=llama_cpus_per_task,
            gres=llama_gres,
            mem_per_gpu=llama_mem_per_gpu,
            time_limit=llama_time_limit,
        )
    return ResourceSpec(
        tp_size=generic_tp_size,
        cpus_per_task=generic_cpus_per_task,
        gres=generic_gres,
        mem_per_gpu=generic_mem_per_gpu,
        time_limit=generic_time_limit,
    )


def make_job_text(
    *,
    pair: ModelTriplet,
    dataset: str,
    k: int,
    method: str,
    batch_size: int,
    batch_tag: str,
    hispec: bool,
    interval: int,
    confidence_threshold: float,
    bonus_method: str,
    bonus_threshold: float,
    resource: ResourceSpec,
    account: str,
    qos: str,
    repo_dir: str,
    venv_dir: str,
    data_dir: str,
    job_root: str,
    out_log_root: str,
    err_log_root: str,
    result_root: str,
    profile_script: str,
    conversation_turn_mode: str,
    conversation_turn_index: int,
    max_prompt_tokens: int,
    topk: int,
    query_window_size: int,
    max_samples: int | None,
    max_new_tokens: int | None,
    hf_home_fallback_scratch: str,
    extra_args: str,
    topk_selection: int,
    expansion_pct: float,
) -> tuple[str, str, str]:
    pair_name = pair_slug(pair)
    config_tag = build_config_tag(
        method=method,
        hispec=hispec,
        interval=interval,
        topk_selection=topk_selection,
        expansion_pct=expansion_pct,
    )
    config_dir = Path(f"cfg__{config_tag}") if config_tag else Path()
    config_suffix = f"_{config_tag}" if config_tag else ""

    job_name = f"spec_{pair_name}_{dataset}_{method}_k{k}_bs{batch_size}{config_suffix}"
    model_run_dir_name = build_model_run_dir_name(pair.draft, pair.intermediate, pair.target)

    job_dir = Path(job_root) / f"method__{method}" / config_dir / f"bs{batch_size}" / f"k{k}" / pair_name
    out_log_dir = Path(out_log_root) / f"method__{method}" / config_dir / f"bs{batch_size}" / f"k{k}" / pair_name
    err_log_dir = Path(err_log_root) / f"method__{method}" / config_dir / f"bs{batch_size}" / f"k{k}" / pair_name

    result_base_root = Path(result_root) / f"method__{method}" / config_dir / f"bs{batch_size}" / f"k{k}"
    pair_result_dir = result_base_root / model_run_dir_name
    result_dataset_root = pair_result_dir / dataset

    ensure_dir(job_dir)
    ensure_dir(out_log_dir)
    ensure_dir(err_log_dir)
    ensure_dir(result_dataset_root)

    job_path = job_dir / f"{job_name}.sh"
    submit_cmd = f"sbatch {job_path}"

    launch = f"torchrun --nproc_per_node {resource.tp_size}" if resource.tp_size > 1 else "python"

    if os.path.isabs(profile_script):
        profile_assign = f"PROFILE_SCRIPT={shell_quote_single(profile_script)}"
    else:
        profile_assign = f'PROFILE_SCRIPT="${{REPO_DIR}}/{profile_script}"'

    max_samples_flag = f"  --max-samples {max_samples} \\\n" if max_samples is not None else ""
    max_new_tokens_flag = dataset_max_tokens_flag(dataset, max_new_tokens)

    topk_expansion_args_block = ""
    if method == "topk_expansion":
        topk_expansion_args_block = (
            f"  --topk-selection {topk_selection} \\\n"
            f"  --expansion-pct {expansion_pct} \\\n"
        )

    hispec_args_block = ""
    if hispec:
        hispec_args_block += "  --hispec \\\n"
    hispec_args_block += (
        f"  --interval {interval} \\\n"
        f"  --confidence-threshold {confidence_threshold} \\\n"
        f"  --bonus-method {bonus_method} \\\n"
        f"  --bonus-threshold {bonus_threshold} \\\n"
    )

    extra_args_block = ""
    if extra_args.strip():
        extra_args_block = "  " + extra_args.strip().rstrip("\\") + " \\\n"

    hispec_str = "true" if hispec else "false"
    topk_selection_echo = str(topk_selection) if method == "topk_expansion" else "N/A"
    expansion_pct_echo = str(expansion_pct) if method == "topk_expansion" else "N/A"

    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resource.cpus_per_task}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --gres={resource.gres}
#SBATCH --mem-per-gpu={resource.mem_per_gpu}
#SBATCH --time={resource.time_limit}
#SBATCH --output={out_log_dir / (job_name + '.out')}
#SBATCH --error={err_log_dir / (job_name + '.err')}

set -eo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR={shell_quote_single(repo_dir)}
VENV_DIR={shell_quote_single(venv_dir)}
DATA_DIR={shell_quote_single(data_dir)}
{profile_assign}
JOB_DIR={shell_quote_single(str(job_dir))}
OUT_LOG_DIR={shell_quote_single(str(out_log_dir))}
ERR_LOG_DIR={shell_quote_single(str(err_log_dir))}
RESULT_ROOT={shell_quote_single(str(result_base_root))}
PAIR_RESULT_DIR={shell_quote_single(str(pair_result_dir))}
RESULT_DATASET_ROOT={shell_quote_single(str(result_dataset_root))}
DATASET_NAME={shell_quote_single(dataset)}
METHOD_NAME={shell_quote_single(method)}
BATCH_SIZE={batch_size}
NUM_SPEC_TOKENS={k}
TP_SIZE={resource.tp_size}
SPECHIVE_DEBUG="${{SPECHIVE_DEBUG:-0}}"

cd "${{REPO_DIR}}"
export BASHRCSOURCED="${{BASHRCSOURCED:-1}}"
set +u
source "${{VENV_DIR}}/bin/activate"
if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi
set -u

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_TOKEN="${{HF_TOKEN:-}}"
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"

if [[ -z "${{HF_HOME:-}}" ]]; then
  if [[ -d "{hf_home_fallback_scratch}" ]]; then
    export HF_HOME="{hf_home_fallback_scratch}"
  else
    export HF_HOME="${{HOME}}/.cache/huggingface"
  fi
fi

export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-${{HF_HOME}}/hub}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-${{HUGGINGFACE_HUB_CACHE}}}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-${{HF_HOME}}/datasets}}"

mkdir -p "${{JOB_DIR}}" "${{OUT_LOG_DIR}}" "${{ERR_LOG_DIR}}" "${{PAIR_RESULT_DIR}}" "${{RESULT_DATASET_ROOT}}"

echo "============================================================"
echo "DATASET: ${{DATASET_NAME}}"
echo "METHOD: ${{METHOD_NAME}}"
echo "HISPEC: {hispec_str}"
echo "INTERVAL: {interval}"
echo "BATCH_SIZE: ${{BATCH_SIZE}}"
echo "NUM_SPEC_TOKENS: ${{NUM_SPEC_TOKENS}}"
echo "TOPK_SELECTION: {topk_selection_echo}"
echo "EXPANSION_PCT: {expansion_pct_echo}"
echo "RESULT_ROOT: ${{RESULT_ROOT}}"
echo "PAIR_RESULT_DIR: ${{PAIR_RESULT_DIR}}"
echo "RESULT_DATASET_ROOT: ${{RESULT_DATASET_ROOT}}"
echo "============================================================"

{launch} "${{PROFILE_SCRIPT}}" \\
  --draft {shell_quote_single(pair.draft)} \\
  --intermediate {shell_quote_single(pair.intermediate)} \\
  --target {shell_quote_single(pair.target)} \\
  --target-tp-size "${{TP_SIZE}}" \\
  --data-dir "${{DATA_DIR}}" \\
  --datasets "${{DATASET_NAME}}" \\
  --output-dir "${{RESULT_ROOT}}" \\
  --method "${{METHOD_NAME}}" \\
  --batch-size "${{BATCH_SIZE}}" \\
  --k "${{NUM_SPEC_TOKENS}}" \\
  --conversation-turn-mode {conversation_turn_mode} \\
  --conversation-turn-index {conversation_turn_index} \\
  --max-prompt-tokens {max_prompt_tokens} \\
  --topk {topk} \\
  --query-window-size {query_window_size} \\
{hispec_args_block}{topk_expansion_args_block}{max_samples_flag}{max_new_tokens_flag}{extra_args_block}  --print-request-progress
"""
    return str(job_path), text, submit_cmd


def iter_jobs(
    pairs: Iterable[ModelTriplet],
    datasets: list[str],
    ks: list[int],
    methods: list[str],
    batch_sizes: list[int],
):
    for pair in pairs:
        for dataset in datasets:
            for method in methods:
                for batch_size in batch_sizes:
                    for k in ks:
                        yield pair, dataset, method, batch_size, k


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Slurm job scripts for batch profile sweeps (including vanila + --hispec).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="Repeated triplets in the form draft|intermediate|target.")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS, help="Comma-separated dataset names")
    parser.add_argument("--ks", type=str, default=DEFAULT_KS, help="Comma-separated k values")
    parser.add_argument(
        "--methods",
        type=str,
        default=DEFAULT_METHODS,
        help="Comma-separated methods (vanila + --hispec is supported; use --no-hispec for baseline vanila).",
    )
    parser.add_argument("--batch-sizes", type=str, default=DEFAULT_BATCH_SIZES, help="Comma-separated batch sizes")
    parser.add_argument("--batch-tag", type=str, default="b1", help="Tag used only for directory and job naming")
    parser.add_argument("--hispec", action=argparse.BooleanOptionalAction, default=DEFAULT_HISPEC)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--bonus-method", type=str, default="adaptive", choices=["proactive", "conservative", "adaptive"])
    parser.add_argument("--bonus-threshold", type=float, default=0.8)

    parser.add_argument("--repo-dir", type=str, default=DEFAULT_REPO_DIR)
    parser.add_argument("--venv-dir", type=str, default=DEFAULT_VENV_DIR)
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Dataset jsonl root passed to the profile script")
    parser.add_argument("--profile-script", type=str, default=DEFAULT_PROFILE_SCRIPT, help="Absolute path or repo-relative path to the profiling script")
    parser.add_argument("--job-root", type=str, default=DEFAULT_JOB_ROOT)
    parser.add_argument("--out-log-root", type=str, default=DEFAULT_OUT_LOG_ROOT)
    parser.add_argument("--err-log-root", type=str, default=DEFAULT_ERR_LOG_ROOT)
    parser.add_argument("--result-root", type=str, default=DEFAULT_RESULT_ROOT)

    parser.add_argument("--llama-tp-size", type=int, default=4)
    parser.add_argument("--llama-cpus-per-task", type=int, default=12)
    parser.add_argument("--llama-gres", type=str, default="gpu:h100:4")
    parser.add_argument("--llama-mem-per-gpu", type=str, default="80G")
    parser.add_argument("--llama-time-limit", type=str, default="12:00:00")

    parser.add_argument("--vicuna-tp-size", type=int, default=1)
    parser.add_argument("--vicuna-cpus-per-task", type=int, default=12)
    parser.add_argument("--vicuna-gres", type=str, default="gpu:h100:1")
    parser.add_argument("--vicuna-mem-per-gpu", type=str, default="80G")
    parser.add_argument("--vicuna-time-limit", type=str, default="04:00:00")

    parser.add_argument("--generic-tp-size", type=int, default=4)
    parser.add_argument("--generic-cpus-per-task", type=int, default=12)
    parser.add_argument("--generic-gres", type=str, default="gpu:h100:4")
    parser.add_argument("--generic-mem-per-gpu", type=str, default="80G")
    parser.add_argument("--generic-time-limit", type=str, default="12:00:00")

    parser.add_argument("--mt-bench-time-limit", type=str, default="05:00:00", help="SBATCH time limit used only for mt_bench jobs")
    parser.add_argument("--other-dataset-time-limit", type=str, default="01:30:00", help="SBATCH time limit used for every dataset except mt_bench")

    parser.add_argument("--topk-selection", type=int, default=5, choices=[2, 5], help="Number of first-token candidates for topk_expansion")
    parser.add_argument("--expansion-pct", type=float, default=0.2, help="Fraction of lowest-confidence requests to expand for topk_expansion")

    parser.add_argument("--account", type=str, default="rrg-pnair_gpu")
    parser.add_argument("--qos", type=str, default="rrg-pnair")
    parser.add_argument("--conversation-turn-mode", type=str, default="all", choices=["all", "selected"])
    parser.add_argument("--conversation-turn-index", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--query-window-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None, help="If set, inject the dataset-specific max-new-tokens flag for each job")
    parser.add_argument("--hf-home-fallback-scratch", type=str, default="${HOME}/scratch/.cache")
    parser.add_argument("--extra-args", type=str, default="", help="Extra raw CLI fragment appended to the profiling command")
    parser.add_argument("--submit-script-name", type=str, default="submit_all.sh")
    args = parser.parse_args()

    pairs = [parse_pair(raw) for raw in args.pairs]
    datasets = parse_csv(args.datasets)
    ks = [int(x) for x in parse_csv(args.ks)]
    methods = [normalize_method(x) for x in parse_csv(args.methods)]
    batch_sizes = [int(x) for x in parse_csv(args.batch_sizes)]

    if not datasets:
        raise ValueError("--datasets must not be empty")
    if not ks:
        raise ValueError("--ks must not be empty")
    if not methods:
        raise ValueError("--methods must not be empty")
    if not batch_sizes:
        raise ValueError("--batch-sizes must not be empty")
    if any(bs < 1 for bs in batch_sizes):
        raise ValueError("--batch-sizes must all be >= 1")
    if not args.data_dir:
        raise ValueError("--data-dir is required")
    if args.interval < 0:
        raise ValueError("--interval must be >= 0")
    if not (0.0 <= args.expansion_pct <= 1.0):
        raise ValueError("--expansion-pct must be in [0, 1]")
    if args.hispec and args.interval < 1:
        raise ValueError("--hispec requires --interval >= 1")
    if args.hispec:
        for method in methods:
            if method not in {"vanila", "topk_expansion"}:
                raise ValueError("--hispec is only valid with methods vanila or topk_expansion")

    all_submit_cmds: list[str] = []
    written_jobs: list[str] = []

    for pair, dataset, method, batch_size, k in iter_jobs(pairs, datasets, ks, methods, batch_sizes):
        if not method_allowed_for_pair(pair, method):
            continue

        base_resource = choose_resources(
            pair=pair,
            llama_tp_size=args.llama_tp_size,
            llama_cpus_per_task=args.llama_cpus_per_task,
            llama_gres=args.llama_gres,
            llama_mem_per_gpu=args.llama_mem_per_gpu,
            llama_time_limit=args.llama_time_limit,
            vicuna_tp_size=args.vicuna_tp_size,
            vicuna_cpus_per_task=args.vicuna_cpus_per_task,
            vicuna_gres=args.vicuna_gres,
            vicuna_mem_per_gpu=args.vicuna_mem_per_gpu,
            vicuna_time_limit=args.vicuna_time_limit,
            generic_tp_size=args.generic_tp_size,
            generic_cpus_per_task=args.generic_cpus_per_task,
            generic_gres=args.generic_gres,
            generic_mem_per_gpu=args.generic_mem_per_gpu,
            generic_time_limit=args.generic_time_limit,
        )

        dataset_time_limit = choose_dataset_time_limit(
            dataset=dataset,
            mt_bench_time_limit=args.mt_bench_time_limit,
            other_dataset_time_limit=args.other_dataset_time_limit,
        )

        resource = ResourceSpec(
            tp_size=base_resource.tp_size,
            cpus_per_task=base_resource.cpus_per_task,
            gres=base_resource.gres,
            mem_per_gpu=base_resource.mem_per_gpu,
            time_limit=dataset_time_limit,
        )

        job_path, text, submit_cmd = make_job_text(
            pair=pair,
            dataset=dataset,
            k=k,
            method=method,
            batch_size=batch_size,
            batch_tag=args.batch_tag,
            hispec=args.hispec,
            interval=args.interval,
            confidence_threshold=args.confidence_threshold,
            bonus_method=args.bonus_method,
            bonus_threshold=args.bonus_threshold,
            resource=resource,
            account=args.account,
            qos=args.qos,
            repo_dir=args.repo_dir,
            venv_dir=args.venv_dir,
            data_dir=args.data_dir,
            job_root=args.job_root,
            out_log_root=args.out_log_root,
            err_log_root=args.err_log_root,
            result_root=args.result_root,
            profile_script=args.profile_script,
            conversation_turn_mode=args.conversation_turn_mode,
            conversation_turn_index=args.conversation_turn_index,
            max_prompt_tokens=args.max_prompt_tokens,
            topk=args.topk,
            query_window_size=args.query_window_size,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            hf_home_fallback_scratch=args.hf_home_fallback_scratch,
            extra_args=args.extra_args,
            topk_selection=args.topk_selection,
            expansion_pct=args.expansion_pct,
        )
        Path(job_path).write_text(text, encoding="utf-8")
        os.chmod(job_path, 0o755)
        written_jobs.append(job_path)
        all_submit_cmds.append(submit_cmd)

    submit_script_root = Path(args.job_root)
    ensure_dir(submit_script_root)
    submit_script_path = submit_script_root / args.submit_script_name

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    submit_lines.extend(all_submit_cmds)
    submit_script_path.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    os.chmod(submit_script_path, 0o755)

    print(f"Generated {len(written_jobs)} job scripts.")
    print(f"Submit script: {submit_script_path}")
    for job in written_jobs:
        print(job)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
