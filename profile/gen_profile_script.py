#!/usr/bin/env python3
"""
Generate Slurm job scripts that run ``bench/bench.py`` with profiling output paths.

Layout for ``--profiler_output_dir`` when ``bench.py`` is run with ``--profile`` (relative to the bench cwd)::

    ./results/<profile_mode>/<method>/<batch_size>/k<k|na>/<target>+<draft>/<temp_tag>/

Sweep flags (optional, Cartesian product with other dimensions):
  --batch   → batch sizes 1, 4, 16, 64, 256
  --length  → speculative k 3, 5, 7, 9 (sync/async only; AR keeps path segment ``kna``)
  --temp    → temperatures 0, 0.3, 0.7, 1.0

Adding a method
    1. Define ``extra_bench_args(k, async_fan_out) -> list[str]`` (tokens only; no ``--gpus``).
    2. Append ``BenchMethodSpec(...)`` to ``METHOD_REGISTRY`` with a unique ``id``.
    3. If GPU count differs by model family, extend ``DEFAULT_GPU_BY_FAMILY_METHOD``.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

# Repo imports (profile/ → bench_helpers)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror bench/bench_helpers.py (bench.py --qwen / --gemma presets); keep in sync if defaults change.
BENCH_PRESET_QWEN_TARGET = "Qwen/Qwen3-32B"
BENCH_PRESET_QWEN_DRAFT = "Qwen/Qwen3-0.6B"
BENCH_PRESET_GEMMA_TARGET = "google/gemma-4-31B-it"
BENCH_PRESET_GEMMA_DRAFT = "google/gemma-4-E4B-it"

DEFAULT_REPO_DIR = "/project/def-pnair/junsu/kv_cache/ssd"
DEFAULT_VENV_DIR = "/project/def-pnair/junsu/kv_cache/.venv"
DEFAULT_JOB_ROOT = str(_REPO_ROOT / "profile" / "scripts" / "jobs" / "bench_profile")
DEFAULT_OUT_LOG_ROOT = str(_REPO_ROOT / "profile" / "scripts" / "logs" / "bench_profile" / "out")
DEFAULT_ERR_LOG_ROOT = str(_REPO_ROOT / "profile" / "scripts" / "logs" / "bench_profile" / "err")
DEFAULT_ACCOUNT = "def-pnair"
DEFAULT_QOS = "normal"
DEFAULT_GPUS = "gpu:2"
DEFAULT_MEM_PER_GPU = "128G"
DEFAULT_TIME_LIMIT = "04:00:00"
DEFAULT_CPUS_PER_TASK = 16

BATCH_SWEEP = (1, 4, 16, 64, 256)
K_SWEEP = (3, 5, 7, 9)
TEMP_SWEEP = (0.0, 0.3, 0.7, 1.0)

FIXED_NUMSEQS = 512
FIXED_OUTPUT_LEN = 2048

PROFILE_MODE_CHOICES = ("cost", "metadata", "cost_metadata")

MODEL_PRESETS: dict[str, tuple[str, str, str]] = {
    # family -> (bench flag name, target hub id, draft hub id)
    "qwen": ("qwen", BENCH_PRESET_QWEN_TARGET, BENCH_PRESET_QWEN_DRAFT),
    "gemma": ("gemma", BENCH_PRESET_GEMMA_TARGET, BENCH_PRESET_GEMMA_DRAFT),
}

DEFAULT_GPU_BY_FAMILY_METHOD: dict[tuple[str, str], int] = {
    ("qwen", "ar"): 2,
    ("qwen", "sync"): 2,
    ("qwen", "async"): 3,
    ("gemma", "ar"): 2,
    ("gemma", "sync"): 2,
    ("gemma", "async"): 3,
}


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def sanitize_path_component(name: str) -> str:
    name = str(name).strip().replace("\\", "/").rstrip("/")
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", name)
    return name or "unknown"


def hub_tail(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1] if "/" in repo_id else repo_id


def target_plus_draft_slug(target: str, draft: str | None) -> str:
    t = sanitize_path_component(hub_tail(target))
    if draft is None:
        return f"{t}+none"
    d = sanitize_path_component(hub_tail(draft))
    return f"{t}+{d}"


def temp_path_tag(temp: float) -> str:
    s = f"{temp:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return "t" + s.replace(".", "p")


def shell_quote_single(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class BenchMethodSpec:
    """Extensible bench method: id is used in paths and Slurm job names."""

    id: str
    description: str
    uses_spec_k: bool
    default_k: int
    extra_bench_args: Callable[[int, int], list[str]]
    """
    ``extra_bench_args(k, async_fan_out)`` → extra argv tokens after model/dataset flags
    (e.g. ``['--spec', '--k', '6']``). Not including ``--gpus`` (added separately).
    """


def _args_ar(_k: int, _f: int) -> list[str]:
    return []


def _args_sync(k: int, _f: int) -> list[str]:
    return ["--spec", "--k", str(k)]


def _args_async(k: int, f: int) -> list[str]:
    return ["--spec", "--async", "--k", str(k), "--f", str(f)]


METHOD_REGISTRY: dict[str, BenchMethodSpec] = {
    "ar": BenchMethodSpec(
        id="ar",
        description="Autoregressive (no speculative decoding)",
        uses_spec_k=False,
        default_k=0,
        extra_bench_args=_args_ar,
    ),
    "sync": BenchMethodSpec(
        id="sync",
        description="Synchronous speculative decoding",
        uses_spec_k=True,
        default_k=6,
        extra_bench_args=_args_sync,
    ),
    "async": BenchMethodSpec(
        id="async",
        description="Async speculative decoding (SSD)",
        uses_spec_k=True,
        default_k=7,
        extra_bench_args=_args_async,
    ),
}


def normalize_methods(raw: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in raw.split(","):
        m = part.strip().lower()
        if not m:
            continue
        if m not in METHOD_REGISTRY:
            raise SystemExit(f"Unknown method {m!r}. Known: {', '.join(sorted(METHOD_REGISTRY))}")
        out.append(m)
    if not out:
        raise SystemExit("Empty --methods")
    return tuple(dict.fromkeys(out))  # stable unique


def normalize_model_families(raw: str) -> tuple[str, ...]:
    out: list[str] = []
    for part in raw.split(","):
        f = part.strip().lower()
        if not f:
            continue
        if f not in MODEL_PRESETS:
            raise SystemExit(f"Unknown model family {f!r}. Known: {', '.join(sorted(MODEL_PRESETS))}")
        out.append(f)
    if not out:
        raise SystemExit("Empty --models")
    return tuple(dict.fromkeys(out))


def dataset_bench_flags(dataset: str) -> list[str]:
    d = dataset.strip().lower()
    mapping: dict[str, list[str]] = {
        "humaneval": ["--humaneval"],
        "alpaca": ["--alpaca"],
        "c4": ["--c4"],
        "gsm": [],
        "ultrafeedback": ["--ultrafeedback"],
        "aime2025": ["--aime2025"],
        "livecodebench": ["--livecodebench"],
        "codeelo": ["--codeelo"],
        "math500": ["--math500"],
        "govreport": ["--govreport"],
        "random": ["--random"],
        "all": ["--all"],
    }
    if d not in mapping:
        raise SystemExit(f"Unknown dataset {dataset!r}. Known: {', '.join(sorted(mapping))}")
    return mapping[d]


def profiler_rel_dir(
    *,
    profile_mode: str,
    method_id: str,
    batch_size: int,
    k_path_token: str,
    pair_slug: str,
    temp: float,
) -> str:
    """Path relative to bench cwd (``profile_mode``: cost | metadata | cost_metadata)."""
    return os.path.join(
        "results",
        profile_mode,
        method_id,
        str(batch_size),
        k_path_token,
        pair_slug,
        temp_path_tag(temp),
    )


def build_bench_argv(
    *,
    model_flag: str,
    dataset_flags: Sequence[str],
    method: BenchMethodSpec,
    k: int,
    async_fan_out: int,
    batch_size: int,
    temp: float,
    numseqs: int,
    output_len: int,
    gpus: int,
    profile_mode: str,
    profiler_output_dir: str,
    extra_bench_args: Sequence[str],
) -> list[str]:
    argv: list[str] = [
        "python",
        "-O",
        "bench.py",
        f"--{model_flag}",
        "--gpus",
        str(gpus),
        "--b",
        str(batch_size),
        "--temp",
        str(temp),
        "--numseqs",
        str(numseqs),
        "--output_len",
        str(output_len),
        *dataset_flags,
        *method.extra_bench_args(k, async_fan_out),
        "--profile",
        "--profile_mode",
        profile_mode,
        "--profiler_output_dir",
        profiler_output_dir,
        *list(extra_bench_args),
    ]
    return argv


def format_multiline_cmd(argv: Sequence[str]) -> str:
    """Bash-safe continuation lines."""
    lines: list[str] = [f"  {shell_quote_single(argv[0])} \\"]
    for tok in argv[1:-1]:
        lines.append(f"  {shell_quote_single(tok)} \\")
    if len(argv) > 1:
        lines.append(f"  {shell_quote_single(argv[-1])}")
    else:
        lines[-1] = lines[-1].rstrip(" \\")
    return "\n".join(lines)


def make_slurm_script(
    *,
    job_name: str,
    account: str,
    qos: str,
    gres: str,
    mem_per_gpu: str,
    time_limit: str,
    cpus_per_task: int,
    out_path: Path,
    err_path: Path,
    repo_dir: Path,
    venv_dir: Path,
    bench_rel: str,
    bench_argv: Sequence[str],
    hf_home_fallback: str,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    prof_dir_token = None
    for i, tok in enumerate(bench_argv):
        if tok == "--profiler_output_dir" and i + 1 < len(bench_argv):
            prof_dir_token = bench_argv[i + 1]
            break
    mkdir_block = ""
    if prof_dir_token:
        _rel = prof_dir_token.lstrip("./").lstrip("/")
        mkdir_block = f'mkdir -p "${{BENCH_DIR}}/{_rel}"\n'

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --gres={gres}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --time={time_limit}
#SBATCH --output={out_path}
#SBATCH --error={err_path}

set -eo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR={shell_quote_single(str(repo_dir))}
VENV_DIR={shell_quote_single(str(venv_dir))}
BENCH_DIR="${{REPO_DIR}}/{bench_rel.strip('/')}"

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

if [[ -z "${{HF_HOME:-}}" ]]; then
  if [[ -d "{hf_home_fallback}" ]]; then
    export HF_HOME="{hf_home_fallback}"
  else
    export HF_HOME="${{HOME}}/.cache/huggingface"
  fi
fi

export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-${{HF_HOME}}/hub}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-${{HUGGINGFACE_HUB_CACHE}}}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-${{HF_HOME}}/datasets}}"

cd "${{BENCH_DIR}}"
{mkdir_block}
{format_multiline_cmd(bench_argv)}
"""


def iter_job_configs(
    *,
    model_families: Sequence[str],
    methods: Sequence[str],
    batch_sizes: Sequence[int],
    ks: Sequence[int],
    temps: Sequence[float],
    sweep_length: bool,
) -> Iterable[tuple[str, str, int, int, float, BenchMethodSpec]]:
    for fam in model_families:
        for mid in methods:
            spec = METHOD_REGISTRY[mid]
            k_candidates: tuple[int, ...]
            if sweep_length and spec.uses_spec_k:
                k_candidates = tuple(ks)
            elif spec.uses_spec_k:
                k_candidates = (spec.default_k,)
            else:
                k_candidates = (0,)

            for k in k_candidates:
                for b in batch_sizes:
                    for t in temps:
                        yield fam, mid, k, b, t, spec


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Slurm scripts for bench.py profiling sweeps.")
    p.add_argument("--repo-dir", type=str, default=DEFAULT_REPO_DIR)
    p.add_argument("--venv-dir", type=str, default=DEFAULT_VENV_DIR)
    p.add_argument("--bench-subdir", type=str, default="bench", help="Directory under repo containing bench.py")
    p.add_argument("--job-root", type=str, default=DEFAULT_JOB_ROOT)
    p.add_argument("--out-log-root", type=str, default=DEFAULT_OUT_LOG_ROOT)
    p.add_argument("--err-log-root", type=str, default=DEFAULT_ERR_LOG_ROOT)
    p.add_argument("--account", type=str, default=DEFAULT_ACCOUNT)
    p.add_argument("--qos", type=str, default=DEFAULT_QOS)
    p.add_argument("--gres", type=str, default=DEFAULT_GPUS)
    p.add_argument("--mem-per-gpu", type=str, default=DEFAULT_MEM_PER_GPU)
    p.add_argument("--time", type=str, default=DEFAULT_TIME_LIMIT, dest="time_limit")
    p.add_argument("--cpus-per-task", type=int, default=DEFAULT_CPUS_PER_TASK)
    p.add_argument(
        "--hf-home-fallback",
        type=str,
        default="${HOME}/scratch/huggingface",
        help="Used when HF_HOME is unset (bash expands ${HOME} at runtime if you pass it literally).",
    )

    p.add_argument("--models", type=str, default="qwen,gemma", help="Comma-separated model families")
    p.add_argument("--methods", type=str, default="ar,sync,async", help="Comma-separated method ids")
    p.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        help="humaneval|alpaca|c4|gsm|ultrafeedback|aime2025|livecodebench|codeelo|math500|govreport|random|all",
    )

    p.add_argument("--batch", action="store_true", help=f"Sweep batch sizes {BATCH_SWEEP}")
    p.add_argument("--length", action="store_true", help=f"Sweep speculative k {K_SWEEP} (sync/async)")
    p.add_argument("--temp", action="store_true", help=f"Sweep temperatures {TEMP_SWEEP}")

    p.add_argument("--batch-sizes", type=str, default="", help="Override batch sweep, e.g. '1,8,32'")
    p.add_argument("--ks", type=str, default="", help="Override k sweep, e.g. '4,6,8'")
    p.add_argument("--temps", type=str, default="", help="Override temp sweep, e.g. '0,0.5,1'")

    p.add_argument(
        "--profile-mode",
        type=str,
        default="cost_metadata",
        choices=PROFILE_MODE_CHOICES,
        help="bench.py --profile_mode (cost | metadata | cost_metadata)",
    )
    p.add_argument("--async-fan-out", type=int, default=3, help="Async method: bench.py --f (default 3)")
    p.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Override total GPUs, e.g. '2' or '3'. If empty, use built-in per (model, method) table.",
    )
    p.add_argument("--numseqs", type=int, default=FIXED_NUMSEQS)
    p.add_argument("--output-len", type=int, default=FIXED_OUTPUT_LEN)

    p.add_argument("--extra-bench-arg", action="append", default=[], help="Extra bench.py token (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    model_families = normalize_model_families(args.models)
    methods = normalize_methods(args.methods)
    dataset_flags = dataset_bench_flags(args.dataset)

    batch_sizes = (
        parse_csv_ints(args.batch_sizes) if args.batch_sizes.strip() else (BATCH_SWEEP if args.batch else (1,))
    )
    temps = parse_csv_floats(args.temps) if args.temps.strip() else (TEMP_SWEEP if args.temp else (0.0,))

    if args.ks.strip():
        ks_final = parse_csv_ints(args.ks)
    elif args.length:
        ks_final = tuple(K_SWEEP)
    else:
        ks_final = (6,)

    gpus_global: int | None = int(args.gpus) if str(args.gpus).strip() else None

    repo_dir = Path(args.repo_dir).resolve()
    venv_dir = Path(args.venv_dir).resolve()
    job_root = Path(args.job_root).resolve()
    out_log_root = Path(args.out_log_root).resolve()
    err_log_root = Path(args.err_log_root).resolve()

    n_written = 0
    for fam, mid, k_val, b_val, temp_val, spec in iter_job_configs(
        model_families=model_families,
        methods=methods,
        batch_sizes=batch_sizes,
        ks=ks_final,
        temps=temps,
        sweep_length=args.length,
    ):
        flag_name, target_hub, draft_hub = MODEL_PRESETS[fam]
        pair_slug = target_plus_draft_slug(target_hub, None if not spec.uses_spec_k else draft_hub)
        k_path = "kna" if not spec.uses_spec_k else f"k{int(k_val)}"
        k_for_bench = int(spec.default_k) if not spec.uses_spec_k else int(k_val)

        prof_rel = profiler_rel_dir(
            profile_mode=args.profile_mode,
            method_id=spec.id,
            batch_size=b_val,
            k_path_token=k_path,
            pair_slug=pair_slug,
            temp=temp_val,
        )
        prof_rel = "./" + prof_rel.replace(os.sep, "/")

        gpu_n = gpus_global if gpus_global is not None else DEFAULT_GPU_BY_FAMILY_METHOD[(fam, spec.id)]

        bench_argv = build_bench_argv(
            model_flag=flag_name,
            dataset_flags=dataset_flags,
            method=spec,
            k=k_for_bench,
            async_fan_out=int(args.async_fan_out),
            batch_size=b_val,
            temp=float(temp_val),
            numseqs=int(args.numseqs),
            output_len=int(args.output_len),
            gpus=gpu_n,
            profile_mode=args.profile_mode,
            profiler_output_dir=prof_rel,
            extra_bench_args=tuple(args.extra_bench_arg),
        )

        temp_tag = temp_path_tag(temp_val)
        job_name = f"bench_{fam}_{spec.id}_b{b_val}_{k_path}_{temp_tag}"[:64]

        rel_bits = Path(args.profile_mode) / fam / spec.id / f"b{b_val}" / k_path / pair_slug / temp_tag
        job_dir = job_root / rel_bits
        out_log_dir = out_log_root / rel_bits
        err_log_dir = err_log_root / rel_bits
        if not args.dry_run:
            ensure_dir(job_dir)
            ensure_dir(out_log_dir)
            ensure_dir(err_log_dir)

        script_path = job_dir / f"{job_name}.sh"
        text = make_slurm_script(
            job_name=job_name,
            account=args.account,
            qos=args.qos,
            gres=args.gres,
            mem_per_gpu=args.mem_per_gpu,
            time_limit=args.time_limit,
            cpus_per_task=int(args.cpus_per_task),
            out_path=out_log_dir / f"{job_name}.out",
            err_path=err_log_dir / f"{job_name}.err",
            repo_dir=repo_dir,
            venv_dir=venv_dir,
            bench_rel=args.bench_subdir,
            bench_argv=bench_argv,
            hf_home_fallback=str(args.hf_home_fallback),
        )
        if args.dry_run:
            print(f"would write: {script_path}")
        else:
            script_path.write_text(text, encoding="utf-8")
            os.chmod(script_path, 0o755)
        n_written += 1

    print(f"Generated {n_written} job script(s) under {job_root}")


if __name__ == "__main__":
    main()
