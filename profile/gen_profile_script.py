#!/usr/bin/env python3
"""
Generate Slurm job scripts that run ``bench/bench.py`` with profiling output paths.

Layout for ``--profiler_output_dir`` when ``bench.py`` is run with ``--profile`` (relative to the bench cwd)::

    ./results/<profile_mode>/<method>/b<batch>/k<k|na>/<target>+<draft>/<temp_tag>[/r<R>/][<dataset_slug>/]

For ``--spec_policy hierarchical``, ``r<R>`` is always present (``R`` = ``bench.py --round`` /
``Config.target_verify_interval``). The engine uses **fused** HV: each decode step runs ``R``
intermediate verifies plus one target verify. If ``--hv-rounds`` is omitted, generated jobs sweep ``R`` in
``{1, 2, 3}``. Job scripts and Slurm logs use ``.../<temp_tag>/r<R>/`` under ``--job-root`` / log roots.

``--batch`` and ``--length`` are independent sweep dimensions (batch sizes vs speculative ``k``).

With ``--batch`` or ``--length``, method ``pivot`` also sweeps ``--pivot_topk`` and
``--pivot_expansion_pct`` over ``{2,3,5} × {0.1,0.2,0.5}`` (``pivot_legacy`` uses CLI defaults only).

When **either** ``--batch`` / ``--length`` / ``--temp`` is set, generated jobs cover the profile dataset set
(alpaca, humaneval, gsm8k, math500, codeelo, livecodebench): profiler paths ``.../b<b>/k.../t0/r<R>/<dataset>/`` for hierarchical.
By default, **one Slurm script per dataset** (job name ``<dataset>_<family>_<method>_b<b>_<kpath>_<temp_tag>``).
With ``--all``, one Slurm script runs every dataset in a ``for`` loop. ``--dataset`` is ignored in that mode.
Job scripts and logs live under the same ``.../b<b>/k.../<pair>/t<tag>/`` layout; hierarchical adds ``r<R>/`` after ``t<tag>``.

When using ``--batch`` / ``--length`` / ``--temp`` for the multi-dataset profile sweep, companion **bench-only**
bash scripts are written: ``profile/run_batch.sh`` (``--batch``), ``profile/run_length.sh`` (``--length``),
and ``profile/run_temp.sh`` (``--temp``), with ``sleep 5`` between ``python -O bench/bench.py`` blocks.

Sweep flags (optional, Cartesian product with other dimensions):
  --batch   → batch sizes 1, 4, 16, 64, 256
  --length  → speculative k 3, 5, 7, 9 (methods with ``uses_spec_k``; AR keeps path segment ``kna``)
  --temp    → temperatures 0, 0.3, 0.7, 1.0

Default ``--methods`` is ``ar,sync,pivot`` (autoregressive, sync spec, planner pivot). Other ids
(``async``, ``hierarchical``, ``pivot_legacy``, …) remain available via ``--methods``.

Adding a method
    1. Define ``extra_bench_args(k, async_fan_out) -> list[str]`` (tokens only; no ``--gpus``).
    2. Append ``BenchMethodSpec(...)`` to ``METHOD_REGISTRY`` with a unique ``id``.
    3. If GPU count differs by model family, extend ``DEFAULT_GPU_BY_FAMILY_METHOD``.
"""
from __future__ import annotations

import argparse
from itertools import product
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence



# Repo imports (profile/ → bench_helpers)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror bench/bench_helpers.py (bench.py HF presets); keep in sync if defaults change.
# Hierarchical default intermediates (when bench ``--intermediate`` omitted): Qwen3-8B | Gemma-4-E4B-it |
# Llama-3.1-8B-Instruct | Vicuna-7B (vicuna13b_160m preset only); see ``resolve_intermediate_model_path``.
BENCH_PRESET_LLAMA_TARGET = "meta-llama/Llama-3.3-70B-Instruct"
BENCH_PRESET_LLAMA_DRAFT = "meta-llama/Llama-3.2-1B-Instruct"
BENCH_PRESET_QWEN_TARGET = "Qwen/Qwen3-32B"
BENCH_PRESET_QWEN_DRAFT = "Qwen/Qwen3-0.6B"
BENCH_PRESET_GEMMA_TARGET = "google/gemma-4-31B-it"
BENCH_PRESET_GEMMA_DRAFT = "google/gemma-4-E4B-it"
BENCH_PRESET_VICUNA13B_160M_TARGET = "lmsys/vicuna-13b-v1.3"
BENCH_PRESET_VICUNA13B_160M_DRAFT = "double7/vicuna-160m"

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

# BATCH_SWEEP = (1, 2, 4, 8, 16)

BATCH_SWEEP = (16, 32)

K_SWEEP = (3, 5, 7, 9, 11)
TEMP_SWEEP = (0.0, 0.3, 0.7, 1.0)

# Hierarchical: default sweep for ``bench.py --round`` when ``--hv-rounds`` is not passed.
HV_ROUND_SWEEP_DEFAULT = (1, 2, 3)

FIXED_NUMSEQS = 128
FIXED_OUTPUT_LEN = 512

PROFILE_MODE_CHOICES = ("cost", "metadata", "cost_metadata")

# Multi-dataset profile sweep datasets used by --batch / --length / --temp.
MULTI_DATASET_PROFILE_SLUGS: tuple[str, ...] = (
    "alpaca",
    "qa",
    # "humaneval",
    # "gsm8k",
    "codeelo",
    # "aime",
    "math500",
    # "livecodebench",
)

MODEL_PRESETS: dict[str, tuple[str, str, str]] = {
    # family -> (bench flag name, target hub id, draft hub id)
    "llama": ("llama", BENCH_PRESET_LLAMA_TARGET, BENCH_PRESET_LLAMA_DRAFT),
    "qwen": ("qwen", BENCH_PRESET_QWEN_TARGET, BENCH_PRESET_QWEN_DRAFT),
    "gemma": ("gemma", BENCH_PRESET_GEMMA_TARGET, BENCH_PRESET_GEMMA_DRAFT),
    "vicuna13b_160m": (
        "vicuna13b_160m",
        BENCH_PRESET_VICUNA13B_160M_TARGET,
        BENCH_PRESET_VICUNA13B_160M_DRAFT,
    ),
}

DEFAULT_GPU_BY_FAMILY_METHOD: dict[tuple[str, str], int] = {
    ("llama", "ar"): 4,
    ("llama", "sync"): 4,
    ("llama", "async"): 4,
    ("llama", "hierarchical"): 4,
    ("llama", "pivot"): 4,
    ("llama", "pivot_precollapse"): 4,
    ("llama", "pivot_legacy"): 4,
    ("qwen", "ar"): 2,
    ("qwen", "sync"): 2,
    ("qwen", "async"): 3,
    ("qwen", "hierarchical"): 2,
    ("qwen", "pivot"): 2,
    ("qwen", "pivot_precollapse"): 2,
    ("qwen", "pivot_legacy"): 3,
    ("gemma", "ar"): 2,
    ("gemma", "sync"): 2,
    ("gemma", "async"): 3,
    ("gemma", "hierarchical"): 2,
    ("gemma", "pivot"): 2,
    ("gemma", "pivot_precollapse"): 2,
    ("gemma", "pivot_legacy"): 3,
    ("vicuna13b_160m", "ar"): 2,
    ("vicuna13b_160m", "sync"): 2,
    ("vicuna13b_160m", "async"): 3,
    ("vicuna13b_160m", "hierarchical"): 2,
    ("vicuna13b_160m", "pivot"): 2,
    ("vicuna13b_160m", "pivot_precollapse"): 2,
    ("vicuna13b_160m", "pivot_legacy"): 3,
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


def pct_path_component(pct: float) -> str:
    s = f"{pct:.6f}".rstrip("0").rstrip(".")
    if not s:
        s = "0"
    return "pct" + s.replace(".", "p")


def uses_pivot_profiler_layout(method_id: str) -> bool:
    return method_id in {"pivot", "pivot_precollapse", "pivot_legacy"}


# With ``--batch`` / ``--length`` and method ``pivot`` only (not ``pivot_legacy``).
PIVOT_BATCH_LENGTH_TOPK_SWEEP = (2, 3, 5)
PIVOT_BATCH_LENGTH_EXPANSION_PCT_SWEEP = (0.1, 0.2, 0.5)
PIVOT_STATIC_TOPK_SWEEP = (2, 3, 4, 5)
PIVOT_STATIC_EXPANSION_PCT = 1.0
PIVOT_STATIC_POLICY = "static"


def uses_pivot_batch_length_topk_pct_sweep(method_id: str) -> bool:
    return method_id in {"pivot", "pivot_precollapse"}


def iter_pivot_topk_pct_for_profile_sweep(
    *,
    multi_dataset_sweep: bool,
    method_id: str,
    cli_topk: int,
    cli_pct: float,
    static_mode: bool = False,
) -> tuple[tuple[int, float], ...]:
    if static_mode and method_id in {"pivot", "pivot_precollapse"}:
        return tuple((int(tk), float(PIVOT_STATIC_EXPANSION_PCT)) for tk in PIVOT_STATIC_TOPK_SWEEP)
    if multi_dataset_sweep and uses_pivot_batch_length_topk_pct_sweep(method_id):
        return tuple(
            (int(tk), float(pc)) for tk in PIVOT_BATCH_LENGTH_TOPK_SWEEP for pc in PIVOT_BATCH_LENGTH_EXPANSION_PCT_SWEEP
        )
    return ((int(cli_topk), float(cli_pct)),)


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


def _args_hierarchical(k: int, _f: int) -> list[str]:
    return ["--spec", "--k", str(k), "--spec_policy", "hierarchical"]


def _args_pivot(k: int, f: int) -> list[str]:
    return ["--spec", "--k", str(k), "--spec_policy", "pivot"]


def _args_pivot_precollapse(k: int, _f: int) -> list[str]:
    """Sync EAGLE3 + pivot_precollapse (bench enables --eagle only without --async for this policy)."""
    return [
        "--spec",
        "--k",
        str(k),
        "--spec_policy",
        "pivot_precollapse",
        "--eagle",
    ]


def _args_pivot_legacy(k: int, f: int) -> list[str]:
    return [
        "--spec",
        "--async",
        "--k",
        str(k),
        "--f",
        str(f),
        "--spec_policy",
        "pivot_legacy",
        "--spec_hive",
    ]


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
        default_k=5,
        extra_bench_args=_args_sync,
    ),
    "async": BenchMethodSpec(
        id="async",
        description="Async speculative decoding (SSD)",
        uses_spec_k=True,
        default_k=5,
        extra_bench_args=_args_async,
    ),
    "hierarchical": BenchMethodSpec(
        id="hierarchical",
        description="Sync spec with hierarchical verification (intermediate until hv_round_idx==R, then target)",
        uses_spec_k=True,
        default_k=5,
        extra_bench_args=_args_hierarchical,
    ),
    "pivot": BenchMethodSpec(
        id="pivot",
        description="Planner pivot policy (--spec_policy pivot)",
        uses_spec_k=True,
        default_k=5,
        extra_bench_args=_args_pivot,
    ),
    "pivot_precollapse": BenchMethodSpec(
        id="pivot_precollapse",
        description="Sync pivot_precollapse with EAGLE3 draft (--spec_policy pivot_precollapse --eagle)",
        uses_spec_k=True,
        default_k=5,
        extra_bench_args=_args_pivot_precollapse,
    ),
    "pivot_legacy": BenchMethodSpec(
        id="pivot_legacy",
        description="Async legacy pivot policy (--spec_policy pivot_legacy --spec_hive)",
        uses_spec_k=True,
        default_k=5,
        extra_bench_args=_args_pivot_legacy,
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
        "gsm8k": [],  # alias for GSM-style default (no extra dataset flag)
        "ultrafeedback": ["--ultrafeedback"],
        "aime2025": ["--aime2025"],
        "aime": ["--aime"],
        "livecodebench": ["--livecodebench"],
        "codeelo": ["--codeelo"],
        "math500": ["--math500"],
        "govreport": ["--govreport"],
        "qa": ["--qa"],
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
    dataset_slug: str | None = None,
    hv_round: int | None = None,
    pivot_round: str = "na",
    pivot_topk: int = 5,
    pivot_expansion_pct: float = 0.2,
    pivot_expansion_policy: str = "dynamic",
) -> str:
    """Path relative to bench cwd (``profile_mode``: cost | metadata | cost_metadata).

    When ``hv_round`` is set (hierarchical jobs), insert ``r<hv_round>/`` (``--round`` value) after
    the temperature tag and before the optional dataset slug.

    For ``pivot``, results go under
    ``.../pivot/<policy>/b*/k*/<pair>/<t*temp>/r_<round>/topk<k>/<pct*>/[<dataset>]``.
    For ``pivot_legacy``, keep the legacy method layout.
    """
    base = [
        "results",
        profile_mode,
        method_id,
        f"b{int(batch_size)}",
        k_path_token,
        pair_slug,
    ]
    if method_id == "pivot":
        parts = [
            "results",
            profile_mode,
            "pivot",
            sanitize_path_component(pivot_expansion_policy),
            f"b{int(batch_size)}",
            k_path_token,
            pair_slug,
            temp_path_tag(temp),
            f"r_{sanitize_path_component(str(pivot_round))}",
            f"topk{int(pivot_topk)}",
            pct_path_component(float(pivot_expansion_pct)),
        ]
        if dataset_slug:
            parts.append(sanitize_path_component(dataset_slug))
        return os.path.join(*parts)
    if method_id == "pivot_precollapse":
        parts = [
            "results",
            profile_mode,
            "pivot_precollapse",
            sanitize_path_component(pivot_expansion_policy),
            f"b{int(batch_size)}",
            k_path_token,
            pair_slug,
            temp_path_tag(temp),
            f"r_{sanitize_path_component(str(pivot_round))}",
            f"topk{int(pivot_topk)}",
            pct_path_component(float(pivot_expansion_pct)),
        ]
        if dataset_slug:
            parts.append(sanitize_path_component(dataset_slug))
        return os.path.join(*parts)
    if method_id == "pivot_legacy":
        parts = [
            *base,
            temp_path_tag(temp),
            f"r_{sanitize_path_component(str(pivot_round))}",
            f"topk{int(pivot_topk)}",
            pct_path_component(float(pivot_expansion_pct)),
        ]
        if dataset_slug:
            parts.append(sanitize_path_component(dataset_slug))
        return os.path.join(*parts)
    parts = [*base, temp_path_tag(temp)]
    if hv_round is not None:
        parts.append(f"r{int(hv_round)}")
    if dataset_slug:
        parts.append(sanitize_path_component(dataset_slug))
    return os.path.join(*parts)


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
    hv_target_verify_interval: int | None = None,
    pivot_topk: int | None = None,
    pivot_expansion_pct: float | None = None,
    pivot_expansion_policy: str | None = None,
) -> list[str]:
    argv: list[str] = [
        "python",
        "-O",
        "bench/bench.py",
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
    ]
    if method.id == "hierarchical" and hv_target_verify_interval is not None:
        argv.extend(["--round", str(int(hv_target_verify_interval))])
    if uses_pivot_profiler_layout(method.id):
        if pivot_topk is None or pivot_expansion_pct is None:
            raise ValueError("pivot_topk and pivot_expansion_pct required for pivot profiler layout")
        argv.extend(
            [
                "--pivot_topk",
                str(int(pivot_topk)),
                "--pivot_expansion_pct",
                str(float(pivot_expansion_pct)),
            ]
        )
        if pivot_expansion_policy is not None:
            argv.extend(["--pivot_expansion_policy", str(pivot_expansion_policy)])
    argv.extend(
        [
            "--profile",
            "--profile_mode",
            profile_mode,
            "--profiler_output_dir",
            profiler_output_dir,
            *list(extra_bench_args),
        ]
    )
    return argv


def format_multiline_cmd(argv: Sequence[str]) -> str:
    """Render bash command with '\' continuations.

    Style:
      python -O bench/bench.py \
        --qwen \
        --gpus 2 \
        --b 1 \
        --temp 0.0 \
        --numseqs 512 \
        --output_len 2048 \
        --humaneval
    """
    if not argv:
        return ""

    def _fmt(tok: str) -> str:
        q = shlex.quote(str(tok))
        return tok if q == tok else q

    lines: list[str] = []

    i = 0
    first = True
    while i < len(argv):
        tok = argv[i]

        if first and len(argv) >= 3:
            head = " ".join(_fmt(x) for x in argv[:3])
            lines.append(head + (" \\" if len(argv) > 3 else ""))
            i = 3
            first = False
            continue

        if tok.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            part = f"  {_fmt(tok)} {_fmt(argv[i + 1])}"
            i += 2
        else:
            part = f"  {_fmt(tok)}"
            i += 1

        if i < len(argv):
            part += " \\"
        lines.append(part)

    return "\n".join(lines)


def profiler_mkdirs_block_for_datasets(
    *,
    profile_mode: str,
    method_id: str,
    batch_size: int,
    k_path_token: str,
    pair_slug: str,
    temp: float,
    dataset_slugs: Sequence[str],
    hv_round: int | None = None,
    pivot_round: str = "na",
    pivot_topk: int = 5,
    pivot_expansion_pct: float = 0.2,
    pivot_expansion_policy: str = "dynamic",
) -> str:
    lines: list[str] = []
    for ds in dataset_slugs:
        rel = profiler_rel_dir(
            profile_mode=profile_mode,
            method_id=method_id,
            batch_size=batch_size,
            k_path_token=k_path_token,
            pair_slug=pair_slug,
            temp=temp,
            dataset_slug=ds,
            hv_round=hv_round,
            pivot_round=pivot_round,
            pivot_topk=pivot_topk,
            pivot_expansion_pct=pivot_expansion_pct,
            pivot_expansion_policy=pivot_expansion_policy,
        ).replace(os.sep, "/")
        lines.append(f'mkdir -p "${{BENCH_DIR}}/{rel}"')
    return ("\n".join(lines) + "\n") if lines else ""


def build_multi_dataset_profile_loop_sh(
    *,
    model_flag: str,
    method: BenchMethodSpec,
    k_bench: int,
    async_fan_out: int,
    batch_size: int,
    temp: float,
    numseqs: int,
    output_len: int,
    gpus: int,
    profile_mode: str,
    profiler_base_rel: str,
    extra_bench_args: Sequence[str],
    hv_target_verify_interval: int | None = None,
    pivot_topk: int | None = None,
    pivot_expansion_pct: float | None = None,
    pivot_expansion_policy: str | None = None,
) -> str:
    """Bash loop: ``--profiler_output_dir`` = ``$PROFILE_BASE/$dataset`` (``dataset`` in MULTI_DATASET_PROFILE_SLUGS)."""
    prof_base_q = shell_quote_single("./" + profiler_base_rel.replace(os.sep, "/"))
    body_lines: list[str] = [
        f"PROFILE_BASE={prof_base_q}",
        "for dataset in " + " ".join(MULTI_DATASET_PROFILE_SLUGS) + "; do",
        '  echo "==== bench profile dataset=${dataset} ===="',
        "  case \"${dataset}\" in",
        "    alpaca) EXTRA_DS=(--alpaca);;",
        "    qa) EXTRA_DS=(--qa);;",
        "    humaneval) EXTRA_DS=(--humaneval);;",
        "    gsm8k) EXTRA_DS=();;",
        "    math500) EXTRA_DS=(--math500);;",
        "    codeelo) EXTRA_DS=(--codeelo);;",
        "    aime) EXTRA_DS=(--aime);;",
        "    livecodebench) EXTRA_DS=(--livecodebench);;",
        '    *) echo "unknown dataset: ${dataset}" >&2; exit 1;;',
        "  esac",
        "  python -O bench/bench.py \\",
        f"    --{model_flag} \\",
        f"    --gpus {int(gpus)} \\",
        f"    --b {int(batch_size)} \\",
        f"    --temp {float(temp)} \\",
        f"    --numseqs {int(numseqs)} \\",
        f"    --output_len {int(output_len)} \\",
        '    "${EXTRA_DS[@]}" \\',
    ]
    for tok in method.extra_bench_args(k_bench, async_fan_out):
        body_lines.append(f"    {shlex.quote(str(tok))} \\")
    if method.id == "hierarchical" and hv_target_verify_interval is not None:
        body_lines.append(f"    --round {int(hv_target_verify_interval)} \\")
    if uses_pivot_profiler_layout(method.id):
        if pivot_topk is None or pivot_expansion_pct is None:
            raise ValueError("pivot_topk and pivot_expansion_pct required for pivot multi-dataset loop")
        body_lines.append(f"    --pivot_topk {int(pivot_topk)} \\")
        body_lines.append(f"    --pivot_expansion_pct {float(pivot_expansion_pct)} \\")
        if pivot_expansion_policy is not None:
            body_lines.append(f"    --pivot_expansion_policy {shlex.quote(str(pivot_expansion_policy))} \\")
    for tok in extra_bench_args:
        body_lines.append(f"    {shlex.quote(str(tok))} \\")
    body_lines += [
        "    --profile \\",
        f"    --profile_mode {shlex.quote(profile_mode)} \\",
        '    --profiler_output_dir "${PROFILE_BASE}/${dataset}"',
        "done",
    ]
    return "\n".join(body_lines)


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
    optional_env_before_bench: str = "",
    custom_bench_body: str | None = None,
    mkdir_block_override: str | None = None,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    prof_dir_token = None
    for i, tok in enumerate(bench_argv):
        if tok == "--profiler_output_dir" and i + 1 < len(bench_argv):
            prof_dir_token = bench_argv[i + 1]
            break
    if mkdir_block_override is not None:
        mkdir_block = mkdir_block_override
    else:
        mkdir_block = ""
        if prof_dir_token:
            _rel = prof_dir_token.lstrip("./").lstrip("/")
            mkdir_block = f'mkdir -p "${{BENCH_DIR}}/{_rel}"\n'

    bench_invocation = custom_bench_body if custom_bench_body is not None else format_multiline_cmd(bench_argv)

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

{optional_env_before_bench}cd "${{BENCH_DIR}}"
{mkdir_block}
{bench_invocation}
"""


def iter_job_configs(
    *,
    model_families: Sequence[str],
    methods: Sequence[str],
    batch_sizes: Sequence[int],
    ks: Sequence[int],
    temps: Sequence[float],
    sweep_length: bool,
    explicit_ks: Sequence[int] | None = None,
) -> Iterable[tuple[str, str, int, int, float, BenchMethodSpec]]:
    for fam in model_families:
        for mid in methods:
            spec = METHOD_REGISTRY[mid]
            k_candidates: tuple[int, ...]
            if explicit_ks is not None and spec.uses_spec_k:
                k_candidates = tuple(int(k) for k in explicit_ks)
            elif sweep_length and spec.uses_spec_k:
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

    p.add_argument(
        "--models",
        type=str,
        default="qwen,llama",
        help="Comma-separated model families (qwen | gemma | vicuna13b_160m | llama)",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="pivot",
        help="Comma-separated method ids: ar | sync | async | hierarchical | pivot | pivot_legacy "
        "(default ar,sync,pivot; pivot is sync planner policy; pivot_legacy keeps async + spec_hive path).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        help="humaneval|alpaca|c4|gsm|ultrafeedback|aime2025|aime|livecodebench|codeelo|math500|govreport|qa|random|all",
    )

    p.add_argument(
        "--batch",
        action="store_true",
        help=f"Sweep batch sizes {BATCH_SWEEP}. With --batch/--length/--temp, jobs use the five profile datasets "
        f"(one Slurm script per dataset unless --all).",
    )
    p.add_argument(
        "--length",
        action="store_true",
        help=f"Sweep speculative k {K_SWEEP} (methods with uses_spec_k). With --batch/--length/--temp, same multi-dataset "
        f"behaviour as --batch.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        dest="multids_all_in_one",
        help="With --batch/--length/--temp: emit one Slurm script that loops all profile datasets. "
        "Default (omit --all): one Slurm script per dataset.",
    )
    p.add_argument("--temp", action="store_true", help=f"Sweep temperatures {TEMP_SWEEP}")
    p.add_argument(
        "--static",
        action="store_true",
        help="Pivot static expansion preset: --pivot_expansion_policy static, --pivot_expansion_pct 1.0, "
        "pivot topk sweep 2,3,4,5, and defaults b=16/k=5 when not overridden.",
    )

    p.add_argument("--batch-sizes", type=str, default="", help="Override batch sweep, e.g. '1,8,32'")
    p.add_argument("--ks", type=str, default="", help="Override k sweep, e.g. '4,6,8'")
    p.add_argument("--temps", type=str, default="", help="Override temp sweep, e.g. '0,0.5,1'")
    p.add_argument(
        "--hv-rounds",
        type=str,
        default="",
        help="Hierarchical method only: comma-separated bench.py --round values (>=1). Each R adds "
        ".../t<tag>/r<R>/ to job, log, and profiler paths. Default when empty: sweep 1,2,3.",
    )
    p.add_argument(
        "--pivot-topk",
        type=int,
        default=5,
        help="pivot / pivot_legacy: bench.py --pivot_topk and profiler path segment topk<N> (default 5).",
    )
    p.add_argument(
        "--pivot-expansion-pct",
        type=float,
        default=0.2,
        help="pivot / pivot_legacy: bench.py --pivot_expansion_pct and profiler pct* segment (default 0.2).",
    )
    p.add_argument(
        "--pivot-expansion-policy",
        type=str,
        choices=("static", "dynamic"),
        default="dynamic",
        help="pivot / pivot_legacy: bench.py --pivot_expansion_policy (default dynamic).",
    )
    p.add_argument(
        "--pivot-profiler-round",
        type=str,
        default="na",
        help="pivot / pivot_legacy: profiler directory r_<value> after t<temp> (not bench --round). Default na.",
    )

    p.add_argument(
        "--profile-mode",
        type=str,
        default="cost",
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
    p.add_argument(
        "--run-batch-sh",
        type=str,
        default=str(_REPO_ROOT / "profile" / "run_batch.sh"),
        help="With --batch (multi-dataset sweep): write this bash script with bench.py commands only "
        "(sleep 5 between each). Ignored if --batch is not set.",
    )
    p.add_argument(
        "--run-length-sh",
        type=str,
        default=str(_REPO_ROOT / "profile" / "run_length.sh"),
        help="With --length (multi-dataset sweep): write this bash script with bench.py commands only "
        "(sleep 5 between each). Ignored if --length is not set.",
    )
    p.add_argument(
        "--run-temp-sh",
        type=str,
        default=str(_REPO_ROOT / "profile" / "run_temp.sh"),
        help="With --temp: write this bash script with bench.py commands only "
        "(sleep 5 between each).",
    )
    p.add_argument(
        "--run-pivot-static-sh",
        type=str,
        default=str(_REPO_ROOT / "profile" / "run_pivot_static.sh"),
        help="With --static: write this bash script with pivot static bench commands only "
        "(sleep 5 between each).",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if int(args.pivot_topk) < 1:
        p.error("--pivot-topk must be >= 1")

    if str(args.hv_rounds).strip():
        hv_rounds_parsed = parse_csv_ints(args.hv_rounds)
        if any(r < 1 for r in hv_rounds_parsed):
            p.error("--hv-rounds values must be >= 1")
    else:
        hv_rounds_parsed = HV_ROUND_SWEEP_DEFAULT

    model_families = normalize_model_families(args.models)
    methods = normalize_methods(args.methods)
    if args.static and "pivot" not in methods:
        print("--static enabled: adding method 'pivot' automatically.", file=sys.stderr)
        methods = tuple(dict.fromkeys([*methods, "pivot"]))
    multi_dataset_sweep = bool(args.batch or args.length or args.temp or args.static)
    if multi_dataset_sweep:
        if args.multids_all_in_one:
            print(
                "batch/length/temp sweep + --all: one Slurm script per sweep cell runs "
                f"{', '.join(MULTI_DATASET_PROFILE_SLUGS)} in a loop; --dataset ignored.",
                file=sys.stderr,
            )
        else:
            print(
                "batch/length/temp sweep: one Slurm script per dataset "
                f"({', '.join(MULTI_DATASET_PROFILE_SLUGS)}); use --all for a single loop script. "
                "--dataset ignored.",
                file=sys.stderr,
            )
    dataset_flags = dataset_bench_flags(args.dataset)

    batch_sizes = (
        parse_csv_ints(args.batch_sizes)
        if args.batch_sizes.strip()
        else (BATCH_SWEEP if args.batch else ((16,) if args.static else (1,)))
    )
    temps = parse_csv_floats(args.temps) if args.temps.strip() else (TEMP_SWEEP if args.temp else (0.0,))

    explicit_ks: tuple[int, ...] | None = None
    if args.ks.strip():
        explicit_ks = parse_csv_ints(args.ks)
    if args.length:
        ks_final = tuple(K_SWEEP)
    else:
        ks_final = (5,) if args.static else (6,)

    pivot_policy_for_run = PIVOT_STATIC_POLICY if args.static else str(args.pivot_expansion_policy)
    pivot_pct_for_run = float(PIVOT_STATIC_EXPANSION_PCT) if args.static else float(args.pivot_expansion_pct)

    gpus_global: int | None = int(args.gpus) if str(args.gpus).strip() else None

    repo_dir = Path(args.repo_dir).resolve()
    venv_dir = Path(args.venv_dir).resolve()
    job_root = Path(args.job_root).resolve()
    out_log_root = Path(args.out_log_root).resolve()
    err_log_root = Path(args.err_log_root).resolve()

    n_written = 0
    run_batch_chunks: list[list[str]] = []
    run_length_chunks: list[list[str]] = []
    run_temp_chunks: list[list[str]] = []
    run_pivot_static_chunks: list[list[str]] = []

    def _record_run_script_argv(argv_cmd: list[str], *, method_id: str) -> None:
        if args.batch:
            run_batch_chunks.append(argv_cmd)
        if args.length:
            run_length_chunks.append(argv_cmd)
        if args.temp:
            run_temp_chunks.append(argv_cmd)
        if args.static and method_id in {"pivot", "pivot_precollapse"}:
            run_pivot_static_chunks.append(argv_cmd)

    for fam, mid, k_val, b_val, temp_val, spec in iter_job_configs(
        model_families=model_families,
        methods=methods,
        batch_sizes=batch_sizes,
        ks=ks_final,
        temps=temps,
        sweep_length=args.length,
        explicit_ks=explicit_ks,
    ):
        hv_round_cells: tuple[int | None, ...] = (
            tuple(int(r) for r in hv_rounds_parsed) if spec.id == "hierarchical" else (None,)
        )
        pivot_topk_pct_cells = iter_pivot_topk_pct_for_profile_sweep(
            multi_dataset_sweep=multi_dataset_sweep,
            method_id=spec.id,
            cli_topk=int(args.pivot_topk),
            cli_pct=float(pivot_pct_for_run),
            static_mode=bool(args.static),
        )
        for hv_round, (pivot_topk_val, pivot_pct_val) in product(
            hv_round_cells,
            pivot_topk_pct_cells,
        ):
            flag_name, target_hub, draft_hub = MODEL_PRESETS[fam]
            pair_slug = target_plus_draft_slug(target_hub, None if not spec.uses_spec_k else draft_hub)
            k_path = "kna" if not spec.uses_spec_k else f"k{int(k_val)}"
            k_for_bench = int(spec.default_k) if not spec.uses_spec_k else int(k_val)

            prof_hv_kw = int(hv_round) if hv_round is not None else None

            pivot_prof_kwargs = dict(
                pivot_round=str(args.pivot_profiler_round),
                pivot_topk=int(pivot_topk_val),
                pivot_expansion_pct=float(pivot_pct_val),
                pivot_expansion_policy=str(pivot_policy_for_run),
            )

            prof_base_rel = profiler_rel_dir(
                profile_mode=args.profile_mode,
                method_id=spec.id,
                batch_size=b_val,
                k_path_token=k_path,
                pair_slug=pair_slug,
                temp=temp_val,
                dataset_slug=None,
                hv_round=prof_hv_kw,
                **pivot_prof_kwargs,
            )
            prof_rel = "./" + prof_base_rel.replace(os.sep, "/")

            gpu_n = gpus_global if gpus_global is not None else DEFAULT_GPU_BY_FAMILY_METHOD[(fam, spec.id)]

            pivot_bench_topk = int(pivot_topk_val) if uses_pivot_profiler_layout(spec.id) else None
            pivot_bench_pct = float(pivot_pct_val) if uses_pivot_profiler_layout(spec.id) else None

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
                hv_target_verify_interval=hv_round,
                pivot_topk=pivot_bench_topk,
                pivot_expansion_pct=pivot_bench_pct,
                pivot_expansion_policy=(pivot_policy_for_run if uses_pivot_profiler_layout(spec.id) else None),
            )

            temp_tag = temp_path_tag(temp_val)
            r_tag = f"r{int(hv_round)}" if hv_round is not None else ""

            if uses_pivot_profiler_layout(spec.id):
                rel_bits = (
                    Path(args.profile_mode)
                    / fam
                    / ("pivot" if spec.id == "pivot" else spec.id)
                    / (
                        sanitize_path_component(str(pivot_policy_for_run))
                        if spec.id in {"pivot", "pivot_precollapse"}
                        else ""
                    )
                    / f"b{b_val}"
                    / k_path
                    / pair_slug
                    / temp_tag
                    / f"r_{sanitize_path_component(str(args.pivot_profiler_round))}"
                    / f"topk{int(pivot_topk_val)}"
                    / pct_path_component(float(pivot_pct_val))
                )
                if spec.id == "pivot_legacy":
                    rel_bits = (
                        Path(args.profile_mode)
                        / spec.id
                        / f"b{b_val}"
                        / k_path
                        / pair_slug
                        / temp_tag
                        / f"r_{sanitize_path_component(str(args.pivot_profiler_round))}"
                        / f"topk{int(pivot_topk_val)}"
                        / pct_path_component(float(pivot_pct_val))
                    )
            else:
                rel_bits = Path(args.profile_mode) / fam / spec.id / f"b{b_val}" / k_path / pair_slug / temp_tag
                if hv_round is not None:
                    rel_bits = rel_bits / r_tag
            job_dir = job_root / rel_bits
            out_log_dir = out_log_root / rel_bits
            err_log_dir = err_log_root / rel_bits
            if not args.dry_run:
                ensure_dir(job_dir)
                ensure_dir(out_log_dir)
                ensure_dir(err_log_dir)

            optional_env = ""
            if spec.id == "hierarchical":
                optional_env = (
                    'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n'
                )

            if multi_dataset_sweep and args.multids_all_in_one:
                job_name = f"bench_{fam}_{spec.id}_b{b_val}_{k_path}_{temp_tag}{'_' + r_tag if r_tag else ''}_multids_bl"[:64]
                script_path = job_dir / f"{job_name}.sh"
                mkdir_ov = profiler_mkdirs_block_for_datasets(
                    profile_mode=args.profile_mode,
                    method_id=spec.id,
                    batch_size=b_val,
                    k_path_token=k_path,
                    pair_slug=pair_slug,
                    temp=temp_val,
                    dataset_slugs=MULTI_DATASET_PROFILE_SLUGS,
                    hv_round=prof_hv_kw,
                    **pivot_prof_kwargs,
                )
                custom_body = build_multi_dataset_profile_loop_sh(
                    model_flag=flag_name,
                    method=spec,
                    k_bench=k_for_bench,
                    async_fan_out=int(args.async_fan_out),
                    batch_size=b_val,
                    temp=float(temp_val),
                    numseqs=int(args.numseqs),
                    output_len=int(args.output_len),
                    gpus=gpu_n,
                    profile_mode=args.profile_mode,
                    profiler_base_rel=prof_base_rel,
                    extra_bench_args=tuple(args.extra_bench_arg),
                    hv_target_verify_interval=hv_round,
                    pivot_topk=pivot_bench_topk,
                    pivot_expansion_pct=pivot_bench_pct,
                    pivot_expansion_policy=(pivot_policy_for_run if uses_pivot_profiler_layout(spec.id) else None),
                )
                for ds in MULTI_DATASET_PROFILE_SLUGS:
                    ds_rb = dataset_bench_flags(ds)
                    prof_rel_rb = "./" + profiler_rel_dir(
                        profile_mode=args.profile_mode,
                        method_id=spec.id,
                        batch_size=b_val,
                        k_path_token=k_path,
                        pair_slug=pair_slug,
                        temp=temp_val,
                        dataset_slug=ds,
                        hv_round=prof_hv_kw,
                        **pivot_prof_kwargs,
                    ).replace(os.sep, "/")
                    _record_run_script_argv(
                        build_bench_argv(
                            model_flag=flag_name,
                            dataset_flags=ds_rb,
                            method=spec,
                            k=k_for_bench,
                            async_fan_out=int(args.async_fan_out),
                            batch_size=b_val,
                            temp=float(temp_val),
                            numseqs=int(args.numseqs),
                            output_len=int(args.output_len),
                            gpus=gpu_n,
                            profile_mode=args.profile_mode,
                            profiler_output_dir=prof_rel_rb,
                            extra_bench_args=tuple(args.extra_bench_arg),
                            hv_target_verify_interval=hv_round,
                            pivot_topk=pivot_bench_topk,
                            pivot_expansion_pct=pivot_bench_pct,
                            pivot_expansion_policy=(pivot_policy_for_run if uses_pivot_profiler_layout(spec.id) else None),
                        )
                        ,
                        method_id=spec.id
                    )
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
                    optional_env_before_bench=optional_env,
                    custom_bench_body=custom_body,
                    mkdir_block_override=mkdir_ov,
                )
                if args.dry_run:
                    print(f"would write: {script_path}")
                else:
                    script_path.write_text(text, encoding="utf-8")
                    os.chmod(script_path, 0o755)
                n_written += 1
            elif multi_dataset_sweep:
                for ds in MULTI_DATASET_PROFILE_SLUGS:
                    ds_flags = dataset_bench_flags(ds)
                    prof_rel_ds = "./" + profiler_rel_dir(
                        profile_mode=args.profile_mode,
                        method_id=spec.id,
                        batch_size=b_val,
                        k_path_token=k_path,
                        pair_slug=pair_slug,
                        temp=temp_val,
                        dataset_slug=ds,
                        hv_round=prof_hv_kw,
                        **pivot_prof_kwargs,
                    ).replace(os.sep, "/")
                    job_name = f"{ds}_{fam}_{spec.id}_b{b_val}_{k_path}_{temp_tag}{'_' + r_tag if r_tag else ''}"[:64]
                    script_path = job_dir / f"{job_name}.sh"
                    bench_argv_ds = build_bench_argv(
                        model_flag=flag_name,
                        dataset_flags=ds_flags,
                        method=spec,
                        k=k_for_bench,
                        async_fan_out=int(args.async_fan_out),
                        batch_size=b_val,
                        temp=float(temp_val),
                        numseqs=int(args.numseqs),
                        output_len=int(args.output_len),
                        gpus=gpu_n,
                        profile_mode=args.profile_mode,
                        profiler_output_dir=prof_rel_ds,
                        extra_bench_args=tuple(args.extra_bench_arg),
                        hv_target_verify_interval=hv_round,
                        pivot_topk=pivot_bench_topk,
                        pivot_expansion_pct=pivot_bench_pct,
                        pivot_expansion_policy=(pivot_policy_for_run if uses_pivot_profiler_layout(spec.id) else None),
                    )
                    _record_run_script_argv(bench_argv_ds, method_id=spec.id)
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
                        bench_argv=bench_argv_ds,
                        hf_home_fallback=str(args.hf_home_fallback),
                        optional_env_before_bench=optional_env,
                    )
                    if args.dry_run:
                        print(f"would write: {script_path}")
                    else:
                        script_path.write_text(text, encoding="utf-8")
                        os.chmod(script_path, 0o755)
                    n_written += 1
            else:
                job_name = f"bench_{fam}_{spec.id}_b{b_val}_{k_path}_{temp_tag}{'_' + r_tag if r_tag else ''}"[:64]
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
                    optional_env_before_bench=optional_env,
                )
                if args.dry_run:
                    print(f"would write: {script_path}")
                else:
                    script_path.write_text(text, encoding="utf-8")
                    os.chmod(script_path, 0o755)
                n_written += 1

    def _write_run_only_script(
        path_str: str,
        chunks: list[list[str]],
        label: str,
        *,
        require_multi_dataset: bool,
    ) -> None:
        if require_multi_dataset and not multi_dataset_sweep:
            return
        if not chunks:
            return
        out_path = Path(path_str).expanduser()
        lines = [
            "#!/usr/bin/env bash",
            "",
            "# Bench-only commands (sleep 5 between). Generated by profile/gen_profile_script.py.",
            "",
        ]
        for i, argv_rb in enumerate(chunks):
            if i:
                lines.extend(["", "sleep 5", ""])
            lines.append(format_multiline_cmd(argv_rb))
        lines.append("")
        body = "\n".join(lines)
        if args.dry_run:
            print(f"would write {label} script ({len(chunks)} commands): {out_path}", file=sys.stderr)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(body, encoding="utf-8")
            os.chmod(out_path, 0o755)
            print(f"Wrote {label} script ({len(chunks)} bench command(s)): {out_path}", file=sys.stderr)

    if args.batch:
        _write_run_only_script(
            args.run_batch_sh,
            run_batch_chunks,
            "run-batch",
            require_multi_dataset=True,
        )
    if args.length:
        _write_run_only_script(
            args.run_length_sh,
            run_length_chunks,
            "run-length",
            require_multi_dataset=True,
        )
    if args.temp:
        _write_run_only_script(
            args.run_temp_sh,
            run_temp_chunks,
            "run-temp",
            require_multi_dataset=False,
        )
    if args.static:
        _write_run_only_script(
            args.run_pivot_static_sh,
            run_pivot_static_chunks,
            "run-pivot-static",
            require_multi_dataset=False,
        )

    print(f"Generated {n_written} job script(s) under {job_root}")


if __name__ == "__main__":
    main()
