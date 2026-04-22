import os
import ssd.paths  # noqa: F401 — sets TORCH_CUDA_ARCH_LIST before flashinfer import
import sys
import time
import argparse
import json
from random import randint, seed
from ssd import LLM, SamplingParams
from ssd.engine.llm_engine import METRICS
from transformers import AutoTokenizer
import wandb
from bench_helpers import (
    benchmark_dataset_label,
    ensure_benchmark_dataset,
    generate_benchmark_inputs,
    get_model_paths,
)


def parse_arguments():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark SSD performance (API similar to example.py)")

    # Model configuration
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="70",
                        help="Model size in billions of parameters (0.6, 1.7, 4, 8, 14, 32, 1, 3, 70)")
    parser.add_argument("--llama", action="store_true", default=True, help="Use Llama models (default)")
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="Bench preset: Qwen/Qwen3-32B target (ignores --size). With --spec, default draft Qwen/Qwen3-0.6B unless --draft is set.",
    )
    parser.add_argument(
        "--gemma",
        action="store_true",
        help="Bench preset: google/gemma-4-31B-it (ignores --size). With --spec, default draft google/gemma-4-E4B-it unless --draft is set.",
    )
    parser.add_argument(
        "--vicuna",
        action="store_true",
        help="Bench preset: lmsys/vicuna-13b-v1.3 (ignores --size). With --spec, default draft double7/vicuna-68m unless --draft is set.",
    )
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model size (0.6 for Qwen-0.6B, 1 for Llama-1B) or path to draft model")

    # Execution configuration
    parser.add_argument("--eager", action="store_true", help="Use eager execution (disable CUDA graphs)")
    parser.add_argument("--gpus", type=int, default=1, help="Total number of gpus")

    # Speculative decoding configuration
    parser.add_argument("--spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--eagle", action="store_true", help="Enable eagle speculative decoding (implies --spec, uses default eagle draft for model)")
    parser.add_argument("--k", type=int, default=6, help="Speculative decoding k value")
    parser.add_argument("--async", action="store_true", help="Enable async speculative decoding")
    parser.add_argument("--f", type=int, default=3, help="Async fan out value")
    parser.add_argument("--fl", type=int, nargs='+', default=None, help="Fan out list (e.g., --fl 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flh", type=int, nargs='+', default=None, help="Fan out list (e.g., --flh 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flm", type=int, nargs='+', default=None, help="Fan out list miss (e.g., --flm 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--backup", type=str, choices=["jit", "fast"], default="jit", help="Backup strategy (jit or fast)")
    parser.add_argument("--spec_policy", type=str, choices=["default", "pivot"], default="default",
                        help="Speculative policy to use")
    parser.add_argument("--spec_hive", action="store_true",
                        help="Enable spec_hive mode for pivot policy")
    parser.add_argument("--interval", type=int, default=0,
                        help="Pivot interval to force target verification")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Pivot confidence threshold")
    parser.add_argument("--expansion_pct", type=float, default=1.0,
                        help="Pivot top-k expansion percentage")

    # Memory and batching configuration
    parser.add_argument("--block_sz", type=int, default=256, help="KV cache block size (see config.py: kvcache_block_size)")
    parser.add_argument("--b", type=int, default=1, help="Maximum number of sequences in batch")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")

    # Generation configuration
    parser.add_argument("--input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=512, help="Maximum output length")
    parser.add_argument("--numseqs", type=int, default=128, help="Number of sequences to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--dtemp", type=float, default=None, help="Draft async temperature (overrides --temp)")
    parser.add_argument("--x", type=float, default=None, help="Sampler x for generation (Saguaro sampling coefficient)")

    # Example mode
    parser.add_argument("--example", action="store_true", help="Use real prompts like in example.py and print generations (supports up to batch size 8)")
    parser.add_argument("--humaneval", action="store_true", help="Use HumanEval prompts")
    parser.add_argument("--alpaca", action="store_true", help="Use Alpaca prompts")
    parser.add_argument("--c4", action="store_true", help="Use C4 prompts")
    parser.add_argument("--ultrafeedback", action="store_true", help="Use UltraFeedback prompts")
    parser.add_argument("--aime2025", action="store_true", help="Use AIME 2025 (math-ai/aime25) prompts from SSD_DATASET_DIR")
    parser.add_argument(
        "--livecodebench",
        "--lcb_lite",
        action="store_true",
        dest="livecodebench",
        help="Use LiveCodeBench code_generation_lite (release_v5) prompts from SSD_DATASET_DIR",
    )
    parser.add_argument("--random", action="store_true", help="Use random tokens instead of dataset prompts")
    parser.add_argument("--prompt_offset", type=int, default=0, help="Skip first N prompts per dataset (for variance testing)")
    parser.add_argument("--all", action="store_true", help="Use numseqs from each dataset (union dataset with numseqs*4 total)")
    parser.add_argument("--chat_template", action="store_true", help="Wrap dataset prompts in chat template before tokenizing")
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Download missing JSONL for the selected dataset into SSD_DATASET_DIR (via scripts/get_data_from_hf.py)",
    )

    # Profiling (same semantics as ssd.config.Config; inactive unless profiler_output_dir is set)
    parser.add_argument(
        "--profiler_mode",
        type=str,
        default=None,
        choices=["cost_breakdown", "metadata", "cost_metadata", "kernel_breakdown"],
        help="Profiler mode when --profiler_output_dir is set (default: cost_metadata). "
        "kernel_breakdown forces eager execution in the engine.",
    )
    parser.add_argument(
        "--profiler_output_dir",
        type=str,
        default=None,
        help="Directory for profiler JSONL / cost_breakdown.json; unset disables profiling.",
    )

    # Debugging and logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves draft inputs during prefill)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps to run")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument("--group", type=str, default=None, help="Wandb group name")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name")

    # Sweep mode: load engine once, run multiple configs
    parser.add_argument("--sweep", type=str, default=None,
                        help="JSON list of override dicts. Sweepable keys: temp, b. "
                             "Each dict also supports 'name' for wandb run name.")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Write one JSON object per sequence with run metadata. "
        "With --sweep: if this is a directory (or ends with /), writes <dir>/<run_name>.jsonl per sweep; "
        "if a file path, writes <stem>_sweep{idx}_t{temp}_b{b}.jsonl per sweep. Default: bench_runs/<run_name>.jsonl",
    )
    parser.add_argument(
        "--output_jsonl_dir",
        type=str,
        default=None,
        help="Directory for per-sweep JSONL files named <cur_run_name>.jsonl (overrides default bench_runs layout).",
    )

    args = parser.parse_args()
    _n_hf_preset = int(bool(args.qwen)) + int(bool(getattr(args, "gemma", False))) + int(
        bool(getattr(args, "vicuna", False))
    )
    if _n_hf_preset > 1:
        parser.error("Use at most one of --qwen --gemma --vicuna")
    if getattr(args, "gemma", False):
        args.llama = False
    if getattr(args, "vicuna", False):
        args.llama = False
    if args.qwen:
        args.llama = False
    assert not (args.qwen and "--llama" in sys.argv), "--llama and --qwen are mutually exclusive"
    if getattr(args, "gemma", False) and "--llama" in sys.argv:
        parser.error("--gemma and explicit --llama are mutually exclusive")
    if getattr(args, "vicuna", False) and "--llama" in sys.argv:
        parser.error("--vicuna and explicit --llama are mutually exclusive")
    if args.eagle:
        args.spec = True
        assert args.llama, "Eagle currently only supports llama models"
        assert args.temp == 0.0 and args.dtemp is None, "Eagle currently only supports greedy decoding (temp=0)"
        assert getattr(args, 'async', False), "Eagle currently only supports async speculative decoding"

    _n_ds = sum(
        bool(x)
        for x in (
            args.humaneval,
            args.alpaca,
            args.c4,
            args.ultrafeedback,
            args.aime2025,
            args.livecodebench,
        )
    )
    if _n_ds > 1:
        parser.error(
            "Choose at most one dataset flag among "
            "--humaneval --alpaca --c4 --ultrafeedback --aime2025 --livecodebench"
        )
    return args


def create_run_name(args):
    """Create a descriptive run name for wandb logging."""
    spec_mode_str = "spec" if args.spec else "normal"
    async_mode_str = "_async" if getattr(args, 'async', False) else ""
    jit_mode_str = "_jit" if args.backup == "jit" else ""
    if getattr(args, "gemma", False):
        model_type = "gemma"
    elif getattr(args, "vicuna", False):
        model_type = "vicuna"
    elif args.llama:
        model_type = "llama"
    else:
        model_type = "qwen"
    size_part = args.size if args.llama else "hub"
    example_str = "_example" if args.example else ""
    humaneval_str = "_humaneval" if args.humaneval else ""
    alpaca_str = "_alpaca" if args.alpaca else ""
    c4_str = "_c4" if args.c4 else ""
    ultrafeedback_str = "_ultrafeedback" if args.ultrafeedback else ""
    aime_str = "_aime2025" if getattr(args, "aime2025", False) else ""
    lcb_str = "_livecodebench" if getattr(args, "livecodebench", False) else ""
    random_str = "_random" if args.random else ""
    all_str = "_all" if args.all else ""
    _non_gsm = (
        args.example
        or args.humaneval
        or args.alpaca
        or args.c4
        or args.ultrafeedback
        or args.random
        or args.all
        or getattr(args, "aime2025", False)
        or getattr(args, "livecodebench", False)
    )
    gsm_str = "" if _non_gsm else "_gsm"
    sampler_x_str = f"_sampler_x{args.x}" if args.x else ""

    _pod = getattr(args, "profiler_output_dir", None) or ""
    prof_short = ""
    if str(_pod).strip():
        pm = getattr(args, "profiler_mode", None) or "cost_metadata"
        prof_token = {"cost_breakdown": "cb", "metadata": "md", "cost_metadata": "cm", "kernel_breakdown": "kb"}.get(
            pm, "prof"
        )
        prof_short = f"_prof_{prof_token}"

    temp_str = f"_temp{args.temp}"
    if args.dtemp is not None:
        temp_str += f"_dtemp{args.dtemp}"

    draft_str = f"_draft{args.draft}" if args.draft is not None else "_nodraft"
    k_str = f"_k{args.k}"
    f_str = f"_f{args.f}"

    return args.name if args.name else (
        f"{model_type}_size{size_part}_{spec_mode_str}{async_mode_str}{jit_mode_str}_b{args.b}{k_str}{f_str}{draft_str}"
        f"{temp_str}{sampler_x_str}{prof_short}{example_str}{humaneval_str}{alpaca_str}{c4_str}{ultrafeedback_str}"
        f"{aime_str}{lcb_str}{random_str}{all_str}{gsm_str}"
    )


def initialize_wandb(args, run_name):
    """Initialize wandb logging if requested."""
    if not args.wandb:
        return

    wandb.init(
        project="ssd",
        name=run_name,
        group=args.group,
        config={
            "model_size": args.size,
            "gpus": args.gpus,
            "speculative_decoding": args.spec,
            "async_speculative": getattr(args, 'async', False),
            "jit_speculative": args.backup == "jit",
            "k": args.k if args.spec else None,
            "f": args.f,
            "fan_out_list": args.flh,
            "fan_out_list_miss": args.flm,
            "llama": args.llama,
            "gemma_preset": getattr(args, "gemma", False),
            "vicuna_preset": getattr(args, "vicuna", False),
            "qwen_hub_preset": bool(args.qwen),
            "bench_cli_size_ignored_for_hf_preset": bool(
                args.qwen or getattr(args, "gemma", False) or getattr(args, "vicuna", False)
            ),
            "max_model_len": args.max_model_len,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "numseqs": args.numseqs,
            "draft_model": args.draft,
            "b": args.b,
            "block_size": args.block_sz,
            "eager": args.eager,
            "example_mode": args.example,
            "humaneval_mode": args.humaneval,
            "alpaca_mode": args.alpaca,
            "c4_mode": args.c4,
            "ultrafeedback_mode": args.ultrafeedback,
            "aime2025_mode": getattr(args, "aime2025", False),
            "livecodebench_mode": getattr(args, "livecodebench", False),
            "benchmark_dataset": benchmark_dataset_label(args),
            "random_mode": args.random,
            "all_mode": args.all,
            "sampler_x": args.x,
            "profiler_mode": getattr(args, "profiler_mode", None),
            "profiler_enabled": bool(getattr(args, "profiler_output_dir", None) and str(args.profiler_output_dir).strip()),
            "profiler_output_dir_basename": (
                os.path.basename(str(args.profiler_output_dir).rstrip(os.sep))
                if getattr(args, "profiler_output_dir", None) and str(args.profiler_output_dir).strip()
                else None
            ),
            "implementation": "ssd",
            "max_steps": args.max_steps,
            "spec_policy": args.spec_policy,
            "spec_hive": args.spec_hive,
            "interval": args.interval,
            "threshold": args.threshold,
            "expansion_pct": args.expansion_pct,
        }
    )


def create_llm_kwargs(args, draft_path):
    """Create LLM initialization arguments."""
    llm_kwargs = dict(
        enforce_eager=args.eager,
        num_gpus=args.gpus,
        speculate=args.spec,
        speculate_k=args.k,
        draft_async=getattr(args, 'async', False),
        async_fan_out=args.f,
        verbose=args.verbose,
        draft=draft_path,
        kvcache_block_size=args.block_sz,
        max_num_seqs=args.b,
        max_model_len=args.max_model_len,
        sampler_x=args.x,
        jit_speculate=(args.backup == "jit"),
        max_steps=args.max_steps,
        spec_policy=args.spec_policy,
        spec_hive=args.spec_hive,
        interval=args.interval,
        threshold=args.threshold,
        expansion_pct=args.expansion_pct,
    )

    if args.flh is not None:
        llm_kwargs["fan_out_list"] = args.flh
    if args.flm is not None:
        llm_kwargs["fan_out_list_miss"] = args.flm

    _pod = getattr(args, "profiler_output_dir", None)
    if _pod and str(_pod).strip():
        llm_kwargs["profiler_output_dir"] = str(_pod).strip()
        llm_kwargs["profiler_mode"] = getattr(args, "profiler_mode", None) or "cost_metadata"

    return llm_kwargs


def _sanitize_run_filename(name: str) -> str:
    bad = '/\\:*?"<>|'
    s = "".join(c if c not in bad else "_" for c in name)
    return s[:220] if len(s) > 220 else s


def resolve_bench_output_jsonl_path(
    args,
    si: int,
    temp: float,
    b: int,
    cur_run_name: str,
    n_sweeps: int,
) -> str:
    has_sweep = n_sweeps > 1
    out_dir = getattr(args, "output_jsonl_dir", None)
    if out_dir and str(out_dir).strip():
        base = str(out_dir).strip().rstrip(os.sep)
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, _sanitize_run_filename(cur_run_name) + ".jsonl")

    user_path = getattr(args, "output_jsonl", None)
    if user_path and str(user_path).strip():
        p = str(user_path).strip()
        is_dir = p.endswith(os.sep) or p.endswith("/") or (os.path.exists(p) and os.path.isdir(p))
        if has_sweep and is_dir:
            base = p.rstrip(os.sep).rstrip("/")
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, _sanitize_run_filename(cur_run_name) + ".jsonl")
        if has_sweep and not is_dir:
            root, ext = os.path.splitext(p)
            if not ext:
                ext = ".jsonl"
            return f"{root}_sweep{si}_t{temp}_b{b}{ext}"
        return p

    os.makedirs("bench_runs", exist_ok=True)
    return os.path.join("bench_runs", _sanitize_run_filename(cur_run_name) + ".jsonl")


def write_bench_outputs_jsonl(
    *,
    path: str,
    prompts,
    outputs,
    tokenizer,
    run_name: str,
    sweep_idx: int,
    temperature: float,
    max_num_seqs: int,
    dataset: str,
    profiler_mode,
    profiler_enabled: bool,
    original_prompts,
) -> None:
    """One JSON object per sequence; always includes sweep / temp / b / dataset for downstream merges."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    pm = profiler_mode if profiler_enabled else None
    with open(path, "w", encoding="utf-8") as f:
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            if isinstance(prompt, list):
                prompt_text = tokenizer.decode(prompt, skip_special_tokens=True)
                prompt_token_ids = prompt
            else:
                prompt_text = str(prompt)
                prompt_token_ids = None
            if original_prompts and i < len(original_prompts):
                display_prompt = original_prompts[i]
            else:
                display_prompt = prompt_text
            row = {
                "run_name": run_name,
                "sweep_idx": sweep_idx,
                "temperature": temperature,
                "max_num_seqs": max_num_seqs,
                "dataset": dataset,
                "profiler_mode": pm,
                "profiler_enabled": profiler_enabled,
                "request_index": i,
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_token_ids,
                "display_prompt": display_prompt,
                "completion_text": output.get("text", ""),
                "completion_token_ids": output.get("token_ids", []),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, mode, run_name):
    """Log metrics to wandb if enabled."""
    if not args.wandb:
        return

    wandb_metrics = {
        "official_total_tokens": total_tokens,
        "official_total_time": total_time,
        "official_end_to_end_throughput": throughput,
        "model_name": model_name,
        "mode": mode,
        "run_name": run_name,
    }

    if metrics:
        if "prefill_total_time" in metrics and "prefill_total_tokens" in metrics:
            if metrics["prefill_total_time"] > 0:
                wandb_metrics["metrics_prefill_throughput"] = metrics["prefill_total_tokens"] / metrics["prefill_total_time"]

        if "decode_total_time" in metrics and "decode_total_tokens" in metrics:
            if metrics["decode_total_time"] > 0:
                wandb_metrics["metrics_decode_throughput"] = metrics["decode_total_tokens"] / metrics["decode_total_time"]

        if "target_step_times" in metrics and metrics["target_step_times"]:
            avg_target_step_time_ms = sum(metrics["target_step_times"]) * 1000 / len(metrics["target_step_times"])
            wandb_metrics["metrics_avg_target_step_time_ms"] = avg_target_step_time_ms

        if "cache_hits" in metrics and metrics["cache_hits"]:
            wandb_metrics["metrics_avg_cache_hits"] = sum(metrics["cache_hits"]) / len(metrics["cache_hits"])

        if "accepted_suffix_lens_with_recovery" in metrics and metrics["accepted_suffix_lens_with_recovery"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_with_recovery"] = sum(metrics["accepted_suffix_lens_with_recovery"]) / len(metrics["accepted_suffix_lens_with_recovery"])
            wandb_metrics["metrics_accepted_suffix_lens_with_recovery_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_with_recovery"])

        if "accepted_suffix_lens_on_hit" in metrics and metrics["accepted_suffix_lens_on_hit"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_on_hit"] = sum(metrics["accepted_suffix_lens_on_hit"]) / len(metrics["accepted_suffix_lens_on_hit"])
            wandb_metrics["metrics_accepted_suffix_lens_on_hit_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_on_hit"])

    wandb.log(wandb_metrics)


def run_benchmark(args, llm, prompts, sampling_params):
    """Run the actual benchmark and return results."""
    if args.wandb:
        wandb.log({"sequences_processed": 0, "total_sequences": len(prompts)})

    start_time = time.time()
    outputs, metrics = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    if args.wandb:
        wandb.log({"sequences_processed": len(prompts), "total_sequences": len(prompts)})

    return outputs, total_time, metrics


def reset_metrics():
    """Reset the global METRICS dict between sweep runs."""
    for k, v in METRICS.items():
        if isinstance(v, list):
            v.clear()
        else:
            METRICS[k] = 0


def reconfigure_engine(llm, b=None):
    """Reconfigure a live engine for a new sweep run without reloading weights."""
    if b is not None:
        assert b <= llm.config.max_num_seqs, f"b={b} > initial max_num_seqs={llm.config.max_num_seqs}"
        llm.config.max_num_seqs = b
        llm.scheduler.max_num_seqs = b


def main():
    args = parse_arguments()
    seed(0)

    if args.example and args.numseqs > 8:
        print("Warning: --example mode supports up to 8 sequences, reducing numseqs from {} to 8".format(args.numseqs))
        args.numseqs = 8

    ensure_benchmark_dataset(args)

    model_name, model_path, draft_path = get_model_paths(args)

    string_prompts, prompt_token_ids, original_prompts = generate_benchmark_inputs(args, model_path)
    prompts = string_prompts if string_prompts is not None else prompt_token_ids

    if prompts:
        num_reqs = len(prompts)
    else:
        num_reqs = args.numseqs
    sampling_params = [SamplingParams(
        temperature=args.temp,
        draft_temperature=args.dtemp,
        ignore_eos=True,
        max_new_tokens=args.output_len,
    ) for _ in range(num_reqs)]

    if prompts:
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                print(f'Prompt: {prompt}')
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            elif isinstance(prompt, list):
                num_tokens = len(prompt)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")

    if args.sweep:
        sweep_configs = json.loads(args.sweep)
        assert isinstance(sweep_configs, list), "--sweep must be a JSON list of dicts"
    else:
        sweep_configs = [{}]
    n_sweeps = len(sweep_configs)

    # Create LLM (once, reused across sweep configs)
    llm_kwargs = create_llm_kwargs(args, draft_path)
    if args.eagle:
        llm_kwargs['use_eagle'] = True
    if args.debug:
        llm_kwargs['debug_mode'] = True

    llm = LLM(model_path, **llm_kwargs)
    bench_jsonl_tokenizer = AutoTokenizer.from_pretrained(model_path)

    for si, sweep_cfg in enumerate(sweep_configs):
        bad_keys = {"backup", "flh", "flm"} & set(sweep_cfg.keys())
        assert not bad_keys, f"Cannot sweep {bad_keys} — draft process won't see changes."

        temp = sweep_cfg.get("temp", args.temp)
        b = sweep_cfg.get("b", args.b)
        run_name_override = sweep_cfg.get("name", None)

        reconfigure_engine(llm, b=b)

        cur_sampling_params = [SamplingParams(
            temperature=temp,
            draft_temperature=args.dtemp,
            ignore_eos=True,
            max_new_tokens=args.output_len,
        ) for _ in range(num_reqs)]

        reset_metrics()

        if run_name_override:
            cur_run_name = run_name_override
        elif args.sweep:
            cur_run_name = f"{create_run_name(args)}_sweep{si}"
        else:
            cur_run_name = create_run_name(args)

        orig_temp, orig_name, orig_b = args.temp, args.name, args.b
        args.temp, args.b = temp, b
        if run_name_override:
            args.name = run_name_override
        initialize_wandb(args, cur_run_name)
        args.temp, args.name, args.b = orig_temp, orig_name, orig_b

        try:
            print(f"\n{'='*60}")
            print(f"SWEEP [{si+1}/{n_sweeps}] temp={temp} b={b}")
            print(f"{'='*60}")

            outputs, total_time, metrics = run_benchmark(args, llm, prompts, cur_sampling_params)

            total_tokens = sum(sp.max_new_tokens for sp in cur_sampling_params)
            throughput = total_tokens / total_time

            mode = "Eager" if args.eager else "CUDA Graphs"
            spec_mode = f" + Speculative(k={args.k})" if args.spec else ""
            async_mode = " + Async" if getattr(args, 'async', False) else ""
            jit_mode = " + JIT" if args.backup == "jit" else ""
            x_mode = f" + X({args.x})" if args.x else ""
            full_mode = mode + spec_mode + async_mode + jit_mode + x_mode

            print(f"Model: {model_name}, Mode: {full_mode}, Total: {total_tokens}tok, Time: {total_time:.2f}s, Total Throughput: {throughput:.2f}tok/s")

            if not args.random and si == 0:
                print("\n" + "="*80)
                print("GENERATIONS:")
                print("="*80)

                for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                    if i >= 10:
                        break
                    if isinstance(prompt, list):
                        decoded_prompt = bench_jsonl_tokenizer.decode(prompt, skip_special_tokens=True)
                    else:
                        decoded_prompt = prompt
                    if original_prompts and i < len(original_prompts):
                        display_prompt = original_prompts[i]
                    else:
                        display_prompt = decoded_prompt
                    print(f"\nPrompt {i+1}: {display_prompt!r}")
                    print(f"Generation: {output['text']!r}")
                    print("-" * 40)

            log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, full_mode, cur_run_name)

            out_jsonl_path = resolve_bench_output_jsonl_path(
                args, si, temp, b, cur_run_name, n_sweeps
            )
            prof_enabled = bool(
                getattr(args, "profiler_output_dir", None)
                and str(args.profiler_output_dir).strip()
            )
            prof_mode = (
                (getattr(args, "profiler_mode", None) or "cost_metadata")
                if prof_enabled
                else None
            )
            write_bench_outputs_jsonl(
                path=out_jsonl_path,
                prompts=prompts,
                outputs=outputs,
                tokenizer=bench_jsonl_tokenizer,
                run_name=cur_run_name,
                sweep_idx=si,
                temperature=float(temp),
                max_num_seqs=int(b),
                dataset=benchmark_dataset_label(args),
                profiler_mode=prof_mode,
                profiler_enabled=prof_enabled,
                original_prompts=original_prompts,
            )
            print(f"Wrote outputs JSONL: {out_jsonl_path}")

            if args.wandb:
                wandb.finish()
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
            break

    print(f'Engine exited!')
    sys.exit(0)


if __name__ == "__main__":
    main()
