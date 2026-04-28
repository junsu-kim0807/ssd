import os
import sys
import importlib.util
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
from random import randint
from typing import List, Optional, Tuple
try:
    from ssd.paths import DATASET_PATHS, HF_CACHE_DIR, EAGLE3_SPECFORGE_70B, EAGLE3_YUHUILI_8B, EAGLE3_QWEN_32B
except ImportError:
    from bench_paths import DATASET_PATHS, HF_CACHE_DIR, EAGLE3_SPECFORGE_70B, EAGLE3_YUHUILI_8B, EAGLE3_QWEN_32B

from ssd.utils.misc import load_auto_tokenizer

# HuggingFace hub repo ids → default bench targets/drafts (--size ignored for these presets).
BENCH_PRESET_QWEN_TARGET = "Qwen/Qwen3-32B"
BENCH_PRESET_QWEN_DRAFT = "Qwen/Qwen3-0.6B"
BENCH_PRESET_GEMMA_TARGET = "google/gemma-4-31B-it"
BENCH_PRESET_GEMMA_DRAFT = "google/gemma-4-E4B-it"
# Default intermediate for ``--spec_policy hierarchical`` when ``--intermediate`` is unset.
BENCH_PRESET_HIERARCHICAL_INTERMEDIATE_QWEN = "Qwen/Qwen3-8B"
BENCH_PRESET_HIERARCHICAL_INTERMEDIATE_LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
BENCH_PRESET_VICUNA_TARGET = "lmsys/vicuna-13b-v1.3"
BENCH_PRESET_VICUNA_DRAFT = "double7/vicuna-68m"
# Single supported Vicuna bench preset for profiling / sweeps (13B + 160M draft; hierarchical uses 7B below).
BENCH_PRESET_VICUNA13B_160M_TARGET = "lmsys/vicuna-13b-v1.3"
BENCH_PRESET_VICUNA13B_160M_DRAFT = "double7/vicuna-160m"
BENCH_PRESET_VICUNA13B_160M_INTERMEDIATE = "lmsys/vicuna-7b-v1.3"


def hf_hub_cache_dir(cache_dir: str, repo_id: str) -> str:
    """Map ``org/name`` hub id to a HuggingFace hub cache directory name under ``cache_dir``."""
    if "/" not in repo_id or repo_id.count("/") != 1:
        raise ValueError(f"Expected HF repo id 'org/name', got {repo_id!r}")
    org, name = repo_id.split("/", 1)
    return os.path.join(cache_dir, f"models--{org}--{name}")


def _get_snapshot_path(base_path: str) -> str:
    """Resolve a model directory to an actual snapshot directory containing config.json.

    Accepts either:
    - a snapshot directory itself (contains config.json)
    - a "snapshots" parent dir (will pick the first child)
    - a base dir containing subdirs with config.json
    """
    if os.path.isdir(base_path):
        # Already a snapshot
        if os.path.exists(os.path.join(base_path, "config.json")):
            return base_path

        # Look for huggingface-style snapshots dir
        snapshots_dir = os.path.join(base_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            for item in os.listdir(snapshots_dir):
                item_path = os.path.join(snapshots_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                    return item_path

        # Otherwise, try direct children
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config.json")):
                return item_path

    raise FileNotFoundError(
        f"No snapshot (config.json) found under {base_path}")


_get_data_from_hf_mod = None


def _load_get_data_from_hf():
    global _get_data_from_hf_mod
    if _get_data_from_hf_mod is not None:
        return _get_data_from_hf_mod
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "get_data_from_hf.py"
    spec = importlib.util.spec_from_file_location("ssd_get_data_from_hf", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _get_data_from_hf_mod = mod
    return mod


def benchmark_dataset_label(args) -> str:
    """Stable dataset id for logging / JSONL rows."""
    if getattr(args, "example", False):
        return "example"
    if getattr(args, "random", False):
        return "random"
    if getattr(args, "all", False):
        return "all_union"
    if getattr(args, "humaneval", False):
        return "humaneval"
    if getattr(args, "alpaca", False):
        return "alpaca"
    if getattr(args, "c4", False):
        return "c4"
    if getattr(args, "ultrafeedback", False):
        return "ultrafeedback"
    if getattr(args, "aime2025", False):
        return "aime2025"
    if getattr(args, "livecodebench", False):
        return "livecodebench_lite"
    if getattr(args, "codeelo", False):
        return "codeelo"
    if getattr(args, "math500", False):
        return "math500"
    if getattr(args, "govreport", False):
        return "govreport"
    if getattr(args, "qa", False):
        return "qa"
    return "gsm"


def ensure_benchmark_dataset(args) -> None:
    """If --prepare_data, download missing JSONL for the selected dataset(s)."""
    if not getattr(args, "prepare_data", False):
        return
    m = _load_get_data_from_hf()
    if getattr(args, "all", False):
        for name in (
            "humaneval",
            "alpaca",
            "gsm",
            "ultrafeedback",
            "codeelo",
            "math500",
            "govreport",
        "qa",
            "livecodebench_lite",
        ):
            _ensure_single_dataset_file(name, m)
        return
    key = benchmark_dataset_label(args)
    if key == "example" or key == "random":
        return
    if key == "all_union":
        return
    _ensure_single_dataset_file(key, m)


def _ensure_single_dataset_file(dataset_key: str, m) -> None:
    if dataset_key not in DATASET_PATHS:
        print(f"[prepare_data] Unknown dataset key {dataset_key!r}, skip.")
        return
    path = DATASET_PATHS[dataset_key]
    if os.path.exists(path):
        print(f"[prepare_data] Found {path}")
        return
    fn = {
        "gsm": m.download_gsm8k_data,
        "humaneval": m.download_humaneval_data,
        "alpaca": m.download_alpaca_data,
        "c4": m.download_c4_data,
        "ultrafeedback": m.download_ultrafeedback_data,
        "aime2025": m.download_aime2025_data,
        "livecodebench_lite": m.download_livecodebench_code_generation_lite_data,
        "codeelo": m.download_codeelo_data,
        "math500": m.download_math500_data,
        "govreport": m.download_govreport_data,
        "qa": m.download_natural_questions_data,
    }.get(dataset_key)
    if fn is None:
        print(f"[prepare_data] No downloader for {dataset_key!r}")
        return
    print(f"[prepare_data] Missing {path}, downloading...")
    fn(None)


def _get_draft_model_path(args, cache_dir: str) -> str:
    """Get draft model path based on size or explicit directory."""
    if args.draft is not None and os.path.isdir(args.draft):
        return args.draft

    if getattr(args, "spec", False) and args.draft is None:
        if getattr(args, "gemma", False):
            return _get_snapshot_path(hf_hub_cache_dir(cache_dir, BENCH_PRESET_GEMMA_DRAFT))
        if getattr(args, "vicuna13b_160m", False):
            return _get_snapshot_path(hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA13B_160M_DRAFT))
        if getattr(args, "vicuna", False):
            return _get_snapshot_path(hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA_DRAFT))
        if args.qwen:
            return _get_snapshot_path(hf_hub_cache_dir(cache_dir, BENCH_PRESET_QWEN_DRAFT))

    # Handle EAGLE auto-selection
    if getattr(args, "eagle", False):
        if args.llama:
            if args.size == "8":
                return EAGLE3_YUHUILI_8B
            elif args.size == "70":
                return EAGLE3_SPECFORGE_70B
            else:
                raise ValueError(f"EAGLE draft not available for Llama size {args.size}")
        else:
            if args.size == "32":
                return EAGLE3_QWEN_32B
            else:
                raise ValueError(f"EAGLE draft not available for Qwen size {args.size}")

    if args.llama:
        draft_size_to_model = {
            "1": "Llama-3.2-1B-Instruct",
            "3": "Llama-3.2-3B-Instruct",
            "8": "Llama-3.1-8B-Instruct",
            "70": "Llama-3.1-70B-Instruct",
        }
        if args.draft not in draft_size_to_model:
            raise ValueError(
                f"Draft size {args.draft} not available for Llama models. Available sizes: {list(draft_size_to_model.keys())}"
            )
        draft_model_name = draft_size_to_model[args.draft]
        return os.path.join(cache_dir, f"models--meta-llama--{draft_model_name}")
    else:
        draft_size_to_model = {
            "0.6": "Qwen3-0.6B",
            "1": "Llama-3.2-1B-Instruct",
        }
        if args.draft not in draft_size_to_model:
            raise ValueError(
                f"Draft size {args.draft} not available for Qwen models. Available sizes: {list(draft_size_to_model.keys())}"
            )
        draft_model_name = draft_size_to_model[args.draft]
        if args.draft == "1":
            return os.path.join(cache_dir, f"models--meta-llama--{draft_model_name}")
        else:
            return os.path.join(cache_dir, f"models--Qwen--{draft_model_name}")


def get_model_paths(args, cache_dir: str = HF_CACHE_DIR) -> Tuple[str, str, Optional[str]]:
    """Resolve model and draft paths (pointing to snapshot dirs with config.json)."""
    if getattr(args, "gemma", False):
        model_name = BENCH_PRESET_GEMMA_TARGET
        model_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_GEMMA_TARGET)
        default_draft_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_GEMMA_DRAFT)
    elif getattr(args, "vicuna13b_160m", False):
        model_name = BENCH_PRESET_VICUNA13B_160M_TARGET
        model_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA13B_160M_TARGET)
        default_draft_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA13B_160M_DRAFT)
    elif getattr(args, "vicuna", False):
        model_name = BENCH_PRESET_VICUNA_TARGET
        model_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA_TARGET)
        default_draft_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA_DRAFT)
    elif args.qwen:
        model_name = BENCH_PRESET_QWEN_TARGET
        model_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_QWEN_TARGET)
        default_draft_base = hf_hub_cache_dir(cache_dir, BENCH_PRESET_QWEN_DRAFT)
    elif args.llama:
        size_to_model = {
            "1": "Llama-3.2-1B-Instruct",
            "3": "Llama-3.2-3B-Instruct",
            "8": "Llama-3.1-8B-Instruct",
            "70": "Llama-3.3-70B-Instruct" if getattr(args, "eagle", False) else "Llama-3.1-70B-Instruct",
        }
        if args.size not in size_to_model:
            raise ValueError(
                f"Size {args.size} not available for Llama models. Available sizes: {list(size_to_model.keys())}"
            )
        model_name = size_to_model[args.size]
        model_base = os.path.join(
            cache_dir, f"models--meta-llama--{model_name}")
        default_draft_base = os.path.join(
            cache_dir, "models--meta-llama--Llama-3.2-1B-Instruct")
    else:
        raise ValueError(
            "Expected --llama (default), --qwen, --gemma, --vicuna, or --vicuna13b_160m for model selection"
        )

    model_path = _get_snapshot_path(model_base)

    # Always resolve a draft path so callers can pass it through unchanged
    if getattr(args, "eagle", False) or getattr(args, "draft", None):
         draft_base = _get_draft_model_path(args, cache_dir)
    else:
         draft_base = default_draft_base
         
    draft_path = _get_snapshot_path(draft_base)

    return model_name, model_path, draft_path


def resolve_intermediate_model_path(args, cache_dir: str = HF_CACHE_DIR) -> Optional[str]:
    """Hierarchical-only: snapshot dir for ``Config.intermediate`` (empty => draft in engine)."""
    if not getattr(args, "spec", False):
        return None
    if getattr(args, "spec_policy", "default") not in {"hierarchical", "pivot_hierarchical"}:
        return None
    explicit = getattr(args, "intermediate", None)
    if explicit and str(explicit).strip():
        raw = str(explicit).strip()
        if os.path.isdir(raw):
            return _get_snapshot_path(raw)
        if "/" in raw:
            return _get_snapshot_path(hf_hub_cache_dir(cache_dir, raw))
        raise ValueError(
            f"--intermediate must be a local directory or HF hub id 'org/name', got {raw!r}"
        )
    if getattr(args, "vicuna13b_160m", False):
        return _get_snapshot_path(
            hf_hub_cache_dir(cache_dir, BENCH_PRESET_VICUNA13B_160M_INTERMEDIATE)
        )
    if getattr(args, "qwen", False):
        return _get_snapshot_path(
            hf_hub_cache_dir(cache_dir, BENCH_PRESET_HIERARCHICAL_INTERMEDIATE_QWEN)
        )
    if getattr(args, "gemma", False):
        return _get_snapshot_path(
            hf_hub_cache_dir(cache_dir, BENCH_PRESET_GEMMA_DRAFT)
        )
    if getattr(args, "llama", False):
        return _get_snapshot_path(
            hf_hub_cache_dir(cache_dir, BENCH_PRESET_HIERARCHICAL_INTERMEDIATE_LLAMA)
        )
    return None


def load_dataset_token_ids(
    dataset_name: str,
    model_path: str,
    num_prompts: int,
    input_len: int,
    use_chat_template: bool = False,
) -> Optional[List[List[int]]]:
    """Load and tokenize dataset prompts to token ids, padding/truncating to target length.

    target_len = max(len(text_tokens), input_len)
    """
    if dataset_name not in DATASET_PATHS:
        print(
            f"Warning: Unknown dataset {dataset_name}, falling back to random tokens")
        return None

    dataset_file_path = DATASET_PATHS[dataset_name]
    if not os.path.exists(dataset_file_path):
        print(
            f"Warning: Dataset file not found at {dataset_file_path}, falling back to random tokens")
        return None

    try:
        tokenizer = load_auto_tokenizer(model_path)
        prompts: List[List[int]] = []
        with open(dataset_file_path, "r") as f:
            for _, line in enumerate(f):
                if len(prompts) >= num_prompts:
                    break
                data = json.loads(line.strip())
                # AIME 2025 uses "problem"; LiveCodeBench exports use "text" or raw "question_content"
                text: str = data.get(
                    "problem",
                    data.get(
                        "text",
                        data.get(
                            "question_content",
                            data.get("instruction", ""),
                        ),
                    ),
                )
                if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
                    tokens = tokenizer.apply_chat_template(
                        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": text}],
                        add_generation_prompt=True,
                    )
                else:
                    tokens = tokenizer.encode(text, add_special_tokens=False)

                target_len = max(len(tokens), input_len)

                if len(tokens) >= target_len:
                    truncated_tokens = tokens[:target_len]
                else:
                    truncated_tokens = tokens
                prompts.append(truncated_tokens)
        return prompts
    except Exception as e:
        print(
            f"Warning: Error loading {dataset_name} prompts: {e}, falling back to random tokens")
        return None


def load_all_dataset_token_ids(
    model_path: str,
    num_prompts_per_dataset: int,
    input_len: int,
    use_chat_template: bool = False,
) -> List[List[int]]:
    """Load tokenized prompts from a union of datasets, falling back to random when needed."""
    datasets = ["humaneval", "alpaca", "gsm", "ultrafeedback"]
    all_prompts: List[List[int]] = []

    for dataset_name in datasets:
        print(
            f"Loading {num_prompts_per_dataset} prompts from {dataset_name}...")
        dataset_prompts = load_dataset_token_ids(
            dataset_name, model_path, num_prompts_per_dataset, input_len,
            use_chat_template=use_chat_template)
        if dataset_prompts is not None:
            all_prompts.extend(dataset_prompts)
        else:
            print(
                f"Failed to load {dataset_name}, adding random tokens instead")
            random_prompts = [[randint(0, 10000) for _ in range(
                input_len)] for _ in range(num_prompts_per_dataset)]
            all_prompts.extend(random_prompts)

    print(f"Total prompts loaded: {len(all_prompts)}")
    return all_prompts


def generate_benchmark_inputs(
    args,
    model_path: str,
) -> Tuple[Optional[List[str]], Optional[List[List[int]]], Optional[List[str]]]:
    """Create input prompts.

    Returns (string_prompts, prompt_token_ids, original_prompts)
    - string_prompts: list[str] when --example is used (chat template applied)
    - prompt_token_ids: list[list[int]] in dataset/random/all modes
    - original_prompts: for display when --example
    """
    if getattr(args, "example", False):
        example_prompts = [
            "introduce yourself",
            "explain the concept of recursion",
            "describe the color blue",
            "what are you doing?",
            "how do you feel?",
            "what's the weather like today?",
            "tell me a joke",
            "what is the meaning of life?",
        ]
        num_prompts = min(args.numseqs, len(example_prompts))
        selected_prompts = example_prompts[:num_prompts]

        tokenizer = load_auto_tokenizer(model_path)
        string_prompts = selected_prompts
        return string_prompts, None, selected_prompts

    if getattr(args, "random", False):
        prompt_token_ids = [[randint(0, 10000) for _ in range(
            args.input_len)] for _ in range(args.numseqs)]
        return None, prompt_token_ids, None

    use_chat_template = getattr(args, "chat_template", False) or getattr(args, "eagle", False)

    if getattr(args, "all", False):
        token_ids = load_all_dataset_token_ids(
            model_path, args.numseqs, args.input_len,
            use_chat_template=use_chat_template)
        if not token_ids:
            print("Warning: All dataset loading failed, falling back to random tokens")
            token_ids = [[randint(0, 10000) for _ in range(
                args.input_len)] for _ in range(args.numseqs * 4)]
        return None, token_ids, None

    # Single dataset case
    if getattr(args, "humaneval", False):
        dataset_name = "humaneval"
    elif getattr(args, "alpaca", False):
        dataset_name = "alpaca"
    elif getattr(args, "c4", False):
        dataset_name = "c4"
    elif getattr(args, "ultrafeedback", False):
        dataset_name = "ultrafeedback"
    elif getattr(args, "aime2025", False):
        dataset_name = "aime2025"
    elif getattr(args, "livecodebench", False):
        dataset_name = "livecodebench_lite"
    elif getattr(args, "codeelo", False):
        dataset_name = "codeelo"
    elif getattr(args, "math500", False):
        dataset_name = "math500"
    elif getattr(args, "govreport", False):
        dataset_name = "govreport"
    elif getattr(args, "qa", False):
        dataset_name = "qa"
    else:
        dataset_name = "gsm"

    dataset_prompts = load_dataset_token_ids(
        dataset_name, model_path, args.numseqs, args.input_len,
        use_chat_template=use_chat_template)
    if dataset_prompts is None:
        token_ids = [[randint(0, 10000) for _ in range(args.input_len)]
                     for _ in range(args.numseqs)]
    else:
        token_ids = dataset_prompts
    return None, token_ids, None
