import os
import json
from datasets import load_dataset

LIVECODEBENCH_LITE_DATASET = "sam-paech/livecodebench-code_generation_lite"
LIVECODEBENCH_LITE_CONFIG = "release_v5"
LIVECODEBENCH_LITE_MAX = 10000

def get_base_output_dir():
    """Root for processed JSONL files. Prefer SSD_DATASET_DIR (same as bench/ssd.paths)."""
    ssd_ds = os.environ.get("SSD_DATASET_DIR", "").strip()
    if ssd_ds:
        print(f"Using SSD_DATASET_DIR: {ssd_ds}")
        return ssd_ds
    hf_cache = os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    print(f"Using HF_DATASETS_CACHE (no SSD_DATASET_DIR): {hf_cache}")
    return os.path.join(hf_cache, "processed_datasets")


def download_gsm8k_data(num_samples=None):
    """Download GSM8K dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "gsm8k")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"gsm8k_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading GSM8K dataset...")
    
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            sample = {"text": example['question']}
            f.write(json.dumps(sample) + '\n')

            if i % 500 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} GSM8K samples to {output_file}")
    return output_file


def download_c4_data(num_samples=None):
    """Download C4 dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "c4")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"c4_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading C4 dataset...")
    
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading C4 dataset: {e}")
        raise

    print(f"Processing C4 samples (streaming)...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
                
            sample = {"text": example['text']}
            f.write(json.dumps(sample) + '\n')

            if i % 1000 == 0:
                print(f"Processed {i} samples...")

    samples_processed = min(i + 1, num_samples) if 'i' in locals() else 0
    print(f"Saved {samples_processed} C4 samples to {output_file}")
    return output_file


def download_ultrafeedback_data(num_samples=None):
    """Download UltraFeedback dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "ultrafeedback")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"ultrafeedback_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading UltraFeedback dataset...")
    
    try:
        dataset = load_dataset("openbmb/UltraFeedback", split="train")
    except Exception as e:
        print(f"Error loading UltraFeedback dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Use the instruction field as the main text
            sample = {"text": example['instruction']}
            f.write(json.dumps(sample) + '\n')

            if i % 500 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} UltraFeedback samples to {output_file}")
    return output_file


def download_humaneval_data(num_samples=None):
    """Download OpenAI HumanEval dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "humaneval")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"humaneval_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading HumanEval dataset...")
    
    try:
        dataset = load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        print(f"Error loading HumanEval dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Use the prompt field as the main text
            sample = {"text": example['prompt']}
            f.write(json.dumps(sample) + '\n')

            if i % 100 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} HumanEval samples to {output_file}")
    return output_file


def download_alpaca_data(num_samples=None):
    """Download Alpaca dataset samples and save as JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "alpaca")
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit to 10k samples max
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    
    output_file = os.path.join(output_dir, f"alpaca_data_{num_samples}.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(f"Loading Alpaca dataset...")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        print(f"Error loading Alpaca dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Combine instruction and input if available, otherwise just instruction
            text = example['instruction']
            if example.get('input', '').strip():
                text = f"{text}\n\n{example['input']}"
            
            sample = {"text": text}
            f.write(json.dumps(sample) + '\n')

            if i % 500 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} Alpaca samples to {output_file}")
    return output_file


def download_aime2025_data(num_samples=None):
    """Download AIME 2025 dataset (math-ai/aime25) and save as JSONL.
    Fields: problem, answer, id. Used for intermediate verifier profile evaluation.
    """
    output_dir = os.path.join(get_base_output_dir(), "aime2025")
    os.makedirs(output_dir, exist_ok=True)

    max_samples = 30  # AIME 2025 has 30 problems
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)

    output_file = os.path.join(output_dir, "aime2025_test.jsonl")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print("Loading AIME 2025 dataset (math-ai/aime25)...")
    try:
        dataset = load_dataset("math-ai/aime25", split="test")
    except Exception as e:
        print(f"Error loading AIME 2025: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(samples_to_process):
            example = dataset[i]
            sample = {
                "problem": example["problem"],
                "answer": example["answer"],
                "id": example["id"],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {samples_to_process} AIME 2025 samples to {output_file}")
    return output_file


def download_aime_data(num_samples=None):
    """Download AIME validation set (AI-MO/aimo-validation-aime, train split) to JSONL."""
    output_dir = os.path.join(get_base_output_dir(), "aime")
    os.makedirs(output_dir, exist_ok=True)

    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)

    output_file = os.path.join(output_dir, "aimo_validation_aime_train.jsonl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print("Loading AIME validation dataset (AI-MO/aimo-validation-aime, train split)...")
    try:
        dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    except Exception as e:
        print(f"Error loading AIME validation dataset: {e}")
        raise

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    print(f"Processing {samples_to_process} samples from {total_samples} total samples...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(samples_to_process):
            example = dataset[i]
            # Keep schema-flexible extraction; bench loader prioritizes "problem" then "text".
            problem = (
                example.get("problem")
                or example.get("question")
                or example.get("prompt")
                or example.get("text")
                or ""
            )
            answer = example.get("answer", "")
            sample = {
                "problem": str(problem).strip(),
                "answer": str(answer).strip(),
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} AIME validation samples to {output_file}")
    return output_file


LIVECODEBENCH_LITE_CONFIG = "release_v5"
LIVECODEBENCH_LITE_MAX = 10000


LIVECODEBENCH_LITE_DATASET = "sam-paech/livecodebench-code_generation_lite"
LIVECODEBENCH_LITE_CONFIG = "release_v5"
LIVECODEBENCH_LITE_MAX = 10000


def download_livecodebench_code_generation_lite_data(num_samples=None):
    """Download LiveCodeBench code_generation_lite to JSONL with {"text": ...} per line."""
    output_dir = os.path.join(get_base_output_dir(), "livecodebench_lite")
    os.makedirs(output_dir, exist_ok=True)

    max_samples = LIVECODEBENCH_LITE_MAX
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)

    output_file = os.path.join(
        output_dir,
        f"livecodebench_lite_{LIVECODEBENCH_LITE_CONFIG}_{num_samples}.jsonl",
    )

    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file

    print(
        f"Loading {LIVECODEBENCH_LITE_DATASET} "
        f"(split={LIVECODEBENCH_LITE_CONFIG})..."
    )

    try:
        dataset = load_dataset(
            LIVECODEBENCH_LITE_DATASET,
            split=LIVECODEBENCH_LITE_CONFIG,
            columns=["question_title", "question_content", "starter_code"],
        )
    except TypeError:
        dataset = load_dataset(
            LIVECODEBENCH_LITE_DATASET,
            split=LIVECODEBENCH_LITE_CONFIG,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load LiveCodeBench lite from "
            f"{LIVECODEBENCH_LITE_DATASET} split={LIVECODEBENCH_LITE_CONFIG}"
        ) from e

    total_samples = len(dataset)
    samples_to_process = min(num_samples, total_samples)
    print(f"Using split={LIVECODEBENCH_LITE_CONFIG!r} ({total_samples} rows)")

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(samples_to_process):
            example = dataset[i]

            title = example.get("question_title") or ""
            body = example.get("question_content") or ""
            starter = example.get("starter_code") or ""

            parts = []
            if title.strip():
                parts.append(title.strip())
            if body.strip():
                parts.append(body.strip())
            if starter.strip():
                parts.append("Starter code:\n" + starter.strip())

            text = "\n\n".join(parts).strip()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

            if i % 200 == 0:
                print(f"Processed {i}/{samples_to_process} samples...")

    print(f"Saved {samples_to_process} LiveCodeBench lite samples to {output_file}")
    return output_file


def _select_split(ds, preferred=("train", "test", "validation")):
    """Pick a split from a DatasetDict (or return a single Dataset)."""
    if not hasattr(ds, "keys"):
        return ds
    for k in preferred:
        if k in ds:
            return ds[k]
    return next(iter(ds.values()))


def download_math500_data(num_samples=None):
    """HuggingFaceH4/MATH-500 → JSONL with {\"text\": problem} per line."""
    output_dir = os.path.join(get_base_output_dir(), "math500")
    os.makedirs(output_dir, exist_ok=True)
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    output_file = os.path.join(output_dir, "math500_data_10000.jsonl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file
    print("Loading HuggingFaceH4/MATH-500...")
    try:
        raw = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading MATH-500: {e}")
        raise
    dataset = _select_split(raw, ("test", "train", "validation"))
    total = len(dataset)
    n = min(num_samples, total)
    print(f"Processing {n} samples from {total} total...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = dataset[i]
            problem = ex.get("problem") or ex.get("question") or ""
            f.write(json.dumps({"text": str(problem).strip()}, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"Processed {i}/{n}...")
    print(f"Saved {n} MATH-500 samples to {output_file}")
    return output_file


def download_codeelo_data(num_samples=None):
    """Qwen/CodeElo → JSONL with {\"text\": ...} per line (description + optional input)."""
    output_dir = os.path.join(get_base_output_dir(), "codeelo")
    os.makedirs(output_dir, exist_ok=True)
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    output_file = os.path.join(output_dir, "codeelo_data_10000.jsonl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file
    print("Loading Qwen/CodeElo...")
    try:
        raw = load_dataset("Qwen/CodeElo", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading CodeElo: {e}")
        raise
    dataset = _select_split(raw, ("train", "test", "validation"))
    total = len(dataset)
    n = min(num_samples, total)
    print(f"Processing {n} samples from {total} total...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = dataset[i]
            parts = []
            desc = ex.get("description")
            if desc is not None and str(desc).strip():
                parts.append(str(desc).strip())
            inp = ex.get("input")
            if inp is not None and str(inp).strip():
                parts.append(str(inp).strip())
            text = "\n\n".join(parts) if parts else str(ex.get("description") or ex.get("title") or "")
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"Processed {i}/{n}...")
    print(f"Saved {n} CodeElo samples to {output_file}")
    return output_file


GOVREPORT_MAX_REPORT_CHARS = 48000
GOVREPORT_MAX_DOWNLOAD_DOC_CHARS = 4000


def download_govreport_data(num_samples=None):
    """ccdv/govreport-summarization → JSONL summarization prompts (truncated report body)."""
    output_dir = os.path.join(get_base_output_dir(), "govreport")
    os.makedirs(output_dir, exist_ok=True)
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    output_file = os.path.join(output_dir, "govreport_data_10000.jsonl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file
    split_name = "test"
    print(f"Loading ccdv/govreport-summarization ({split_name} split)...")
    try:
        raw = load_dataset("ccdv/govreport-summarization", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading govreport-summarization: {e}")
        raise
    if hasattr(raw, "keys"):
        if split_name not in raw:
            raise RuntimeError(
                f"Expected split '{split_name}' in ccdv/govreport-summarization, "
                f"but only found: {list(raw.keys())}"
            )
        dataset = raw[split_name]
    else:
        dataset = raw
    total = len(dataset)
    target_n = min(num_samples, total)
    prefix = (
        "Summarize the following government report in clear, structured prose. "
        "Focus on main findings and policy implications.\n\n---\n\n"
    )
    print(
        f"Collecting up to {target_n} samples from {total} total "
        f"(report length <= {GOVREPORT_MAX_DOWNLOAD_DOC_CHARS} chars)..."
    )
    written = 0
    skipped_long = 0
    skipped_empty = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(total):
            ex = dataset[i]
            report = str(ex.get("report") or "").strip()
            if not report:
                skipped_empty += 1
                continue
            if len(report) > GOVREPORT_MAX_DOWNLOAD_DOC_CHARS:
                skipped_long += 1
                continue
            if len(report) > GOVREPORT_MAX_REPORT_CHARS:
                report = report[:GOVREPORT_MAX_REPORT_CHARS] + "\n\n[... document truncated ...]"
            text = prefix + report
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            if written >= target_n:
                break
            if i % 500 == 0:
                print(
                    f"Scanned {i}/{total} rows | written={written} "
                    f"(skipped_long={skipped_long}, skipped_empty={skipped_empty})..."
                )
    print(
        f"Saved {written} GovReport samples to {output_file} "
        f"(skipped_long={skipped_long}, skipped_empty={skipped_empty})"
    )
    return output_file


def download_natural_questions_data(num_samples=None):
    """sentence-transformers/natural-questions → JSONL QA prompts with {"text": question}."""
    output_dir = os.path.join(get_base_output_dir(), "natural_questions")
    os.makedirs(output_dir, exist_ok=True)
    max_samples = 10000
    if num_samples is None:
        num_samples = max_samples
    else:
        num_samples = min(num_samples, max_samples)
    output_file = os.path.join(output_dir, "natural_questions_data_10000.jsonl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping download.")
        return output_file
    print("Loading sentence-transformers/natural-questions...")
    try:
        raw = load_dataset("sentence-transformers/natural-questions", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading natural-questions: {e}")
        raise
    dataset = _select_split(raw, ("train", "validation", "test"))
    total = len(dataset)
    n = min(num_samples, total)
    print(f"Processing {n} samples from {total} total...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = dataset[i]
            # Keep extraction resilient across dataset schema changes.
            question = (
                ex.get("question")
                or ex.get("query")
                or ex.get("text")
                or ex.get("sentence")
                or ""
            )
            question = str(question).strip()
            if not question:
                continue
            f.write(json.dumps({"text": question}, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(f"Processed {i}/{n}...")
    print(f"Saved up to {n} Natural Questions samples to {output_file}")
    return output_file


def download_all_datasets(num_samples=None):
    """Download all datasets."""
    print("Downloading all datasets...")
    
    datasets = [
        ("GSM8K", download_gsm8k_data),
        ("C4", download_c4_data),
        ("UltraFeedback", download_ultrafeedback_data),
        ("HumanEval", download_humaneval_data),
        ("Alpaca", download_alpaca_data),
        ("AIME2025", download_aime2025_data),
        ("AIMEValidation", download_aime_data),
        ("LiveCodeBenchLite", download_livecodebench_code_generation_lite_data),
        ("MATH500", download_math500_data),
        ("CodeElo", download_codeelo_data),
        ("GovReport", download_govreport_data),
        ("NaturalQuestionsQA", download_natural_questions_data),
    ]
    
    output_files = {}
    for name, download_func in datasets:
        print(f"\n{'='*50}")
        print(f"Downloading {name}...")
        print('='*50)
        
        try:
            output_file = download_func(num_samples)
            output_files[name] = output_file
            print(f"✓ Successfully downloaded {name} to {output_file}")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            output_files[name] = None
    
    return output_files


DOWNLOADERS_BY_KEY = {
    "gsm8k": download_gsm8k_data,
    "gsm": download_gsm8k_data,
    "c4": download_c4_data,
    "ultrafeedback": download_ultrafeedback_data,
    "humaneval": download_humaneval_data,
    "alpaca": download_alpaca_data,
    "aime2025": download_aime2025_data,
    "aime": download_aime_data,
    "livecodebench": download_livecodebench_code_generation_lite_data,
    "livecodebench_lite": download_livecodebench_code_generation_lite_data,
    "math500": download_math500_data,
    "codeelo": download_codeelo_data,
    "govreport": download_govreport_data,
    "qa": download_natural_questions_data,
    "natural_questions": download_natural_questions_data,
    "natural-questions": download_natural_questions_data,
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    parser.add_argument("--num-samples", type=int, default=None, 
                       help="Number of samples to download (default: all)")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated keys to download, or 'all'. Keys: "
        + ", ".join(sorted(set(DOWNLOADERS_BY_KEY.keys())))
        + " (plus aliases).",
    )
    
    args = parser.parse_args()
    sel = args.datasets.strip().lower()
    if sel in ("", "all"):
        output_files = download_all_datasets(args.num_samples)
    else:
        output_files = {}
        keys = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
        for k in keys:
            fn = DOWNLOADERS_BY_KEY.get(k)
            if fn is None:
                print(f"Unknown dataset key {k!r}, skip. Known: {sorted(set(DOWNLOADERS_BY_KEY.keys()))}")
                output_files[k] = None
                continue
            print(f"\n{'='*50}\nDownloading {k}...\n{'='*50}")
            try:
                output_files[k] = fn(args.num_samples)
                print(f"✓ {k}: {output_files[k]}")
            except Exception as e:
                print(f"✗ {k}: {e}")
                output_files[k] = None
    
    print(f"\n{'='*60}")
    print("Download Summary:")
    print('='*60)
    for name, file_path in output_files.items():
        if file_path:
            print(f"✓ {name}: {file_path}")
        else:
            print(f"✗ {name}: Failed")