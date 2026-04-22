from transformers import AutoTokenizer


# Infer model family based on model path name
def infer_model_family(model_path: str) -> str:
        """Infer if model is Llama or Qwen based on path name."""
        model_path_lower = model_path.lower()
        if "vicuna" in model_path_lower:
            return "vicuna"
        elif "llama" in model_path_lower:
            return "llama"
        elif "qwen" in model_path_lower:
            return "qwen"
        elif "gemma" in model_path_lower:
            return "gemma"
        else:
            return "unknown"


def load_auto_tokenizer(
    model_or_path: str,
    *,
    tokenizer_path: str | None = None,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """Load tokenizer; avoid forced fast path for Vicuna / LLaMA-style repos (sentencepiece)."""
    name = tokenizer_path if tokenizer_path else model_or_path
    lower = str(name).lower()
    prefer_slow = ("vicuna" in lower) or ("llama" in lower)
    try:
        return AutoTokenizer.from_pretrained(
            name,
            use_fast=not prefer_slow,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            name,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )


def decode_tokens(token_ids: list[int], tokenizer: AutoTokenizer) -> list[str]:
    decoded = []
    for token in token_ids:
        try:
            text = tokenizer.decode([token], skip_special_tokens=False)
            decoded.append(text)
        except Exception:
            decoded.append(f"<token_id:{token}>")
    return decoded
