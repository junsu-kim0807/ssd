# Intermediate verifier profile (AIME 2025)

This directory contains a **profiling** script used before implementing the full intermediate-verifier speculative decoding pipeline. The goal is to compare the **intermediate verifier** (e.g. Qwen3-4B) and the **target** (e.g. Qwen3-30B-A3B) on:

- **Position-wise acceptance rate**: at each draft position (0..K-1), how often the draft token matches the model’s top-1.
- **Average acceptance length**: average number of consecutive accepted tokens per verify round when using target vs intermediate for the accept decision.

Accept/reject is always decided by the **target** model’s top-1; the script only records what the intermediate would have done for comparison.

## Models

- **Draft**: Qwen3-0.6B  
- **Intermediate verifier**: Qwen3-4B  
- **Target**: Qwen3-30B-A3B  

Paths can be overridden via `--draft`, `--intermediate`, `--target` or env vars `SSD_PROFILE_DRAFT_MODEL`, `SSD_PROFILE_INTERMEDIATE_MODEL`, `SSD_PROFILE_TARGET_MODEL`.

## Dataset

Evaluation uses **AIME 2025** (30 problems). Either:

- Set `SSD_DATASET_DIR` and run `scripts/get_data_from_hf.py` so that `aime2025/aime2025_test.jsonl` (or `aime2025_test.jsonl`) exists under that dir, or  
- Install `datasets` and the script will load `math-ai/aime25` from Hugging Face.

## Usage

```bash
# Optional: install datasets if loading AIME from HuggingFace
pip install datasets

# Optional: download AIME 2025 to SSD_DATASET_DIR
python scripts/get_data_from_hf.py  # includes AIME2025

# Run profile (override paths as needed)
python profile/run_intermediate_verifier_profile.py \
  --draft /path/to/Qwen3-0.6B \
  --intermediate /path/to/Qwen3-4B \
  --target /path/to/Qwen3-30B-A3B \
  --output-dir profile/results \
  --k 5 \
  --max-new-tokens 256 \
  --max-samples 5 \
  --chat-template
```

Target을 다른 GPU에 두어도 됩니다 (모델 간 데이터는 CPU 리스트/스칼라로만 전달됩니다):

```bash
  --device-draft cuda:0 --device-intermediate cuda:0 --device-target cuda:1
```

Outputs under `--output-dir`:

- `intermediate_vs_target_summary.json`: position-wise accept rates (target vs intermediate), avg acceptance length, config, and:
  - **1) Same accept length**: `prob_same_accept_length` (intermediate와 target이 같은 accept length를 가질 확률), `prob_bonus_same_given_same_length` (같은 length일 때 둘이 만든 bonus token이 같을 확률).
  - **2) Different accept length**: `prob_inter_bonus_in_next_draft` (accept length가 다를 때, intermediate의 bonus token이 다음 draft에 포함될 확률), `prob_inter_bonus_accept_next_target` (그 bonus token이 다음 target verification에서 accept될 확률).
- `per_position_top5_detail.jsonl` (if `--save-per-position-detail`): per round, per position, draft token id, intermediate/target top-5, and accept flags.

## Flow (per round)

1. **Prefill**: target runs on current prompt → first token (recovery).
2. **Draft**: draft model runs autoregressively for K steps from (prompt + recovery) → K draft tokens.
3. **Verify (serial)**: intermediate model forward on (prompt + recovery + draft_0..draft_{K-1}) → logits at K+1 positions; then target same → logits at K+1 positions.
4. **Stats**: for each position j, compare draft token with intermediate top-5 and target top-5; record accept by target top-1 and by intermediate top-1.
5. **Advance**: accept/reject by **target** top-1 only; append accepted tokens and set recovery; repeat until `max_new_tokens` or EOS.

This gives position-wise and aggregate stats to see how well the intermediate verifier approximates the target before wiring it into the real GPU0/GPU1 pipeline.
