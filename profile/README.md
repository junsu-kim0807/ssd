# Intermediate verifier profile

This directory contains a **profiling** script used before implementing the full intermediate-verifier speculative decoding pipeline. The goal is to compare the **intermediate verifier** (e.g. Qwen3-4B) and the **target** (e.g. Qwen3-30B-A3B) on:

- **Position-wise acceptance rate**: at each draft position (0..K-1), how often the draft token matches the model’s top-1.
- **Average acceptance length**: average number of consecutive accepted tokens per verify round when using target vs intermediate for the accept decision.

Accept/reject is always decided by the **target** model’s top-1; the script only records what the intermediate would have done for comparison.

**No path setup required**: models and datasets are loaded from HuggingFace by default.

## Models (HuggingFace ids)

- **Draft**: `Qwen/Qwen3-0.6B`
- **Intermediate verifier**: `Qwen/Qwen3-4B`
- **Target**: `Qwen/Qwen3-30B-A3B`

Override with `--draft`, `--intermediate`, `--target` or env vars `SSD_PROFILE_DRAFT_MODEL`, etc.

## Datasets

- **aime25**: `opencompass/AIME2025` (AIME2025-I + AIME2025-II concatenated), prompt style as in the reference script.
- **codeelo**: `Qwen/CodeElo`, competitive programming prompt style.

Install: `pip install datasets`

## Usage

```bash
pip install datasets

# AIME25 only (default), no path or env needed
python profile/run_intermediate_verifier_profile.py --max-samples-aime25 5

# AIME25 + CodeElo
python profile/run_intermediate_verifier_profile.py --datasets aime25,codeelo

# Optional: different devices for target
python profile/run_intermediate_verifier_profile.py \
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

## `batch_profile.py` (batch speculative profile)

- **`--method vanilla`**: accepted as an alias of **`vanila`** (same draft/verify path after normalization).
- **`--hispec`**: intermediate rounds plus interval target commit; supported for **`vanila`** (including the `vanilla` spelling) and **`topk_expansion`**. The latter can branch on multiple first-step draft candidates when expansion is enabled; with **`--expansion-pct 0`** the first-step draft matches vanila greedy behavior for the shared verify layout helpers in code.
