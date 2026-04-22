#!/bin/bash
#SBATCH --job-name=bench_qwen_hierarchical_b1_k6_t0_multids_bl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=def-pnair
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=128G
#SBATCH --time=04:00:00
#SBATCH --output=/Users/junsu/ssd/profile/scripts/logs/bench_profile/out/cost_metadata/qwen/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/multids_batch_length/bench_qwen_hierarchical_b1_k6_t0_multids_bl.out
#SBATCH --error=/Users/junsu/ssd/profile/scripts/logs/bench_profile/err/cost_metadata/qwen/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/multids_batch_length/bench_qwen_hierarchical_b1_k6_t0_multids_bl.err

set -eo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR='/project/def-pnair/junsu/kv_cache/ssd'
VENV_DIR='/project/def-pnair/junsu/kv_cache/.venv'
BENCH_DIR="${REPO_DIR}/bench"

cd "${REPO_DIR}"
export BASHRCSOURCED="${BASHRCSOURCED:-1}"
set +u
source "${VENV_DIR}/bin/activate"
if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi
set -u

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d "${HOME}/scratch/huggingface" ]]; then
    export HF_HOME="${HOME}/scratch/huggingface"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi
fi

export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${BENCH_DIR}"
mkdir -p "${BENCH_DIR}/results/cost_metadata/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/alpaca"
mkdir -p "${BENCH_DIR}/results/cost_metadata/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/humaneval"
mkdir -p "${BENCH_DIR}/results/cost_metadata/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/gsm8k"
mkdir -p "${BENCH_DIR}/results/cost_metadata/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/math500"
mkdir -p "${BENCH_DIR}/results/cost_metadata/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0/codeelo"

PROFILE_BASE='./results/cost_metadata/hierarchical/b1/k6/Qwen3-32B+Qwen3-0.6B/t0'
for dataset in alpaca humaneval gsm8k math500 codeelo; do
  echo "==== bench profile dataset=${dataset} ===="
  case "${dataset}" in
    alpaca) EXTRA_DS=(--alpaca);;
    humaneval) EXTRA_DS=(--humaneval);;
    gsm8k) EXTRA_DS=();;
    math500) EXTRA_DS=(--math500);;
    codeelo) EXTRA_DS=(--codeelo);;
    *) echo "unknown dataset: ${dataset}" >&2; exit 1;;
  esac
  python -O bench.py \
    --qwen \
    --gpus 2 \
    --b 1 \
    --temp 0.0 \
    --numseqs 512 \
    --output_len 2048 \
    "${EXTRA_DS[@]}" \
    --spec \
    --k \
    6 \
    --spec_policy \
    hierarchical \
    --profile \
    --profile_mode cost_metadata \
    --profiler_output_dir "${PROFILE_BASE}/${dataset}"
done
