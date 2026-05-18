#!/usr/bin/env bash
#SBATCH --job-name=rl-kto-infer-tf
#SBATCH --partition=defq
#SBATCH --exclude=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/workflow.sh" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs/slurm

source "${REPO_ROOT}/scripts/slurm/activate_sft_env.sh"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT_NAME="${SPLIT_NAME:-test_only}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-1971}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_report_test}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

echo "Transformers KTO inference suite dir: ${SUITE_DIR}"
echo "Split: ${SPLIT_NAME}"
echo "Samples: ${EVAL_NUM_SAMPLES}"
echo "Initial generation batch size: ${INFER_BATCH_SIZE}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"

run_infer() {
  local label="$1"
  local model_path="$2"
  local infer_dir="${SUITE_DIR}/${label}/inference"

  rm -rf "${infer_dir}"
  mkdir -p "${infer_dir}"

  python -m src.cli.run_infer \
    --model-path "${model_path}" \
    --results-dir "${infer_dir}" \
    --split-name "${SPLIT_NAME}" \
    --num-samples "${EVAL_NUM_SAMPLES}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
}

run_infer "kto" "${REPO_ROOT}/models/kto_base_violation_pair_checkpoints/final"
run_infer "kto_sft" "${REPO_ROOT}/models/kto_violation_pair_checkpoints/final"
