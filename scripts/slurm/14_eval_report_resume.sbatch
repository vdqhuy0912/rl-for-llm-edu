#!/usr/bin/env bash
#SBATCH --job-name=rl-eval-report-resume
#SBATCH --partition=defq
#SBATCH --exclude=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=48:00:00
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
mkdir -p logs/slurm results/report_figures

source "${REPO_ROOT}/scripts/slurm/activate_sft_env.sh"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLIT_NAME="${SPLIT_NAME:-test_only}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-1971}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_report_test}"

echo "Resume suite dir: ${SUITE_DIR}"
echo "Split: ${SPLIT_NAME}"
echo "Samples: ${EVAL_NUM_SAMPLES}"

bash scripts/workflow.sh eval-model \
  kto \
  "${REPO_ROOT}/models/kto_base_violation_pair_checkpoints/final" \
  "${SPLIT_NAME}" \
  "${EVAL_NUM_SAMPLES}" \
  "${SUITE_DIR}"

bash scripts/workflow.sh eval-model \
  kto_sft \
  "${REPO_ROOT}/models/kto_violation_pair_checkpoints/final" \
  "${SPLIT_NAME}" \
  "${EVAL_NUM_SAMPLES}" \
  "${SUITE_DIR}"

jupyter nbconvert \
  --to notebook \
  --execute notebooks/report_charts.ipynb \
  --output "${REPO_ROOT}/results/report_figures/report_charts_executed.ipynb" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.cwd="${REPO_ROOT}"
