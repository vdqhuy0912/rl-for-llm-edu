#!/usr/bin/env bash
#SBATCH --job-name=rl-eval-report-suite
#SBATCH --partition=defq
#SBATCH --exclude=node001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
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
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_report_test}"

echo "Suite dir: ${SUITE_DIR}"
echo "Split: ${SPLIT_NAME}"
echo "Samples: ${EVAL_NUM_SAMPLES:-all}"

bash scripts/workflow.sh eval-report-suite "${SPLIT_NAME}" "${EVAL_NUM_SAMPLES}" "${SUITE_DIR}"

REPORT_NOTEBOOK="${REPORT_NOTEBOOK:-${REPO_ROOT}/notebooks/report_charts.ipynb}"
REPORT_EXECUTED_NOTEBOOK="${REPORT_EXECUTED_NOTEBOOK:-${REPO_ROOT}/results/report_figures/report_charts_executed.ipynb}"

mkdir -p "$(dirname "${REPORT_EXECUTED_NOTEBOOK}")"
jupyter nbconvert \
  --to notebook \
  --execute "${REPORT_NOTEBOOK}" \
  --output "${REPORT_EXECUTED_NOTEBOOK}" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.cwd="${REPO_ROOT}"
