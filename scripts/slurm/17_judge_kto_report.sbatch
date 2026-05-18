#!/usr/bin/env bash
#SBATCH --job-name=rl-kto-judge-report
#SBATCH --partition=defq
#SBATCH --exclude=node001
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
mkdir -p logs/slurm results/report_figures

source "${REPO_ROOT}/scripts/slurm/activate_sft_env.sh"

export TOKENIZERS_PARALLELISM=false

SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_report_test}"

echo "Judge/report suite dir: ${SUITE_DIR}"

run_judge_and_metrics() {
  local label="$1"
  local infer_dir="${SUITE_DIR}/${label}/inference"
  local judge_dir="${SUITE_DIR}/${label}/judge"
  local metrics_dir="${SUITE_DIR}/${label}/autometrics"

  if [[ ! -f "${infer_dir}/generated_responses.json" ]]; then
    echo "Missing inference output: ${infer_dir}/generated_responses.json" >&2
    exit 1
  fi

  rm -rf "${judge_dir}" "${metrics_dir}"
  mkdir -p "${judge_dir}" "${metrics_dir}"

  bash scripts/workflow.sh judge-file "${infer_dir}/generated_responses.json" "${judge_dir}"
  bash scripts/workflow.sh auto-metrics "${infer_dir}/generated_responses.json" "${metrics_dir}" "${label}"
}

run_judge_and_metrics "kto"
run_judge_and_metrics "kto_sft"

jupyter nbconvert \
  --to notebook \
  --execute notebooks/report_charts.ipynb \
  --output "${REPO_ROOT}/results/report_figures/report_charts_executed.ipynb" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.cwd="${REPO_ROOT}"
