#!/usr/bin/env bash
#SBATCH --job-name=rl-judge-sft
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - job này gọi Gemini API, cần GEMINI_API_KEY trong .env
# - không xin GPU vì judge chạy qua API

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-${REPO_ROOT}}"
mkdir -p logs/slurm

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-rl-llm-edu}"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-rl-llm-edu}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME:-rl-llm-edu}"
fi

bash scripts/workflow.sh judge-file ./results/sft_eval/inference/generated_responses.json ./results/sft_eval/judge
