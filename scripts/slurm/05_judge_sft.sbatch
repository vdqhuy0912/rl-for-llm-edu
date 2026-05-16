#!/usr/bin/env bash
#SBATCH --job-name=rl-judge-sft
#SBATCH --partition=defq
#SBATCH --nodelist=node002
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - job này gọi Gemini API, cần GEMINI_API_KEY trong .env
# - không xin GPU vì judge chạy qua API

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/workflow.sh" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs/slurm

source "${REPO_ROOT}/scripts/slurm/activate_sft_env.sh"

bash scripts/workflow.sh judge-file ./results/sft_eval/inference/generated_responses.json ./results/sft_eval/judge
