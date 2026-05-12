#!/usr/bin/env bash
#SBATCH --job-name=rl-prepare
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - job này không xin GPU vì chỉ chuẩn hóa split dữ liệu
# - nếu cluster bắt buộc xin GPU trên defq, thêm "#SBATCH --gres=gpu:1"

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

bash scripts/workflow.sh prepare-data
