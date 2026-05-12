#!/usr/bin/env bash
#SBATCH --job-name=rl-train-sft
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - tăng GPU: đổi "--gres=gpu:1" thành "--gres=gpu:2" nếu code/config hỗ trợ
# - ép node đang còn GPU: thêm "#SBATCH --nodelist=node002"
# - nếu OOM RAM CPU: tăng "--mem=80G"
# - nếu hết time: tăng "--time=24:00:00"

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

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/workflow.sh train-sft
