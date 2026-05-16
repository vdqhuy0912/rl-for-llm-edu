#!/usr/bin/env bash
#SBATCH --job-name=rl-train-sft
#SBATCH --partition=defq
#SBATCH --nodelist=node002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - tăng GPU: đổi "--gres=gpu:1" thành "--gres=gpu:2" nếu code/config hỗ trợ
# - nếu OOM RAM CPU: tăng "--mem=80G"
# - nếu hết time: tăng "--time=24:00:00"

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

bash scripts/workflow.sh train-sft
