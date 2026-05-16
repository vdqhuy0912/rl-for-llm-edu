#!/usr/bin/env bash
#SBATCH --job-name=rl-custom
#SBATCH --partition=defq
#SBATCH --nodelist=node002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - dùng để chạy nhanh một command workflow bất kỳ
# - ví dụ:
#   sbatch --export=ALL,WORKFLOW_ARGS="infer-model ./models/sft_checkpoints/final ./results/tmp test_only 20" scripts/slurm/09_custom_workflow.sbatch
# - nếu command không cần GPU, xóa dòng "#SBATCH --gres=gpu:1"

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

if [[ -z "${WORKFLOW_ARGS:-}" ]]; then
  echo "Missing WORKFLOW_ARGS."
  echo "Example:"
  echo '  sbatch --export=ALL,WORKFLOW_ARGS="prepare-data" scripts/slurm/09_custom_workflow.sbatch'
  exit 1
fi

# shellcheck disable=SC2086
bash scripts/workflow.sh ${WORKFLOW_ARGS}
