#!/usr/bin/env bash
#SBATCH --job-name=rl-download
#SBATCH --partition=defq
#SBATCH --nodelist=node002
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# EDIT nhanh:
# - job này không xin GPU vì chỉ tải/materialize dữ liệu
# - nếu cluster bắt buộc xin GPU trên defq, thêm "#SBATCH --gres=gpu:1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/workflow.sh" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
mkdir -p logs/slurm

source "${REPO_ROOT}/scripts/slurm/activate_sft_env.sh"

bash scripts/workflow.sh download-data
