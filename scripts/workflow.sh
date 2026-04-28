#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/workflow.sh download-data
  scripts/workflow.sh prepare-data
  scripts/workflow.sh train-sft
  scripts/workflow.sh train-kto
  scripts/workflow.sh tmux-wait-gpu <session_name> <command> [args...]
  scripts/workflow.sh eval-model <model_path> [results_dir] [split_name] [num_samples]
  scripts/workflow.sh eval-sft
  scripts/workflow.sh eval-kto
  scripts/workflow.sh eval-all
  scripts/workflow.sh full-pipeline
EOF
}

require_gemini_key() {
  load_local_env
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "GEMINI_API_KEY is not set. Put it in .env or export it before running evaluation." >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    exit 1
  fi
}

run_tmux_wait_gpu() {
  if [[ $# -lt 2 ]]; then
    usage
    exit 1
  fi

  require_command tmux
  require_command nvidia-smi

  local session_name="$1"
  shift

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session already exists: ${session_name}" >&2
    exit 1
  fi

  local interval_sec="${GPU_WAIT_INTERVAL_SEC:-60}"
  local max_memory_mb="${GPU_MAX_MEMORY_MB:-1024}"
  local max_utilization="${GPU_MAX_UTILIZATION:-10}"
  local command_string=""
  printf -v command_string '%q ' "$@"
  command_string="${command_string% }"

  local runner_script
  printf -v runner_script '%q ' \
    "set -euo pipefail" \
    "while true; do" \
    "GPU_INDEX=\$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v max_mem=${max_memory_mb} -v max_util=${max_utilization} '{gsub(/^[ \t]+|[ \t]+$/, \"\", \$1); gsub(/^[ \t]+|[ \t]+$/, \"\", \$2); gsub(/^[ \t]+|[ \t]+$/, \"\", \$3); if (\$2+0 <= max_mem && \$3+0 <= max_util) { print \$1; exit }}')" \
    "if [[ -n \"\${GPU_INDEX:-}\" ]]; then" \
    "echo \"[\$(date '+%F %T')] GPU \${GPU_INDEX} is available. Starting command.\"" \
    "cd ${REPO_ROOT}" \
    "export CUDA_VISIBLE_DEVICES=\"\${GPU_INDEX}\"" \
    "${command_string}" \
    "break" \
    "fi" \
    "echo \"[\$(date '+%F %T')] Waiting for GPU with memory.used<=${max_memory_mb}MB and utilization<=${max_utilization}% ...\"" \
    "sleep ${interval_sec}" \
    "done"
  runner_script="${runner_script% }"

  tmux new-session -d -s "${session_name}" "bash -lc ${runner_script}"
  echo "Started tmux session: ${session_name}"
  echo "Attach with: tmux attach -t ${session_name}"
  echo "Thresholds: GPU_MAX_MEMORY_MB=${max_memory_mb}, GPU_MAX_UTILIZATION=${max_utilization}, GPU_WAIT_INTERVAL_SEC=${interval_sec}"
}

run_eval_model() {
  if [[ $# -lt 1 || $# -gt 4 ]]; then
    usage
    exit 1
  fi

  local model_path="$1"
  local results_dir="${2:-${REPO_ROOT}/results/$(basename "${model_path}")_eval}"
  local split_name="${3:-test_only}"
  local num_samples="${4:-}"
  require_gemini_key
  if [[ -n "${num_samples}" ]]; then
    run_cli_module src.cli.run_eval --model-path "${model_path}" --results-dir "${results_dir}" --split-name "${split_name}" --num-samples "${num_samples}"
  else
    run_cli_module src.cli.run_eval --model-path "${model_path}" --results-dir "${results_dir}" --split-name "${split_name}"
  fi
}

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  usage
  exit 1
fi
shift || true

case "${command_name}" in
  download-data)
    run_cli_module src.cli.download_data "$@"
    ;;
  prepare-data)
    run_cli_module src.cli.prepare_data "$@"
    ;;
  train-sft)
    run_cli_module src.cli.run_sft "$@"
    ;;
  train-kto)
    run_cli_module src.cli.run_kto "$@"
    ;;
  tmux-wait-gpu)
    run_tmux_wait_gpu "$@"
    ;;
  eval-model)
    run_eval_model "$@"
    ;;
  eval-sft)
    run_eval_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval" "test_only"
    ;;
  eval-kto)
    run_eval_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval" "kto_test"
    ;;
  eval-all)
    if [[ -d "${REPO_ROOT}/models/sft_checkpoints/final" ]]; then
      run_eval_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval" "test_only"
    fi
    if [[ -d "${REPO_ROOT}/models/kto_checkpoints/final" ]]; then
      run_eval_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval" "kto_test"
    fi
    ;;
  full-pipeline)
    run_cli_module src.cli.download_data
    run_cli_module src.cli.prepare_data
    run_cli_module src.cli.run_sft
    run_eval_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval" "test_only"
    run_cli_module src.cli.run_kto
    run_eval_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval" "kto_test"
    ;;
  *)
    usage
    exit 1
    ;;
esac
