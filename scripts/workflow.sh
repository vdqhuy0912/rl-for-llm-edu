#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/workflow.sh download-data
  scripts/workflow.sh preprocess-data
  scripts/workflow.sh split-data
  scripts/workflow.sh train-sft
  scripts/workflow.sh train-kto
  scripts/workflow.sh eval-model <model_path> [results_dir]
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

run_eval_model() {
  if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
  fi

  local model_path="$1"
  local results_dir="${2:-${REPO_ROOT}/results/$(basename "${model_path}")_eval}"
  require_gemini_key
  run_cli_module src.cli.run_eval --model-path "${model_path}" --results-dir "${results_dir}"
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
  preprocess-data)
    run_cli_module src.cli.process_data --normalize-only "$@"
    ;;
  split-data)
    run_cli_module src.cli.process_data --split-only "$@"
    ;;
  train-sft)
    run_cli_module src.cli.run_sft "$@"
    ;;
  train-kto)
    run_cli_module src.cli.run_kto "$@"
    ;;
  eval-model)
    run_eval_model "$@"
    ;;
  eval-sft)
    run_eval_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval"
    ;;
  eval-kto)
    run_eval_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval"
    ;;
  eval-all)
    if [[ -d "${REPO_ROOT}/models/sft_checkpoints/final" ]]; then
      run_eval_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval"
    fi
    if [[ -d "${REPO_ROOT}/models/kto_checkpoints/final" ]]; then
      run_eval_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval"
    fi
    ;;
  full-pipeline)
    run_cli_module src.cli.download_data
    run_cli_module src.cli.process_data --normalize-only
    run_cli_module src.cli.process_data --split-only
    run_cli_module src.cli.run_sft
    run_eval_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval"
    run_cli_module src.cli.run_kto
    run_eval_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval"
    ;;
  *)
    usage
    exit 1
    ;;
esac
