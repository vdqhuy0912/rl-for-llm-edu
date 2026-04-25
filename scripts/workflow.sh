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
