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
  scripts/workflow.sh build-judged-kto-data <train_evaluation_results.json> <val_evaluation_results.json> [train_output_dir] [val_output_dir]
  scripts/workflow.sh train-kto-judged [train_data_dir] [val_data_dir]
  scripts/workflow.sh infer-model <model_path> [results_dir] [split_name] [num_samples]
  scripts/workflow.sh judge-file <input_path> [results_dir]
  scripts/workflow.sh rerun-oom-judge <generated_responses.json> [evaluation_results.json] [results_dir] [model_path]
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

run_infer_model() {
  if [[ $# -lt 1 || $# -gt 4 ]]; then
    usage
    exit 1
  fi

  local model_path="$1"
  local results_dir="${2:-${REPO_ROOT}/results/$(basename "${model_path}")_infer}"
  local split_name="${3:-test_only}"
  local num_samples="${4:-}"
  if [[ -n "${num_samples}" ]]; then
    run_cli_module src.cli.run_infer --model-path "${model_path}" --results-dir "${results_dir}" --split-name "${split_name}" --num-samples "${num_samples}"
  else
    run_cli_module src.cli.run_infer --model-path "${model_path}" --results-dir "${results_dir}" --split-name "${split_name}"
  fi
}

run_judge_file() {
  if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
  fi

  local input_path="$1"
  local results_dir="${2:-${REPO_ROOT}/results/judge_results}"
  require_gemini_key
  run_cli_module src.cli.run_judge --input-path "${input_path}" --results-dir "${results_dir}"
}

run_build_judged_kto_data() {
  if [[ $# -lt 2 || $# -gt 4 ]]; then
    usage
    exit 1
  fi

  local train_input="$1"
  local val_input="$2"
  local train_output="${3:-${REPO_ROOT}/data/splits/kto_judged_train}"
  local val_output="${4:-${REPO_ROOT}/data/splits/kto_judged_val}"

  run_cli_module src.cli.build_judged_kto_data --input-path "${train_input}" --output-dir "${train_output}"
  run_cli_module src.cli.build_judged_kto_data --input-path "${val_input}" --output-dir "${val_output}"
}

run_train_kto_judged() {
  if [[ $# -gt 2 ]]; then
    usage
    exit 1
  fi

  local train_data="${1:-${REPO_ROOT}/data/splits/kto_judged_train}"
  local val_data="${2:-${REPO_ROOT}/data/splits/kto_judged_val}"

  run_cli_module src.cli.run_kto --train-kto-data "${train_data}" --eval-kto-data "${val_data}"
}

run_rerun_oom_judge() {
  if [[ $# -lt 1 || $# -gt 4 ]]; then
    usage
    exit 1
  fi

  local generated_path="$1"
  local evaluation_path="${2:-}"
  local results_dir="${3:-${REPO_ROOT}/results/oom_rerun}"
  local model_path="${4:-${REPO_ROOT}/models/sft_checkpoints/final}"

  require_gemini_key
  if [[ -n "${evaluation_path}" && "${evaluation_path}" != "-" ]]; then
    run_cli_module src.cli.rerun_oom_judge \
      --generated-path "${generated_path}" \
      --evaluation-path "${evaluation_path}" \
      --results-dir "${results_dir}" \
      --model-path "${model_path}"
  else
    run_cli_module src.cli.rerun_oom_judge \
      --generated-path "${generated_path}" \
      --results-dir "${results_dir}" \
      --model-path "${model_path}"
  fi
}

run_infer_and_judge_model() {
  if [[ $# -lt 1 || $# -gt 4 ]]; then
    usage
    exit 1
  fi

  local model_path="$1"
  local results_dir="${2:-${REPO_ROOT}/results/$(basename "${model_path}")_eval}"
  local split_name="${3:-test_only}"
  local num_samples="${4:-}"
  local infer_dir="${results_dir}/inference"
  local judge_dir="${results_dir}/judge"

  run_infer_model "${model_path}" "${infer_dir}" "${split_name}" "${num_samples}"
  run_judge_file "${infer_dir}/generated_responses.json" "${judge_dir}"
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
  build-judged-kto-data)
    run_build_judged_kto_data "$@"
    ;;
  train-kto-judged)
    run_train_kto_judged "$@"
    ;;
  infer-model)
    run_infer_model "$@"
    ;;
  judge-file)
    run_judge_file "$@"
    ;;
  rerun-oom-judge)
    run_rerun_oom_judge "$@"
    ;;
  eval-sft)
    run_infer_and_judge_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval" "test_only"
    ;;
  eval-kto)
    run_infer_and_judge_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval" "kto_test"
    ;;
  eval-all)
    if [[ -d "${REPO_ROOT}/models/sft_checkpoints/final" ]]; then
      run_infer_and_judge_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval" "test_only"
    fi
    if [[ -d "${REPO_ROOT}/models/kto_checkpoints/final" ]]; then
      run_infer_and_judge_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval" "kto_test"
    fi
    ;;
  full-pipeline)
    run_cli_module src.cli.download_data
    run_cli_module src.cli.prepare_data
    run_cli_module src.cli.run_sft
    run_infer_and_judge_model "${REPO_ROOT}/models/sft_checkpoints/final" "${REPO_ROOT}/results/sft_eval" "test_only"
    run_cli_module src.cli.run_kto
    run_infer_and_judge_model "${REPO_ROOT}/models/kto_checkpoints/final" "${REPO_ROOT}/results/kto_eval" "kto_test"
    ;;
  *)
    usage
    exit 1
    ;;
esac
