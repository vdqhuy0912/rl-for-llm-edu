#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
EVAL_SPLIT="${EVAL_SPLIT:-test_only}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"
KTO_SIGNAL_TRAIN_SPLIT="${KTO_SIGNAL_TRAIN_SPLIT:-kto_train}"
KTO_SIGNAL_VAL_SPLIT="${KTO_SIGNAL_VAL_SPLIT:-kto_val}"
KTO_SIGNAL_TRAIN_SAMPLES="${KTO_SIGNAL_TRAIN_SAMPLES:-}"
KTO_SIGNAL_VAL_SAMPLES="${KTO_SIGNAL_VAL_SAMPLES:-}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/results/report_artifacts_${RUN_NAME}}"
BACKUP_DIR="${BACKUP_DIR:-${REPO_ROOT}/models/backups/${RUN_NAME}}"

require_gemini_key() {
  load_local_env
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "GEMINI_API_KEY is not set. Put it in .env or export it before running LLM-as-judge steps." >&2
    exit 1
  fi
}

backup_path_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    mkdir -p "${BACKUP_DIR}"
    mv "${path}" "${BACKUP_DIR}/$(basename "${path}")"
  fi
}

run_infer_for_signal() {
  local split_name="$1"
  local output_dir="$2"
  local sample_limit="$3"
  if [[ -n "${sample_limit}" ]]; then
    bash scripts/workflow.sh infer-model ./models/sft_checkpoints/final "${output_dir}" "${split_name}" "${sample_limit}"
  else
    bash scripts/workflow.sh infer-model ./models/sft_checkpoints/final "${output_dir}" "${split_name}"
  fi
}

main() {
  cd "${REPO_ROOT}"
  require_gemini_key
  mkdir -p results logs models

  echo "Run name: ${RUN_NAME}"
  echo "Eval suite: ${SUITE_DIR}"
  echo "Report artifacts: ${REPORT_DIR}"

  echo "Backing up previous train outputs if present"
  backup_path_if_exists "${REPO_ROOT}/models/sft_checkpoints"
  backup_path_if_exists "${REPO_ROOT}/models/kto_checkpoints"
  backup_path_if_exists "${REPO_ROOT}/data/splits/kto_judged_train"
  backup_path_if_exists "${REPO_ROOT}/data/splits/kto_judged_val"

  echo "Preparing deterministic data splits"
  bash scripts/workflow.sh prepare-data

  echo "Training SFT from scratch"
  bash scripts/workflow.sh train-sft

  echo "Generating SFT responses for KTO signal train/val"
  run_infer_for_signal "${KTO_SIGNAL_TRAIN_SPLIT}" "${REPO_ROOT}/results/judged_kto_train_${RUN_NAME}" "${KTO_SIGNAL_TRAIN_SAMPLES}"
  run_infer_for_signal "${KTO_SIGNAL_VAL_SPLIT}" "${REPO_ROOT}/results/judged_kto_val_${RUN_NAME}" "${KTO_SIGNAL_VAL_SAMPLES}"

  echo "Judging SFT train/val responses with LLM-as-judge"
  bash scripts/workflow.sh judge-file \
    "${REPO_ROOT}/results/judged_kto_train_${RUN_NAME}/generated_responses.json" \
    "${REPO_ROOT}/results/judged_kto_train_${RUN_NAME}/judge"
  bash scripts/workflow.sh judge-file \
    "${REPO_ROOT}/results/judged_kto_val_${RUN_NAME}/generated_responses.json" \
    "${REPO_ROOT}/results/judged_kto_val_${RUN_NAME}/judge"

  echo "Building judged KTO train/val datasets"
  bash scripts/workflow.sh build-judged-kto-data \
    "${REPO_ROOT}/results/judged_kto_train_${RUN_NAME}/judge/evaluation_results.json" \
    "${REPO_ROOT}/results/judged_kto_val_${RUN_NAME}/judge/evaluation_results.json" \
    "${REPO_ROOT}/data/splits/kto_judged_train" \
    "${REPO_ROOT}/data/splits/kto_judged_val"

  echo "Training KTO on LLM-as-judge signal data"
  bash scripts/workflow.sh train-kto-judged \
    "${REPO_ROOT}/data/splits/kto_judged_train" \
    "${REPO_ROOT}/data/splits/kto_judged_val"

  echo "Evaluating Qwen3-8B, SFT final, and SFT+KTO final"
  bash scripts/workflow.sh eval-model-suite "${EVAL_SPLIT}" "${EVAL_NUM_SAMPLES}" "${SUITE_DIR}"

  echo "Building charts, tables, and report draft"
  bash scripts/workflow.sh build-report-artifacts "${SUITE_DIR}" "${REPORT_DIR}"

  echo "Pipeline completed"
  echo "Suite dir: ${SUITE_DIR}"
  echo "Report dir: ${REPORT_DIR}"
}

main "$@"
