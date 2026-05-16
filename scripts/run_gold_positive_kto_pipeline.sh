#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

find_latest_source_run() {
  local latest
  latest="$(
    find "${REPO_ROOT}/results" -path "${REPO_ROOT}/results/judged_kto_train_*/judge/evaluation_results.json" -print 2>/dev/null \
      | sed -E 's#^.*/judged_kto_train_([^/]+)/judge/evaluation_results\.json$#\1#' \
      | sort \
      | tail -1
  )"
  if [[ -z "${latest}" ]]; then
    echo "Could not infer SOURCE_RUN_NAME. Set SOURCE_RUN_NAME=<run_name>." >&2
    exit 1
  fi
  echo "${latest}"
}

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-$(find_latest_source_run)}"
EVAL_SPLIT="${EVAL_SPLIT:-test_only}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_${SOURCE_RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/results/report_artifacts_${SOURCE_RUN_NAME}}"
GOLD_TRAIN_DATA="${GOLD_TRAIN_DATA:-${REPO_ROOT}/data/splits/kto_judged_gold_train}"
GOLD_VAL_DATA="${GOLD_VAL_DATA:-${REPO_ROOT}/data/splits/kto_judged_gold_val}"
GOLD_OUTPUT_DIR="${GOLD_OUTPUT_DIR:-${REPO_ROOT}/models/kto_gold_checkpoints}"
BACKUP_DIR="${BACKUP_DIR:-${REPO_ROOT}/models/backups/gold_${SOURCE_RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
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

run_eval_gold_model() {
  if [[ -n "${EVAL_NUM_SAMPLES}" ]]; then
    bash scripts/workflow.sh eval-model "sft_kto_gold" "${GOLD_OUTPUT_DIR}/final" "${EVAL_SPLIT}" "${EVAL_NUM_SAMPLES}" "${SUITE_DIR}"
  else
    bash scripts/workflow.sh eval-model "sft_kto_gold" "${GOLD_OUTPUT_DIR}/final" "${EVAL_SPLIT}" "" "${SUITE_DIR}"
  fi
}

main() {
  cd "${REPO_ROOT}"

  local train_eval="${REPO_ROOT}/results/judged_kto_train_${SOURCE_RUN_NAME}/judge/evaluation_results.json"
  local val_eval="${REPO_ROOT}/results/judged_kto_val_${SOURCE_RUN_NAME}/judge/evaluation_results.json"
  require_file "${train_eval}"
  require_file "${val_eval}"

  echo "Source run: ${SOURCE_RUN_NAME}"
  echo "Gold KTO train data: ${GOLD_TRAIN_DATA}"
  echo "Gold KTO val data: ${GOLD_VAL_DATA}"
  echo "Gold KTO output: ${GOLD_OUTPUT_DIR}"
  echo "Eval suite: ${SUITE_DIR}"
  echo "Report artifacts: ${REPORT_DIR}"

  backup_path_if_exists "${GOLD_TRAIN_DATA}"
  backup_path_if_exists "${GOLD_VAL_DATA}"
  backup_path_if_exists "${GOLD_OUTPUT_DIR}"

  echo "Building judged KTO data with reference_answer added as label=True"
  bash scripts/workflow.sh build-judged-kto-data-with-gold \
    "${train_eval}" \
    "${val_eval}" \
    "${GOLD_TRAIN_DATA}" \
    "${GOLD_VAL_DATA}"

  echo "Training KTO gold-positive variant"
  bash scripts/workflow.sh train-kto-judged-gold \
    "${GOLD_TRAIN_DATA}" \
    "${GOLD_VAL_DATA}" \
    "${GOLD_OUTPUT_DIR}"

  echo "Evaluating SFT+KTO gold-positive model"
  run_eval_gold_model

  echo "Rebuilding report artifacts with gold-positive KTO included"
  bash scripts/workflow.sh build-report-artifacts "${SUITE_DIR}" "${REPORT_DIR}"

  echo "Gold-positive KTO pipeline completed"
}

main "$@"
