#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-}"
if [[ -z "${SOURCE_RUN_NAME}" ]]; then
  SOURCE_RUN_NAME="$(
    find "${REPO_ROOT}/results" -path "${REPO_ROOT}/results/judged_kto_train_*/generated_responses.json" -print 2>/dev/null \
      | sed -E 's#^.*/judged_kto_train_([^/]+)/generated_responses\.json$#\1#' \
      | sort \
      | tail -1
  )"
fi
if [[ -z "${SOURCE_RUN_NAME}" ]]; then
  echo "Could not infer SOURCE_RUN_NAME. Set SOURCE_RUN_NAME=<run_name>." >&2
  exit 1
fi

EVAL_SPLIT="${EVAL_SPLIT:-test_only}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-}"
SUITE_DIR="${SUITE_DIR:-${REPO_ROOT}/results/model_suite_${SOURCE_RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/results/report_artifacts_${SOURCE_RUN_NAME}}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

main() {
  cd "${REPO_ROOT}"

  local train_generated="${REPO_ROOT}/results/judged_kto_train_${SOURCE_RUN_NAME}/generated_responses.json"
  local val_generated="${REPO_ROOT}/results/judged_kto_val_${SOURCE_RUN_NAME}/generated_responses.json"
  require_file "${train_generated}"
  require_file "${val_generated}"

  echo "Source run: ${SOURCE_RUN_NAME}"
  echo "Eval suite: ${SUITE_DIR}"
  echo "Report artifacts: ${REPORT_DIR}"

  echo "Resuming judge for KTO train responses"
  bash scripts/workflow.sh judge-file \
    "${train_generated}" \
    "${REPO_ROOT}/results/judged_kto_train_${SOURCE_RUN_NAME}/judge"

  echo "Resuming judge for KTO val responses"
  bash scripts/workflow.sh judge-file \
    "${val_generated}" \
    "${REPO_ROOT}/results/judged_kto_val_${SOURCE_RUN_NAME}/judge"

  echo "Building judged KTO train/val datasets"
  bash scripts/workflow.sh build-judged-kto-data \
    "${REPO_ROOT}/results/judged_kto_train_${SOURCE_RUN_NAME}/judge/evaluation_results.json" \
    "${REPO_ROOT}/results/judged_kto_val_${SOURCE_RUN_NAME}/judge/evaluation_results.json" \
    "${REPO_ROOT}/data/splits/kto_judged_train" \
    "${REPO_ROOT}/data/splits/kto_judged_val"

  echo "Training KTO on judged generated-answer signal"
  bash scripts/workflow.sh train-kto-judged \
    "${REPO_ROOT}/data/splits/kto_judged_train" \
    "${REPO_ROOT}/data/splits/kto_judged_val"

  echo "Evaluating Qwen3-8B, SFT final, and SFT+KTO final"
  bash scripts/workflow.sh eval-model-suite "${EVAL_SPLIT}" "${EVAL_NUM_SAMPLES}" "${SUITE_DIR}"

  echo "Building report artifacts"
  bash scripts/workflow.sh build-report-artifacts "${SUITE_DIR}" "${REPORT_DIR}"

  echo "Resume pipeline completed"
}

main "$@"
