#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_SFT_ENV_PATH="${SFT_ENV_PATH:-${HOME}/.conda/envs/SFT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${DEFAULT_SFT_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${DEFAULT_SFT_ENV_PATH}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python was not found in PATH." >&2
  exit 1
fi

load_local_env() {
  local env_file="${REPO_ROOT}/.env"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

run_repo_python() {
  (cd "${REPO_ROOT}" && "${PYTHON_BIN}" "$@")
}

run_cli_module() {
  local module="$1"
  shift
  run_repo_python -m "${module}" "$@"
}
