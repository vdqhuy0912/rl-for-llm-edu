#!/usr/bin/env bash
set -euo pipefail

SFT_ENV_PATH="${SFT_ENV_PATH:-${HOME}/.conda/envs/SFT}"

if [[ ! -x "${SFT_ENV_PATH}/bin/python" ]]; then
  echo "SFT environment not found at ${SFT_ENV_PATH}" >&2
  echo "Set SFT_ENV_PATH to the correct conda environment path before submitting." >&2
  exit 1
fi

export CONDA_PREFIX="${SFT_ENV_PATH}"
export CONDA_DEFAULT_ENV="$(basename "${SFT_ENV_PATH}")"
export VIRTUAL_ENV="${SFT_ENV_PATH}"
export PATH="${SFT_ENV_PATH}/bin:${PATH}"

hash -r
echo "Using Python: $(command -v python)"
python --version
