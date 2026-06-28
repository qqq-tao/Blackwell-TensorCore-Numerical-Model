#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NVCC_BIN="${NVCC:-nvcc}"
USE_SYSTEM_PYTHON="${USE_SYSTEM_PYTHON:-0}"

cd "${ROOT_DIR}"

USE_VENV=0
if [[ "${USE_SYSTEM_PYTHON}" != "1" ]]; then
  if [[ -x .venv/bin/python ]]; then
    USE_VENV=1
  elif "${PYTHON_BIN}" -m venv .venv; then
    USE_VENV=1
  else
    rm -rf .venv
    echo "python venv is unavailable; falling back to .deps with pip --target" >&2
  fi
fi

if [[ "${USE_VENV}" == "1" ]]; then
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r cmodel/requirements.txt
  RUN_PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  DEPS_DIR="${DEPS_DIR:-${ROOT_DIR}/.deps}"
  mkdir -p "${DEPS_DIR}"
  "${PYTHON_BIN}" -m pip install --upgrade --target "${DEPS_DIR}" -r cmodel/requirements.txt
  "${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' > "${DEPS_DIR}/.python-version"
  export PYTHONPATH="${DEPS_DIR}:${PYTHONPATH:-}"
  RUN_PYTHON="${PYTHON_BIN}"
fi

cd cmodel/benchmark_values/fp_cvt
make clean
make PYTHON="${RUN_PYTHON}" NVCC="${NVCC_BIN}"
