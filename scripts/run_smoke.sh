#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z "${NVCC:-}" && -x "/usr/local/cuda/bin/nvcc" ]]; then
  export NVCC="/usr/local/cuda/bin/nvcc"
fi

if [[ -d "${ROOT_DIR}/.deps" ]]; then
  if [[ -f "${ROOT_DIR}/.deps/.python-version" ]]; then
    deps_python_version="$(<"${ROOT_DIR}/.deps/.python-version")"
    run_python_version="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    if [[ "${deps_python_version}" != "${run_python_version}" ]]; then
      echo "error: .deps was built for Python ${deps_python_version}, but PYTHON_BIN=${PYTHON_BIN} is Python ${run_python_version}." >&2
      echo "Set PYTHON_BIN to the same interpreter or rerun scripts/setup_env.sh." >&2
      exit 1
    fi
  fi
  export PYTHONPATH="${ROOT_DIR}/.deps:${PYTHONPATH:-}"
fi

DTYPE_A="${1:-f16}"
DTYPE_B="${2:-f16}"
DTYPE_C="${3:-f16}"
STYPE="${4:-}"

export MMA_SIM_TEST_CASES="${MMA_SIM_TEST_CASES:-16}"

cd "${ROOT_DIR}/cmodel"

if [[ -n "${STYPE}" ]]; then
  "${PYTHON_BIN}" nv_result_cmp_cmodel.py "${DTYPE_A}" "${DTYPE_B}" "${DTYPE_C}" "${STYPE}"
else
  "${PYTHON_BIN}" nv_result_cmp_cmodel.py "${DTYPE_A}" "${DTYPE_B}" "${DTYPE_C}"
fi
