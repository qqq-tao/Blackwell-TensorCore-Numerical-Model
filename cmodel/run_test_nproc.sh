#!/usr/bin/env bash

# Shell 脚本：并行启动 nv_result_cmp_cmodel.py 测试并计时。
# 默认按发布验证口径运行 2^16 cases/组合，并按可见 GPU 轮转绑定任务。

set -euo pipefail

# --- 配置 ---
if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
  source "${VIRTUAL_ENV}/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

DEFAULT_TEST_CASES=$((1 << 16))
export MMA_SIM_TEST_CASES="${MMA_SIM_TEST_CASES:-${DEFAULT_TEST_CASES}}"
MMA_SIM_COVERAGE="${MMA_SIM_COVERAGE:-0}"

# Python 脚本的名称或路径
PYTHON_SCRIPT="nv_result_cmp_cmodel.py"
export PYTHON_BIN PYTHON_SCRIPT

COMBINATIONS=(
  "f16 f16 f32"
  "e4m3 e4m3 f32"
  "e5m2 e5m2 f32"
  "e4m3 e5m2 f32"
  "e5m2 e4m3 f32"
  "tf32 tf32 f32"
  "f16 f16 f16"
  "bf16 bf16 f32"
)

detect_gpu_csv() {
  if [[ -n "${MMA_SIM_GPU_LIST:-}" ]]; then
    printf '%s\n' "${MMA_SIM_GPU_LIST}"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | tr -d '[:space:]' | paste -sd, -
  else
    printf '0\n'
  fi
}

GPU_CSV="$(detect_gpu_csv)"
GPU_CSV="${GPU_CSV// /}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_CSV}"

if [[ "${#GPU_IDS[@]}" -eq 0 || -z "${GPU_IDS[0]}" ]]; then
  GPU_IDS=("0")
fi

COMBO_COUNT="${#COMBINATIONS[@]}"
GPU_COUNT="${#GPU_IDS[@]}"

if [[ -n "${MMA_SIM_MAX_PROCS:-}" ]]; then
  MAX_PROCS="${MMA_SIM_MAX_PROCS}"
else
  MAX_PROCS="${GPU_COUNT}"
fi

if (( MAX_PROCS > COMBO_COUNT )); then
  MAX_PROCS="${COMBO_COUNT}"
fi

if (( MAX_PROCS < 1 )); then
  MAX_PROCS=1
fi

JOB_FILE="$(mktemp)"
trap 'rm -f "${JOB_FILE}"' EXIT

for idx in "${!COMBINATIONS[@]}"; do
  gpu="${GPU_IDS[$((idx % GPU_COUNT))]}"
  printf '%s %s\n' "${gpu}" "${COMBINATIONS[$idx]}" >> "${JOB_FILE}"
done

# --- 预备 ---
echo "=================================================="
echo "并行启动测试脚本: ${PYTHON_SCRIPT}"
echo "最大并行进程数: ${MAX_PROCS}"
echo "可见/指定 GPU: ${GPU_CSV}"
echo "每个组合测试用例数: ${MMA_SIM_TEST_CASES}"
echo "覆盖率: ${MMA_SIM_COVERAGE}"
echo "测试组合:"
cat -n "${JOB_FILE}"
echo "=================================================="
echo # 空行

# --- 计时开始 ---
start_time_seconds=$(date +%s)
start_time_human=$(date '+%Y-%m-%d %H:%M:%S')
echo "[计时开始] ${start_time_human}"
echo # 空行

# --- 执行并行任务 ---
echo "正在并行执行测试..."
if [[ "${MMA_SIM_COVERAGE}" == "1" ]]; then
  "${PYTHON_BIN}" -m coverage erase || true
  xargs -P "${MAX_PROCS}" -L 1 bash -c '
    set -euo pipefail
    gpu="$1"
    shift
    echo "[GPU ${gpu}] ${PYTHON_SCRIPT} $*"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m coverage run --parallel-mode --source=. "${PYTHON_SCRIPT}" "$@"
  ' bash < "${JOB_FILE}"
else
  xargs -P "${MAX_PROCS}" -L 1 bash -c '
    set -euo pipefail
    gpu="$1"
    shift
    echo "[GPU ${gpu}] ${PYTHON_SCRIPT} $*"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${PYTHON_SCRIPT}" "$@"
  ' bash < "${JOB_FILE}"
fi

# xargs 会等待所有它启动的进程结束后才退出
echo # 空行
echo "所有 Python 进程已执行完毕。"

if [[ "${MMA_SIM_COVERAGE}" == "1" ]]; then
  # --- 合并覆盖率数据 ---
  echo "--------------------------------------------------"
  echo "合并并行覆盖率数据..."
  "$PYTHON_BIN" -m coverage combine || echo "警告: 无法合并覆盖率数据。确保 'coverage' 已安装且存在 .coverage.* 文件。"
  echo "覆盖率数据已尝试合并到 .coverage 文件。"
  echo "--------------------------------------------------"
fi

# --- 计时结束 ---
end_time_seconds=$(date +%s)
end_time_human=$(date '+%Y-%m-%d %H:%M:%S')
echo "[计时结束] ${end_time_human}"
echo # 空行

# --- 计算并显示总耗时 ---
duration=$((end_time_seconds - start_time_seconds))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=================================================="
echo "所有测试已完成。"
echo "总耗时: ${duration} 秒 (即 ${minutes} 分 ${seconds} 秒)"
if [[ "${MMA_SIM_COVERAGE}" == "1" ]]; then
  echo "覆盖率数据已合并 (如果成功)。"
  echo "请在脚本结束后手动运行:"
  echo "  coverage report -m  (查看终端报告)"
  echo "  coverage html       (生成 HTML 报告到 htmlcov/ 目录)"
fi
echo "=================================================="

# 脚本正常退出
exit 0
