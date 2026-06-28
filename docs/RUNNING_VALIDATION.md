# Running Validation

This document describes the project layout, setup, and reproducible validation
commands.

## Repository Layout

```text
cmodel/
  tensor_core_mma_cmodel.py        # Core numerical cmodel
  nv_result_cmp_cmodel.py          # Hardware-vs-cmodel comparison entry point
  nvidia_dense_mma/dense_mma_sync.py
  random_case_generator.py
  benchmark_values/                # Special values and CUDA conversion helper
  configs/mma_shape_configuration.json
  test_cases/                      # Retained directed/manual YAML cases
  legacy/                          # Debug, scratch, and historical artifacts
scripts/
  setup_env.sh
  run_smoke.sh
docs/
  TECHNICAL_OVERVIEW.md
  RUNNING_VALIDATION.md
  RELEASE_CHECKLIST.md
```

The original `cmodel/` layout is intentionally preserved because validation
scripts import sibling modules directly.

## Requirements

- Linux with Python 3.10+
- NVIDIA driver with RTX 5090 / SM120 support
- CUDA toolkit with `nvcc`
- Python packages from `cmodel/requirements.txt`

The hardware comparison path requires PyCUDA and compiles inline PTX kernels for
`sm_120a`.

## Setup

```bash
./scripts/setup_env.sh
source .venv/bin/activate
./scripts/run_smoke.sh f16 f16 f16
```

On systems without `python3-venv`, install dependencies into local `.deps/`:

```bash
USE_SYSTEM_PYTHON=1 PYTHON_BIN=/usr/bin/python3 ./scripts/setup_env.sh
PYTHON_BIN=/usr/bin/python3 ./scripts/run_smoke.sh f16 f16 f16
```

If `nvcc` is not on `PATH`, wrapper scripts use `/usr/local/cuda/bin/nvcc` when
it exists. Otherwise set `NVCC=/path/to/nvcc`.

## Local Checks

```bash
git diff --check
python3 -m compileall cmodel
bash -n scripts/setup_env.sh scripts/run_smoke.sh cmodel/run_test_nproc.sh
```

These checks confirm syntax and whitespace hygiene only. They do not prove
numerical alignment with hardware.

## Hardware Smoke

The smoke script defaults to 16 random cases:

```bash
./scripts/run_smoke.sh f16 f16 f16
```

Increase the count when needed:

```bash
MMA_SIM_TEST_CASES=1024 ./scripts/run_smoke.sh f16 f16 f16
```

## Release-Candidate Matrix

The release-candidate matrix defaults to `65536` random cases per dense dtype
combination and binds workers across visible GPUs:

```bash
cd cmodel
PYTHON_BIN=../.venv/bin/python ./run_test_nproc.sh
```

Useful overrides:

```bash
# Quick matrix while iterating.
MMA_SIM_TEST_CASES=1024 PYTHON_BIN=../.venv/bin/python ./run_test_nproc.sh

# Explicit GPU assignment on an 8-GPU RTX 5090 host.
MMA_SIM_GPU_LIST=0,1,2,3,4,5,6,7 PYTHON_BIN=../.venv/bin/python ./run_test_nproc.sh

# Optional coverage run. This is slower and is not needed for numeric validation.
MMA_SIM_COVERAGE=1 MMA_SIM_TEST_CASES=1024 PYTHON_BIN=../.venv/bin/python ./run_test_nproc.sh
```

The model and generated-kernel path can be run with other configured dense
cases. The release-candidate evidence reports one selected dense shape per dtype
combination:

```text
f16.f16.f32: m16n8k16
e4m3.e4m3.f32: m16n8k32
e5m2.e5m2.f32: m16n8k32
e4m3.e5m2.f32: m16n8k32
e5m2.e4m3.f32: m16n8k32
tf32.tf32.f32: m16n8k8
f16.f16.f16: m16n8k16
bf16.bf16.f32: m16n8k16
```

## Current RTX 5090 Evidence

Release-candidate revalidation on June 28, 2026:

```text
Host: RTX 5090 validation server
GPU: 8x NVIDIA GeForce RTX 5090, driver 570.124.06
Setup: USE_SYSTEM_PYTHON=1 PYTHON_BIN=/usr/bin/python3 NVCC=/usr/local/cuda/bin/nvcc ./scripts/setup_env.sh
Command: PYCUDA_CACHE_DISABLE=1 MMA_SIM_TEST_CASES=65536 MMA_SIM_GPU_LIST=0,1,2,3,4,5,6,7 PYTHON_BIN=/usr/bin/python3 ./run_test_nproc.sh
Result: 8 dense dtype combinations, selected dense shape per combination, 65536 cases each, 0 errors in every result file
Total matrix wall time: 131 seconds
```

Parsed summaries:

```text
bf16.bf16.f32: 65536 cases, 0 errors, elapsed 105.296s
e4m3.e4m3.f32: 65536 cases, 0 errors, elapsed 126.997s
e4m3.e5m2.f32: 65536 cases, 0 errors, elapsed 127.692s
e5m2.e4m3.f32: 65536 cases, 0 errors, elapsed 123.471s
e5m2.e5m2.f32: 65536 cases, 0 errors, elapsed 122.181s
f16.f16.f16: 65536 cases, 0 errors, elapsed 101.734s
f16.f16.f32: 65536 cases, 0 errors, elapsed 102.056s
tf32.tf32.f32: 65536 cases, 0 errors, elapsed 91.417s
```

## Directed Cases

`cmodel/test_cases/` retains directed/manual YAML cases that document boundary
patterns used during model development. They are validation assets, but they are
not currently wired into the release-candidate matrix command.

## RTX 5090 Server Deployment

Use any Linux server with RTX 5090 / SM120 support. Replace
`<rtx5090-server>` and `<remote-repo>` with your target host and checkout path:

```bash
rsync -a --delete \
  --exclude .git/ \
  --exclude .venv/ \
  --exclude .deps/ \
  --exclude __pycache__/ \
  --exclude result/ \
  --exclude failed_cases/ \
  --exclude htmlcov/ \
  --exclude '*.so' \
  --exclude validation_logs/ \
  ./ <rtx5090-server>:<remote-repo>/
```

On an 8-GPU RTX 5090 server:

```bash
cd <remote-repo>
USE_SYSTEM_PYTHON=1 PYTHON_BIN=/usr/bin/python3 NVCC=/usr/local/cuda/bin/nvcc ./scripts/setup_env.sh
cd cmodel
PYCUDA_CACHE_DISABLE=1 \
MMA_SIM_TEST_CASES=65536 \
MMA_SIM_GPU_LIST=0,1,2,3,4,5,6,7 \
PYTHON_BIN=/usr/bin/python3 \
./run_test_nproc.sh
```

## Generated Outputs

Generated files are intentionally excluded from git:

- `.venv/`
- `.deps/`
- `cmodel/result/`
- `cmodel/failed_cases/`
- `cmodel/htmlcov/`
- `validation_logs/`
- compiled extensions such as `fpemu*.so`
- cubins, PTX/SASS dumps, profiler reports, and coverage files
