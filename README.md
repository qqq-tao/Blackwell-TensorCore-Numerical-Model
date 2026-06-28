# Blackwell TensorCore Numerical Model

**Source-available for non-commercial learning, teaching, and academic research. Commercial use is prohibited without prior written permission.**

Blackwell TensorCore Numerical Model is a Python reference model for dense Tensor Core MMA numerical behavior on NVIDIA Blackwell / RTX 5090-class hardware. The repository contains:

- a bit-level Tensor Core MMA cmodel in `cmodel/tensor_core_mma_cmodel.py`
- PyCUDA-based hardware comparison tests in `cmodel/nv_result_cmp_cmodel.py`
- generated inline-PTX MMA kernels in `cmodel/nvidia_dense_mma/dense_mma_sync.py`
- test-case generators and CUDA conversion helpers under `cmodel/benchmark_values/`

The code is kept close to the original cmodel layout so existing validation commands continue to work.

## Repository Layout

```text
cmodel/
  tensor_core_mma_cmodel.py        # Core numerical cmodel
  nv_result_cmp_cmodel.py          # Compare cmodel output with RTX 5090 MMA output
  nvidia_dense_mma/dense_mma_sync.py
  benchmark_values/                # Special-value generators and fp conversion extension
  configs/mma_shape_configuration.json
  test_cases/                      # Retained YAML directed/manual cases
  run_test_nproc.sh                # Parallel validation matrix
scripts/
  setup_env.sh                     # Create venv, install deps, build fpemu
  run_smoke.sh                     # Short hardware smoke test
docs/
  TECHNICAL_OVERVIEW.md             # Numerical characterization and model
  RUNNING_VALIDATION.md             # Setup, deployment, and validation evidence
  RELEASE_CHECKLIST.md              # Source-available release hygiene
COMMERCIAL_USE.md                  # Commercial-use boundary and authorization notice
LICENSE                            # Non-commercial academic source license
```

Generated files such as `.venv/`, `.deps/`, `result/`, `failed_cases/`, coverage HTML, PyCUDA cache, cubins, and profile reports are intentionally excluded from git.

## Requirements

- Linux with Python 3.10+
- NVIDIA driver with RTX 5090 / SM120 support
- CUDA toolkit with `nvcc`
- Python packages from `cmodel/requirements.txt`

The hardware comparison path requires PyCUDA and compiles inline PTX kernels for `sm_120a`.

## Quick Start

```bash
./scripts/setup_env.sh
source .venv/bin/activate
./scripts/run_smoke.sh f16 f16 f16
```

On systems without `python3-venv`, install dependencies into local `.deps/` instead:

```bash
USE_SYSTEM_PYTHON=1 PYTHON_BIN=/usr/bin/python3 ./scripts/setup_env.sh
PYTHON_BIN=/usr/bin/python3 ./scripts/run_smoke.sh f16 f16 f16
```

By default the smoke script runs 16 random cases. Increase the count with:

```bash
MMA_SIM_TEST_CASES=1024 ./scripts/run_smoke.sh f16 f16 f16
```

Run the release-scale validation matrix. The matrix script defaults to 65536
random cases per combination and binds combinations across visible GPUs:

```bash
cd cmodel
VIRTUAL_ENV=../.venv PYTHON_BIN=../.venv/bin/python ./run_test_nproc.sh
```

For a quicker matrix pass, override the case count:

```bash
MMA_SIM_TEST_CASES=1024 VIRTUAL_ENV=../.venv PYTHON_BIN=../.venv/bin/python ./run_test_nproc.sh
```

## Main Validation Command

From `cmodel/`:

```bash
MMA_SIM_TEST_CASES=1024 ../.venv/bin/python nv_result_cmp_cmodel.py f16 f16 f16
```

Supported dense combinations and shapes are described in `cmodel/configs/mma_shape_configuration.json`.
The model and kernel generator can be run with other configured cases, but the
current release-candidate matrix reports one selected dense shape per dtype
combination.

## Findings and Significance

The technically difficult part is exact numerical reconstruction, not API
wrapping. The cmodel has to reproduce operand decoding, special-value handling,
accumulation, rounding, carry/guard/sticky behavior, and output conversion
closely enough that every observed result bit matches RTX 5090-class dense MMA
hardware for the validated type combinations.

Bitwise alignment matters because it makes the model useful as a regression
oracle for validated instruction paths. Instead of only checking whether a
result is within a tolerance, tests can detect one-bit differences in rounding
and edge-case behavior.

The verification case design is part of the contribution. Directed/manual YAML
cases are retained in `cmodel/test_cases/`, while the current release validation
path is the scalable random hardware-comparison matrix. The repository includes
the RTX 5090 hardware-comparison path needed to rerun that matrix on compatible
NVIDIA GPUs.

This release is intentionally scoped to the dense MMA paths listed in the
validation matrix. Block-scale support is a future release plan and should not
be treated as part of the current validated surface.

See `docs/TECHNICAL_OVERVIEW.md` for the numerical characterization method and
the inferred behavioral pipeline.

## Validation Status

This cleanup preserved the original cmodel source layout and removed generated artifacts from the repository. The current RTX 5090 server release-candidate revalidation passed 8 dense combinations with 65536 cases each and 0 errors per combination. The model, generated MMA kernels, and verification harness are included so the validation can be rerun with larger case counts on compatible NVIDIA GPUs. See `docs/RUNNING_VALIDATION.md` for exact commands and boundaries.

## License

This project is released under the `Blackwell TensorCore Numerical Model Non-Commercial Academic Source License`. It may be used for non-commercial learning, teaching, and academic research.

Commercial use is not permitted under this license. Commercial use, including internal research and development, benchmarking, validation, product development, chip design flow integration, toolchain integration, paid services, or production use by or for a commercial entity, requires separate prior written permission. Unauthorized commercial use is outside the scope of the license, and all rights and remedies are reserved.

Because commercial use is restricted, this is a source-available license rather than an OSI-approved open-source license.

See `COMMERCIAL_USE.md` for the commercial-use boundary.
