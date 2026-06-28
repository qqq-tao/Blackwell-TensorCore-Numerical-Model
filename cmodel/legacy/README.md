# Legacy and Scratch Files

This directory keeps pre-cleanup scripts that are not part of the main validation path.

Use `../nv_result_cmp_cmodel.py` for current hardware-vs-cmodel validation.

Contents:

- `*_nv_result_cmp_cmodel.py`: dtype-specific validation entry points superseded by the generic runner.
- `general_cmodel_precison_check.py`: older debug runner that imports the debug cmodel.
- `*_debug.py`: earlier debug variants of the cmodel and random generator.
- `experiments/`: disassembly and standalone CUDA experiments that are not part of the release validation path.
- `regression_cases/`: timestamped failed-case captures retained for historical debugging only. They are not curated release fixtures.
- `scratch/`: local import/rounding experiments retained only for reference.
