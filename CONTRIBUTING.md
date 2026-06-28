# Contributing

Keep changes focused and validation-backed.

## Development

```bash
./scripts/setup_env.sh
source .venv/bin/activate
python3 -m compileall cmodel
./scripts/run_smoke.sh f16 f16 f16
```

## Pull Request Expectations

- Explain which dtype and MMA shape are affected.
- Include the exact validation command and case count.
- Keep generated outputs out of commits.
- Do not change the core numerical model and hardware comparison harness in the same patch unless the validation evidence requires both.
