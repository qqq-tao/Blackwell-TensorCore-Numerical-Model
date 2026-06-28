# Release Checklist

This document is for private upload and public-release hygiene. It is not legal
advice.

## License Boundary

The root `LICENSE` permits non-commercial learning, teaching, academic
coursework, academic publication support, and non-commercial academic research.
Commercial use requires separate prior written permission.

Because commercial use is restricted, describe the project as
`source-available` or under a `non-commercial academic license`. Do not describe
it as OSI-approved open source.

## Public Wording

Use wording like:

```text
Source-available for non-commercial academic and educational use.
Commercial use requires prior written permission.
The technical target is bitwise reproduction of observed dense Tensor Core MMA
results on RTX 5090-class hardware, not tolerance-based approximation.
The repository retains directed/manual edge cases and uses a scalable random
hardware-comparison matrix for the current release-candidate evidence.
The model is intended as a reproducible study reference, not as an official
hardware specification or performance model.
Any discussion of Tensor Core internals is an inferred numerical-behavior
pipeline from observable input/output behavior, not a claim about private
hardware implementation.
```

Avoid:

```text
Open source
Free for commercial use
Apache/MIT/BSD-style permissive license
Official hardware model
```

## Pre-Publish Checks

Run locally:

```bash
git diff --check
python3 -m compileall cmodel
bash -n scripts/setup_env.sh scripts/run_smoke.sh cmodel/run_test_nproc.sh
```

Scan for machine-local paths and unique host aliases:

```bash
rg -n '(<local-user>|<internal-host>|<absolute-home-path>|<private-repo-path>)' .
```

Replace the placeholder terms with any local usernames, host aliases, or
project paths used during private development.

Run the RTX 5090 validation matrix or document why only a smaller smoke run was
performed. See `docs/RUNNING_VALIDATION.md`.

## Do Not Publish

Confirm generated or machine-local files are absent:

- `.venv/`
- `.deps/`
- `cmodel/result/`
- `cmodel/failed_cases/`
- `cmodel/htmlcov/`
- `validation_logs/`
- wheel caches
- cubins, PTX/SASS dumps, profiler reports
- local archives such as `*.tar.gz`

## IP Provenance

Before public release, record a private provenance note:

- date prior employment ended
- date this repository was created
- dates for major implementation steps
- source of each major design claim: public documentation, public examples,
  independently generated experiments, or general personal experience
- confirmation that former-employer source, tests, comments, generated
  artifacts, internal paths, hostnames, reports, screenshots, and private test
  vectors are not included

Do not publish claims or data derived from non-public former-employer
information. If public release is planned, obtain legal review for employment
agreement, confidentiality, invention-assignment, trade-secret, and
patent-publication risks.

## Contact Before Public Release

Before making the repository public, configure:

- maintainer contact
- security/private disclosure contact
- commercial licensing contact

`COMMERCIAL_USE.md` and `SECURITY.md` should point to those contacts once they
exist.
