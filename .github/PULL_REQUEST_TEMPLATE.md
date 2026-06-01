# Summary

<!-- A short, sharp summary of WHAT this PR does and WHY. -->

Fixes # <!-- issue number, if applicable -->

## Type of change

- [ ] Bug fix (`fix(...)`)
- [ ] New feature (`feat(...)`)
- [ ] Performance improvement (`perf(...)`)
- [ ] Refactor, no behaviour change (`refactor(...)`)
- [ ] Tests only (`test(...)`)
- [ ] Documentation (`docs(...)`)
- [ ] Chore — build, CI, deps, etc. (`chore(...)`)

## Areas touched

- [ ] `pose/tools.py`
- [ ] `pose/energy.py`
- [ ] `pose/pose.py` &nbsp; — **requires BDFL pre-approval** (see [`CONTRIBUTING.md`](../CONTRIBUTING.md#special-policy-posepy))
- [ ] `pose/database.json`
- [ ] `tests/`
- [ ] `recipes/`
- [ ] `benchmarks/`
- [ ] `docs/`
- [ ] `.github/` or CI

## Checklist

- [ ] `pytest` passes locally
- [ ] Coverage on changed code is ≥ 85%
- [ ] `ruff check .` is clean
- [ ] `mypy pose/tools.py pose/energy.py` is clean (if those files were touched)
- [ ] I added an entry to `CHANGELOG.md` under **Unreleased**
- [ ] Public-API changes are documented in `docs/`
- [ ] A regression test was added for the fix or feature
- [ ] No new dependency was added without discussion (NumPy-only is the default rule)
- [ ] If `pose/pose.py` was modified, the BDFL pre-approved this change

## Test plan

<!--
What did you do to validate this change? Be specific.
Examples:
- "Ran `pytest tests/test_rmsd.py -v` — all 14 tests pass."
- "Built on 1UBQ, ran the OpenMM minimization recipe, compared output positions."
- "Benchmark on 4HHB: before 12.3s, after 0.8s."
-->

## Notes for reviewers

<!-- Anything reviewers should pay extra attention to, design alternatives considered, follow-up work tracked elsewhere, etc. -->
