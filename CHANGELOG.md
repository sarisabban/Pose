# Changelog

All notable changes to Pose will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(see [`VERSIONING.md`](VERSIONING.md)).

## [Unreleased]

### Added

- Apache-2.0 `NOTICE` file with attribution.
- PEP 621 `pyproject.toml` with `dev`, `docs`, and `recipes` optional dependency groups.
- `.python-version` pinning the dev host to Python 3.12 (CI matrix still covers 3.10–3.13).
- `GOVERNANCE.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `MAINTAINERS.md`.
- `CONTRIBUTING.md` with explicit policy that `pose/pose.py` requires BDFL pre-approval.
- `SECURITY.md` with vulnerability-reporting process.
- `VERSIONING.md` documenting SemVer policy, deprecation rules, and stability tiers.
- `ROADMAP.md` with 3-month / 12-month / 3-year horizons.
- GitHub issue templates: bug report, feature request, integration recipe request, scientific question.
- GitHub pull-request template with mandatory checklist.
- GitHub RFC template for design discussions.
- `.github/FUNDING.yml` placeholder.
- `.pre-commit-config.yaml` with ruff, ruff-format, mypy, and standard whitespace hooks. `pose/pose.py` excluded.
- Repo scaffolding for `tests/`, `benchmarks/`, `docs/`, `examples/`, `recipes/`, `manuscripts/`.

### Changed

- **Licence: GPL-2.0 → Apache-2.0.** This is a major change. Adopters who relied on GPL-2.0 should pin to the last pre-relicense tag (TBD). The motivation is the Apache-2.0 patent grant and conformance with the scientific Python ecosystem norm (TensorFlow, PyTorch, JAX, AlphaFold).
- `setup.py` license metadata updated to `Apache-2.0`; the file remains for backwards-compatible installs but `pyproject.toml` is now the source of truth.
- `README.md` license badge swapped to Apache-2.0; License section rewritten with `NOTICE` and SPDX reference.
- `CITATION.cff` gained a `license: Apache-2.0` field.
- `.gitignore` expanded for venvs, caches, build artifacts, docs builds, asv runs, and IDE noise.

### Notes

- `pose/pose.py` does **not** carry an `SPDX-License-Identifier` header. Per the project's read-only-core constraint (`PLAN.txt` §2 C1), no edits to `pose.py` are made during the OSS-readiness plan execution. The SPDX header will be added by the maintainer when the file is next opened intentionally.
- `pose/__init__.py`, `pose/tools.py`, and `pose/energy.py` all carry `# SPDX-License-Identifier: Apache-2.0` at the top.

---

## [1.0.0] — 2023-06-01

Initial public release. See git history for details prior to this CHANGELOG.

[Unreleased]: https://github.com/sarisabban/Pose/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/sarisabban/Pose/releases/tag/v1.0.0
