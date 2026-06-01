# Contributing to Pose

Thanks for your interest in contributing to Pose. This document covers everything you need to file an issue, send a pull request, or join the maintainer team.

## TL;DR

```bash
git clone https://github.com/sarisabban/Pose
cd Pose
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

If `pytest` is green on a fresh clone, you have a working development environment.

## What's welcome

We welcome contributions of:

- **Bug reports** — please use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
- **Bug fixes** — small fixes (≤ ~50 lines diff) with a regression test can usually be merged by any maintainer.
- **New tests** — coverage of edge cases in `pose/tools.py` and `pose/energy.py` is especially valuable.
- **Integration recipes** — runnable scripts in `recipes/` that show "Pose + X" workflows (Pose + OpenMM, Pose + RDKit, Pose + OpenFF, etc.). See [`recipes/`](recipes/) for the pattern.
- **Documentation** — improvements to docstrings, tutorials, theory pages, comparison tables.
- **Plugins** — Pose exposes entry points for `pose.scorers`, `pose.parsers`, `pose.builders`, `pose.exporters` (see [`docs/extending.md`](docs/extending.md)). Plugins live in your own package — Pose just discovers them.
- **Benchmarks** — additions to the [`benchmarks/`](benchmarks/) `asv` suite.

## What requires a discussion first

Open an issue or a draft [RFC](.github/RFC_TEMPLATE.md) **before writing code** for:

- Any change to a public API (functions, classes, kwargs, return shapes, defaults)
- New top-level modules
- Anything that requires modifying `pose/pose.py`
- Anything that adds a runtime dependency beyond NumPy

For `pose/pose.py` specifically, see the [Special policy](#special-policy-posepy) below.

## Special policy: `pose/pose.py`

`pose/pose.py` is the project's **stable golden core**. The BDFL ([Sari Sabban](https://github.com/sarisabban)) has declared this file untouchable to ensure long-term API stability. Concretely:

> Contributions to `pose.py` are accepted only through co-design with the BDFL — open a [Discussion](https://github.com/sarisabban/Pose/discussions) or an RFC issue **before writing any code**. A PR that modifies `pose.py` without prior discussion will be closed regardless of merit.

All other files welcome PRs through the normal flow. See [`GOVERNANCE.md`](GOVERNANCE.md) for the full reasoning.

## Development setup

### Prerequisites

- Python ≥ 3.10 (development host pins to 3.12; CI runs 3.10/3.11/3.12/3.13)
- Git
- Optional but recommended: `uv` for fast dependency installs

### One-time setup

```bash
git clone https://github.com/sarisabban/Pose
cd Pose
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,docs]"
pre-commit install
```

`pre-commit install` registers the git hook that runs `ruff`, `mypy`, and basic file-format checks on every commit.

### Running tests

```bash
pytest                      # full suite
pytest tests/test_tools.py  # one file
pytest -k "rmsd"            # by keyword
pytest --cov=pose           # with coverage report
```

The default config (`pyproject.toml`) treats Python warnings as errors. If your change emits a new warning, fix it or annotate it explicitly.

### Running benchmarks

```bash
asv run                  # compare HEAD to main
asv run --quick          # one revision, one sample
asv publish && asv preview
```

### Building the docs

```bash
sphinx-build docs/ docs/_build/html
open docs/_build/html/index.html
```

## Style guide

These rules come from [`CLAUDE.md`](CLAUDE.md) at the repo root and are enforced (where automatable) by `ruff` and `pre-commit`:

- **Tabs for indentation**, not spaces.
- **80-character line limit.**
- **Minimal nesting** — flatten nested `if` and `for` blocks where it doesn't sacrifice clarity.
- **No top-level helper functions** — keep everything within the main function. Sub-functions inside the main function are allowed.
- **No underscores in function names** (consequence of the project's CamelCase convention).
- **No `import` statements inside functions** — imports go at the top of the module.
- **Docstrings**: the project uses a specific format documented in [`CLAUDE.md`](CLAUDE.md). New public functions must include a docstring in this format.

`pose/pose.py` predates `ruff` and is excluded from lint per the special policy above.

## Pull request checklist

Before opening a PR:

- [ ] Tests pass locally (`pytest`)
- [ ] Coverage of changed code is ≥ 85% (`pytest --cov=pose`)
- [ ] `ruff check .` is clean
- [ ] `mypy pose/tools.py pose/energy.py` is clean (if you touched those files)
- [ ] If you touched `pose/pose.py`, you have BDFL pre-approval (see [Special policy](#special-policy-posepy))
- [ ] You added an entry to `CHANGELOG.md` under the **Unreleased** section
- [ ] If you changed public API, you added or updated docs in `docs/`
- [ ] Your commit messages follow the convention `type(scope): short summary` (e.g., `fix(tools): correct RMSD on disordered residues`)

Reviewers may ask for changes. Don't take it personally; it's the norm.

## Commit message convention

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(<scope>):` — new functionality
- `fix(<scope>):` — bug fix
- `perf(<scope>):` — performance improvement, no behaviour change
- `refactor(<scope>):` — internal change, no behaviour change
- `test(<scope>):` — test-only change
- `docs(<scope>):` — documentation change
- `chore(<scope>):` — build, CI, dependencies, license, etc.

Common scopes: `pose`, `tools`, `energy`, `tests`, `bench`, `docs`, `ci`, `build`, `license`.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md). **Do not** open public issues for security vulnerabilities.

## Code of Conduct

Participation in this project is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Getting credit

- Every merged PR adds you to the auto-generated `CONTRIBUTORS.md`.
- Each tagged release on GitHub mints a Zenodo DOI that lists all contributors as co-authors. See `.zenodo.json`.
- Sustained contributors are considered for maintainer promotion — see [`MAINTAINERS.md`](MAINTAINERS.md).

Thank you for helping make Pose better.
