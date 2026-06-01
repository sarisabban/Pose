# Versioning policy

Pose follows [Semantic Versioning 2.0.0](https://semver.org/) strictly.

## Version format

`MAJOR.MINOR.PATCH`

- **MAJOR** is bumped for backwards-incompatible changes to the public API, removal of deprecated features, or changes to default behaviour that could surprise existing users.
- **MINOR** is bumped for backwards-compatible new features.
- **PATCH** is bumped for backwards-compatible bug fixes only.

Pre-release suffixes (`-alpha.N`, `-beta.N`, `-rc.N`) follow SemVer's rules.

## Public API surface

The **public API** of Pose is anything that meets all of the following:

1. Documented in `docs/api/` (auto-generated from public docstrings).
2. Not prefixed with a leading underscore.
3. Imported by users from `pose` (e.g., `pose.Pose`, `pose.Molecule`, `pose.tools.RMSD`).

Everything else — module-internal helpers, undocumented attributes, anything in a `_private` submodule — is **not** part of the public API and may change at any time without a SemVer bump.

The `pose/pose.py` file is the load-bearing definition of the public API surface. Changes to it require BDFL authorship (see [`GOVERNANCE.md`](GOVERNANCE.md)).

## Deprecation policy

When we need to remove or rename a public API, we follow a graceful deprecation path:

1. In version `X.Y.0`, the old API continues to work but emits a `DeprecationWarning` pointing to the replacement.
2. The deprecation is announced in `CHANGELOG.md` and in the docstring of the deprecated API.
3. The old API is removed no earlier than **two minor versions later** (so deprecated in `2.3.0` → removed at earliest in `2.5.0`).
4. If the API has been deprecated for ≥ 12 months, MAJOR is bumped on the removal.

For experimental features (explicitly marked in the docstring as experimental), the deprecation window may be shorter, but always at least one minor version with a warning.

## Stability tiers

Public APIs carry one of three stability markers in their docstring:

- **Stable** — covered by the deprecation policy above.
- **Experimental** — may change between minor versions with at least one minor version's notice.
- **Internal** — anything prefixed with `_` or in a `_private` submodule. No stability guarantee.

If a docstring does not specify a tier, assume **Stable**.

## `pose.__version__`

The runtime value of `pose.__version__` always matches the `version` field in `pyproject.toml`. Tagged releases use `vMAJOR.MINOR.PATCH` (e.g., `v2.0.0`). Development versions use `MAJOR.MINOR.PATCH.devN` per [PEP 440](https://peps.python.org/pep-0440/).

## Initial release

The current `1.0.0` carries the project's existing history. The first release after this OSS-readiness plan executes is expected to bump to **`2.0.0`** — major version because:

- License changed from GPL-2.0 to Apache-2.0
- Package metadata moved from `setup.py` to `pyproject.toml`
- Plugin entry points added
- Test suite restructured under `tests/`

The intent is to use the `2.x` line as the stable home and accumulate minor releases as features land.
