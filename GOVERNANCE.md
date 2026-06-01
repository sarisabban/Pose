# Governance

This document describes how decisions are made in the Pose project.

## Model — BDFL with a path to a steering committee

Pose currently operates under a **Benevolent Dictator For Life (BDFL)** model. The BDFL is the founder and primary author, **Sari Sabban**.

The BDFL model is appropriate for Pose's current size (single primary maintainer, small contributor pool). The intent is to migrate to a **3-person Steering Committee** once at least three sustained contributors with merge rights exist. The transition will be announced in a release-note and codified by an amendment to this document.

## Roles

- **BDFL** — has final authority on all technical and project-direction decisions. Currently: Sari Sabban (`@sarisabban`). The BDFL is also the sole maintainer of `pose/pose.py` (see "Special files" below).
- **Maintainers** — listed in [`MAINTAINERS.md`](MAINTAINERS.md). Have commit/merge rights on the repository.
- **Contributors** — anyone who has had a pull request merged. Listed in `CONTRIBUTORS.md` (auto-generated).

## Decision process

### Non-controversial changes

The following may be reviewed and merged by any maintainer (not just the BDFL):

- Documentation fixes (typos, clarifications, formatting)
- Test additions that do not change public API
- Bug fixes with a clear regression test and ≤ ~50 lines diff
- Dependency-version bumps that pass CI without changes
- New integration recipes in `recipes/` that follow existing patterns

### API-affecting changes

Any pull request that **adds, removes, or modifies a public API** (functions, classes, kwargs, file formats, defaults, return types, or anything documented in `docs/api/`) requires:

1. An RFC issue using the [RFC template](.github/RFC_TEMPLATE.md), opened before code is written, *or*
2. BDFL approval on the PR itself for changes small enough to skip the RFC step (judgement call).

### Special files — `pose/pose.py`

`pose/pose.py` is the project's **golden core**. The BDFL has declared this file untouchable to ensure long-term API stability. Concretely:

- Code changes to `pose.py` are accepted **only** with BDFL authorship or BDFL-explicit co-authorship.
- Bug reports, RFCs, and feature requests against `pose.py` are welcome. They are tracked in `POSE_FINDINGS.md` and acted on by the BDFL personally on the BDFL's schedule.
- The exclusion does **not** apply to `pose/tools.py`, `pose/energy.py`, tests, docs, recipes, plugins, benchmarks, governance, or CI.

If a community contribution genuinely requires a change to `pose.py`, the workflow is: open an RFC → discuss with BDFL → the BDFL implements the change (potentially based on the contributor's prototype patch). The contributor is credited.

## Conflict resolution

1. **Design disagreements** should be raised in GitHub Discussions or as an RFC. The BDFL will respond with a decision; the discussion is then closed.
2. **Code-of-conduct issues** are reported privately to the maintainer email listed in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The BDFL handles these directly; if the BDFL is the subject of the complaint, the matter is escalated to NumFOCUS (if/when Pose joins) or to an independent mediator agreed upon by the complainant.
3. **Technical disputes between maintainers** are resolved by the BDFL.

## Amendments

This governance document can be amended by the BDFL. Amendments are announced in the release that includes them and are summarised in `CHANGELOG.md`.

## See also

- [`MAINTAINERS.md`](MAINTAINERS.md) — current maintainer list and how to join.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — community standards.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to contribute.
- [`VERSIONING.md`](VERSIONING.md) — versioning and deprecation policy.
