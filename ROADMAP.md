# Roadmap

This document records what the Pose maintainers intend to ship. It is not a contract — priorities change as the community grows and as feedback comes in. The roadmap is reviewed at each minor release.

Time horizons are intentionally fuzzy. Concrete deliverables that are ready to ship sooner become entries in [`CHANGELOG.md`](CHANGELOG.md); items here are larger initiatives.

## Near-term (3 months)

- Land the OSS-readiness work currently in progress (this plan): test suite, benchmarks, plugin architecture, docs site, CI matrix, JOSS submission.
- First batch of integration recipes: Pose + OpenMM, Pose + RDKit, Pose + OpenFF, Pose + Biopython, Pose + PyMOL.
- Backlog of well-scoped `good first issue` and `help wanted` items (≥ 50) for new contributors.
- Stable plugin architecture with at least one community-contributed plugin example.
- conda-forge feedstock.

## Mid-term (12 months)

- Comprehensive D-amino-acid rotamer library (no equivalent exists today; ties into flagship paper).
- More force-field interop: `tools.Port()` covers AMBER family fully (ff14SB, ff19SB, ff99SB-ILDN), CHARMM36 variants (m, mff14), Rosetta REF15, AutoDock Vina; OpenFF parameter packages tracked as they release.
- Optional differentiable-geometry backend (JAX) for `pose.tools` — opt-in via a separate dependency group; pure-NumPy remains the default.
- Sphinx tutorials covering: building peptides, mutation scans, simulated annealing, geometric features for ML pipelines.
- First non-BDFL maintainer promoted (per [`GOVERNANCE.md`](GOVERNANCE.md) criteria).
- Flagship application paper accepted to a peer-reviewed journal.

## Long-term (3 years)

- Steering-committee governance (3 maintainers) replaces BDFL model.
- Stable, community-maintained registry of plugins for niche scoring/parsing/exporting needs.
- A formal partnership or merge with at least one adjacent project where it makes scientific sense (candidates: OpenFF for force-field parameters, NumFOCUS for fiscal sponsorship).
- Either (a) Pose develops a novel internal force field with a defining paper, OR (b) Pose becomes the de facto pure-NumPy structural-manipulation substrate for the ecosystem with no need to re-implement force fields.
- Reproducibility audit and a published bit-reproducibility study of cross-platform structural geometry, with Pose as the deterministic baseline.

## Things Pose deliberately will not pursue

(In addition to [What Pose is NOT](README.md#what-pose-is-not) in the README:)

- A GUI. Pose is a library; visualisation belongs to PyMOL/ChimeraX/Mol*.
- A web service. Pose runs locally.
- A neural-network model registry. Pose provides the features that downstream ML packages consume.
- Vendoring third-party force-field parameters under non-permissive licences. The `tools.Port()` architecture lets users bring their own.

## How to influence the roadmap

- Open an issue tagged `roadmap` with the case for a new initiative.
- For substantive proposals, open an [RFC](.github/RFC_TEMPLATE.md).
- Pull requests that implement roadmap items are welcome — please open an issue first to coordinate.
