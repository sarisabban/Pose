---
name: Integration recipe request
about: Request a "Pose + X" worked example
title: "[RECIPE] Pose + "
labels: recipe, help wanted
assignees: ''
---

## What integration?

Pose + _<tool name and version>_

Examples that already ship: OpenMM, RDKit, OpenFF, Biopython, PyMOL, AlphaFold.

## What workflow?

_Describe the end-to-end workflow you want a recipe for. What does the user start with? What do they end with?_

Example:
> "I want to build a peptide in Pose, parameterise it with AMBER ff19SB (loaded via Pose's `Port()`), minimise with OpenMM, and read the minimised coordinates back into Pose for analysis."

## Why is this important?

_Who would use this recipe? What real problem does it solve?_

## Constraints

- Target platform(s): _Linux only? Linux + macOS? GPU required?_
- Approximate compute budget: _seconds, minutes, hours? GPU?_
- Reproducibility requirements: _bit-exact, deterministic up to floating-point error, or stochastic OK?_

## Can you help?

- [ ] I can draft the recipe and send a PR
- [ ] I can help test a draft
- [ ] I am looking for a maintainer to write it

Recipes live in [`recipes/`](recipes/) and follow a standard structure (see existing recipes for the pattern).
