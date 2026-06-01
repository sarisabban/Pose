---
name: Feature request
about: Suggest a new feature or capability
title: "[FEATURE] "
labels: enhancement
assignees: ''
---

## Problem

_What problem does this feature solve? Who is affected? Is there a workaround today?_

## Proposed solution

_What you'd like Pose to do. Be specific about the API if you have one in mind._

```python
# Sketch of how the new feature would be used
import pose
p = pose.Pose()
result = p.new_method(...)
```

## Alternatives considered

_Other approaches you thought about, and why this one is preferred._

## Scope check

Pose has a bounded scope (see [`README.md` → What Pose is NOT](https://github.com/sarisabban/Pose#what-pose-is-not)). Does this feature fit?

- [ ] This is structure manipulation / measurement / I/O (in scope)
- [ ] This is a plugin-shaped extension (force field, scorer, parser) — could ship as a plugin instead
- [ ] This is something Pose does NOT intend to be (MD engine, docking, prediction, visualisation, energy minimisation)
- [ ] Unsure — looking for guidance

## Anything else?

- [ ] I would be willing to send a PR for this (with maintainer guidance)
- [ ] This affects public API — I'm willing to write an [RFC](.github/RFC_TEMPLATE.md) first
- [ ] This touches `pose/pose.py` (requires BDFL co-design per [CONTRIBUTING.md](CONTRIBUTING.md))
