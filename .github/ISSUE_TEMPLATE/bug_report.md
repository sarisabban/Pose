---
name: Bug report
about: Report something that is broken, behaves incorrectly, or crashes
title: "[BUG] "
labels: bug
assignees: ''
---

## Summary

_A one-sentence summary of what is broken._

## Minimal reproduction

```python
# Smallest possible code that reproduces the bug
import pose
p = pose.Pose()
# ...
```

## Expected behaviour

_What you expected to happen._

## Actual behaviour

_What actually happened. Include the full traceback if there is one._

```
<traceback or output here>
```

## Environment

- Pose version: _output of `python -c "import pose; print(pose.__version__)"`_
- Python version: _output of `python --version`_
- NumPy version: _output of `python -c "import numpy; print(numpy.__version__)"`_
- OS: _e.g. Ubuntu 24.04, macOS 14.5, Windows 11_
- Install method: _`pip install pose`, `pip install -e .`, `pip install git+…`_

## Input file (if applicable)

If the bug depends on a specific input file (PDB, mmCIF, SDF, …):

- Attach the file (or a minimal redaction) if you can share it
- Otherwise, paste a short excerpt showing the relevant portion

## Additional context

_Anything else that might help diagnose the issue. Linked issues, related PRs, screenshots, etc._

## Have you checked?

- [ ] Latest version of Pose (`pip install -U pose`)
- [ ] No similar issue already exists (search [issues](https://github.com/sarisabban/Pose/issues))
- [ ] The bug is in Pose, not in a wrapper/recipe/downstream tool
