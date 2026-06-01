# Extending Pose with plugins

Pose exposes four extension points so contributors can ship new capabilities **without modifying the core library**. This document covers how the plugin system works, the two ways to register a plugin, and a complete worked example.

## Why a plugin system?

`pose/pose.py` is the project's stable golden core (see [`GOVERNANCE.md`](https://github.com/sarisabban/Pose/blob/main/GOVERNANCE.md)). Force fields, alternative file parsers, custom builders, and bespoke exporters live **outside** the core, registered into Pose at runtime.

A plugin is:

- An ordinary Python class or callable
- Living in your own package (`my_pose_extension`)
- Discovered by Pose via Python entry points or programmatic registration

## The four categories

| Group name | What it provides | Typical example |
|---|---|---|
| `pose.parsers` | additional file-format readers/writers | cryo-EM map import, in-house structure files |
| `pose.scorers` | scoring functions / force fields | AMBER ff19SB scorer, custom ML model |
| `pose.builders` | alternative builders | templated-loop builder, ML-driven backbone builder |
| `pose.exporters` | additional export formats | session files for visualization tools, custom JSON |

## Method 1 — entry points (recommended for installed packages)

In your plugin's `pyproject.toml`:

```toml
[project]
name = "pose-amber-scorer"
version = "0.1.0"
dependencies = ["pose"]

[project.entry-points."pose.scorers"]
amber_ff19sb = "pose_amber_scorer.scorer:AmberScorer"
```

When a user installs your package alongside Pose:

```bash
pip install pose pose-amber-scorer
```

`pose.plugins.ListScorers()` will include `"amber_ff19sb"` and `pose.plugins.GetScorer("amber_ff19sb")` returns your class. Discovery is automatic and lazy — entry points are resolved on the first call into the relevant group.

## Method 2 — programmatic registration

Use this for in-process plugins (e.g., a notebook session, a research script):

```python
from pose.plugins import RegisterScorer

class MyScorer:
    def __init__(self, **kwargs):
        ...
    def __call__(self, pose):
        return some_energy_value

RegisterScorer("my_scorer", MyScorer)
```

After registration:

```python
from pose.plugins import ListScorers, GetScorer

assert "my_scorer" in ListScorers()
ScorerCls = GetScorer("my_scorer")
score = ScorerCls()(p)
```

## API reference

```python
pose.plugins.Discover(group)
pose.plugins.Register(group, name, cls)
pose.plugins.Unregister(group, name)
pose.plugins.Get(group, name)
pose.plugins.List(group)

# Convenience wrappers per category:
pose.plugins.RegisterParser(name, cls)
pose.plugins.RegisterScorer(name, cls)
pose.plugins.RegisterBuilder(name, cls)
pose.plugins.RegisterExporter(name, cls)
pose.plugins.ListParsers()
pose.plugins.ListScorers()
pose.plugins.ListBuilders()
pose.plugins.ListExporters()
pose.plugins.GetParser(name)
pose.plugins.GetScorer(name)
pose.plugins.GetBuilder(name)
pose.plugins.GetExporter(name)
```

`Register` raises `ValueError` for an unknown group and `KeyError` if the name is already registered. `Get` raises `KeyError` with the list of known names when the name is missing.

## Plugin interface contracts

Pose does **not** force a base class — the plugin system is duck-typed. The conventions below are what core Pose code expects from each category. Plugin authors should follow them so users can swap implementations.

### Scorer

```python
class Scorer:
    def __init__(self, **kwargs):
        '''Optional configuration. Should not perform expensive I/O.'''

    def __call__(self, pose) -> float:
        '''Return the energy of the pose, in kcal/mol unless documented.'''
```

### Parser

```python
class Parser:
    extensions = ('.xyz',)  # the file extensions this parser claims

    def Parse(self, path: str, pose) -> None:
        '''Populate `pose.data` in place from the file at `path`.'''
```

### Builder

```python
class Builder:
    def Build(self, pose, **kwargs) -> None:
        '''Construct atoms into `pose.data` from inputs (sequence, etc.).'''
```

### Exporter

```python
class Exporter:
    extensions = ('.xyz',)

    def Export(self, pose, path: str) -> None:
        '''Write `pose` to `path` in this exporter's format.'''
```

These contracts may be tightened in future minor releases; new methods will be added with default implementations on a default base class (TBD).

## Worked example — a tiny scorer plugin

A complete external package that ships a scorer via entry points:

`pose_constant_scorer/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pose-constant-scorer"
version = "0.1.0"
dependencies = ["pose>=2.0"]

[project.entry-points."pose.scorers"]
constant = "pose_constant_scorer.scorer:ConstantScorer"
```

`pose_constant_scorer/scorer.py`:

```python
class ConstantScorer:
    '''
    A scorer that returns a fixed energy regardless of pose state.
    Useful as a regression sentinel and as a minimal plugin template.
    '''
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self, pose):
        return self.value
```

Once installed:

```python
import pose
from pose.plugins import GetScorer

S = GetScorer("constant")
print(S(value=42.0)(some_pose))  # 42.0
```

## Testing a plugin

Plugins should be tested in their own repository with their own `pytest` suite. Pose provides no special test scaffolding — your plugin is a normal Python package.

For an internal smoke test, the Pose test suite includes `tests/test_plugins.py` which exercises register/list/get/unregister round-trips and entry-point discovery idempotency. Adapt those patterns for your own.

## Listing plugins in the wild

We maintain a curated list of community plugins at TODO (link from `docs/community.md`). Add yours via PR once it's installable.
