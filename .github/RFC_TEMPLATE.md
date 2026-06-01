# RFC: <Title>

<!--
Open an issue using this template for any change that:
- Adds, removes, or modifies a public API in Pose
- Adds a new module or extension point
- Touches `pose/pose.py` (requires BDFL pre-approval — see GOVERNANCE.md)
- Adds a new runtime dependency

Small, obvious changes do not need an RFC. When in doubt, ask in Discussions first.
-->

**Status:** Draft <!-- Draft → Accepted → Implemented | Rejected | Withdrawn -->
**Author(s):** @<your-handle>
**Created:** YYYY-MM-DD
**Last updated:** YYYY-MM-DD
**Tracking issue:** (this issue)
**Tracking PR(s):** (filled in once code lands)

## Summary

_One paragraph elevator-pitch._

## Motivation

_What problem does this solve? Who is affected? What are the consequences of doing nothing?_

## Detailed design

_The bulk of the RFC. Be specific. Include API sketches, file layouts, function signatures, type annotations, edge cases. A reader should be able to **start implementing** from this section._

### Public API surface

```python
# Sketch the proposed public API.
import pose
# ...
```

### Internal mechanics

_How does it work under the hood? Where in the code does it live? What does it touch?_

### Failure modes

_What can go wrong? How does it fail? How is failure surfaced to the user?_

### Examples

_Concrete worked examples showing the feature in use._

## Alternatives considered

_What other designs were on the table? Why this one?_

## Migration / deprecation

_If this changes existing behaviour: what is the deprecation path? When can the old behaviour be removed? See [`VERSIONING.md`](../VERSIONING.md)._

## Backwards-compatibility

_Is this a breaking change? If so, what's the bump (MAJOR/MINOR)?_

## Performance impact

_Expected effect on memory and runtime, especially for the benchmark structures in `benchmarks/`._

## Test plan

_What tests will be added? Which existing tests cover regression risk?_

## Documentation plan

_What docs change? Tutorial added? API reference auto-updated?_

## Open questions

_Things the author isn't sure about and wants feedback on._

## Adoption

_Once accepted, what is the next step? Is there an owner? Is mentoring needed?_

---

## Review notes

<!-- Maintainers and community add comments below as the RFC is discussed.
Once accepted, the final decision is summarised here.  -->
