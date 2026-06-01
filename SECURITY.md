# Security Policy

## Supported versions

Pose follows SemVer (see [`VERSIONING.md`](VERSIONING.md)). Security fixes are backported to:

| Version | Supported |
|---|---|
| Latest minor of the current major | ✅ |
| Previous major (most recent minor) | ✅ for high/critical only |
| Older versions | ❌ |

If you need a security fix on a version older than the supported range, please open an issue describing the constraints and we will discuss.

## Reporting a vulnerability

**Do not** open a public GitHub issue for security vulnerabilities.

Email **sari.sabban@trubuild.io** with:

- A description of the vulnerability
- Steps to reproduce or a proof-of-concept
- Your assessment of the impact
- Your name and affiliation (if you would like credit)

You should receive an initial acknowledgement within **5 business days**. We aim to:

- Confirm the vulnerability within **14 days**
- Issue a fix within **30 days** for high/critical severity, **90 days** for medium/low

We follow coordinated disclosure: we will work with you on a disclosure timeline and credit you in the release notes unless you prefer otherwise.

## Threat model

Pose is a Python library that reads structural biology file formats (PDB, mmCIF, FASTA, SDF, MOL, MOL2, custom JSON parameter sets) and writes results to disk. Realistic threats include:

- **Malicious input files** that exploit a parser bug to cause denial-of-service (excessive CPU/memory), crash, or to escape the input domain (e.g., path traversal through file-name fields). Reports involving crafted input files are in scope.
- **Untrusted JSON parameter files** loaded by `tools.Port()` or `pose/database.json`. Reports of code execution or path traversal via these inputs are in scope.
- **Dependency vulnerabilities** in the (small) set of dependencies. We pin and update them via Dependabot.

Out of scope:

- Pose makes no security guarantees against untrusted code that imports Pose and uses its internal helper utilities outside the documented API.
- Pose does not provide cryptographic guarantees; it is not used for authentication or signing.

## Encryption for the report email

If you need encrypted communication, request the BDFL's PGP key via the same address before sending sensitive details.

## Hall of fame

Researchers who have responsibly disclosed valid vulnerabilities will be listed here (with their consent).

_None yet._
