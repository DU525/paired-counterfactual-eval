# Security Policy

## Supported versions

| version | supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's
private vulnerability reporting on this repository
("Security" → "Report a vulnerability"), or contact the maintainer directly.

Please include: what you did, what you expected, what happened, and the
minimal reproduction if you have one.

## What counts as a security issue here

- **Credential exposure.** This project is designed to run evaluations that
  touch proprietary simulators and checkpoints. A leaked API key, token,
  private key or a `.env` file committed to the repository is the main risk.
- **Path traversal** in artefact handling: an artefact name that escapes the
  transaction root.
- **Unverified data** being accepted as verified: a bundle that passes
  verification although its artefacts were modified.
- **Budget bypass**: a path that performs simulation steps without being
  charged by the meter.

## What is not a vulnerability

- Failing to reproduce a result.
- A negative result you disagree with.
- Missing documentation.

## Secret hygiene in this repository

- `.env` files, keys and certificates are git-ignored; only `.env.example`
  templates are tracked.
- A scanner lives at `scripts/scan_repo.py`. Run it before every push:

  ```bash
  python scripts/scan_repo.py .
  ```

  It reports the file, line and a **sha256 fingerprint**, never the secret
  itself, so the output is safe to paste into an issue or a pull request.
- If a credential is exposed, rotate it first, then remove it from the
  history. Removing a file from the tip of a branch is not enough.