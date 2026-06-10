# Security Policy

## Supported versions

bioamla is on the 0.2.x line (a ground-up rebuild); security fixes target the
latest release on `main`.

| Version | Supported |
|---|---|
| 0.2.x | ✅ |
| < 0.2 | ❌ |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue or PR.

- Preferred: open a [GitHub security advisory](https://github.com/jmcmeen/bioamla/security/advisories/new)
  ("Report a vulnerability").
- Or email **johnmcmeen@gmail.com** with details and reproduction steps.

We'll acknowledge within a few days and keep you updated on the fix and
disclosure timeline. Please give us reasonable time to address the issue before
any public disclosure.

## Scope notes

bioamla downloads models and datasets from external services (HuggingFace Hub,
Xeno-canto, iNaturalist, eBird, Macaulay) and loads model weights. Treat
untrusted model/dataset sources with the same caution as any other untrusted
input. API keys are read from the environment / `.env` at call time and are never
logged.
