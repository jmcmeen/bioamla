---
name: catalog-smoke
description: Smoke-test the bioamla catalog integrations (Xeno-canto, eBird, iNaturalist, HuggingFace) against live APIs. Use when asked to verify catalog commands, test API key configuration, or confirm that external data sources are reachable.
---

# Catalog smoke test

`bioamla catalogs` connects to four external providers: Xeno-canto (`xc`), eBird
(`ebird`), iNaturalist (`inat`), and HuggingFace (`hf`). These commands require API
keys and a network connection, so they are **intentionally excluded** from the
`run-bioamla` smoke driver. This skill fills that gap.

> Paths below are relative to the repo root; the smoke driver `cd`s there itself.

## Prerequisites

API keys — set in your shell or in `.env` at the repo root:

| Key | Provider | Where to get it |
|---|---|---|
| `XC_API_KEY` | Xeno-canto | https://xeno-canto.org/explore/api |
| `EBIRD_API_KEY` | eBird | https://ebird.org/api/keygen |
| `HF_TOKEN` | HuggingFace | https://huggingface.co/settings/tokens |

iNaturalist does not require a key for read-only taxon searches.

Each check is guarded: if its key is absent the step prints `SKIP` and exits 0,
so the script is safe to run in environments where only some keys are set.

## Run

```bash
.claude/skills/catalog-smoke/smoke.sh
```

The script prints a per-step trace. Steps for missing keys are skipped; all
others must return a zero exit code and non-empty output.

## Run individual commands

```bash
# Xeno-canto — search for recordings of a species
uv run bioamla catalogs xc search -s "Turdus migratorius"

# eBird — look up a species record
uv run bioamla catalogs ebird species "American Robin"

# iNaturalist — taxon search (no key required)
uv run bioamla catalogs inat search -s "Rana"

# HuggingFace — list locally cached datasets/models
uv run bioamla catalogs hf cache
```

## Gotchas

- **Rate limits apply.** Xeno-canto and eBird throttle requests; the smoke
  driver uses a single low-cost query per provider.
- **iNaturalist read-only searches need no key**, but `inat download` does
  require authentication.
- **HuggingFace `hf push-*` commands modify remote repos** — the smoke driver
  only calls `hf cache` (read-only).
- **Network required.** This skill does not work in an offline environment.
  Use `run-bioamla` for the offline smoke path.
- Errors surface as `Error: ...` on exit 1 (centrally caught in
  `cli/cli.py:main`). A raw traceback escaping to the terminal is a bug.

## Troubleshooting

- `Error: Unauthorized` / 403: key is set but invalid — regenerate it at the
  provider.
- `Error: ...rate limit...`: wait a few seconds and retry.
- Step completes with empty output: the query returned no results — try a more
  common species name (e.g. "robin" for xc/ebird).
