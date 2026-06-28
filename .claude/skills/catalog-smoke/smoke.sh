#!/usr/bin/env bash
# Smoke test for `bioamla catalogs` — exercises live catalog API integrations.
#
# Each check is guarded: if its required env key is absent the step is skipped
# (SKIP message, exit 0 for that step). The script exits non-zero only when a
# check whose key IS present fails.
#
# Usage:  .claude/skills/catalog-smoke/smoke.sh
# Keys:   XC_API_KEY   — Xeno-canto
#         EBIRD_API_KEY — eBird
#         HF_TOKEN     — HuggingFace (only needed for push/pull; cache is keyless)
#         iNaturalist read-only taxon search needs no key.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

b() { uv run bioamla "$@"; }

step()  { echo; echo "### $*"; }
ok()    { echo "    OK: $*"; }
skip()  { echo "    SKIP: $*"; }
die()   { echo "    FAIL: $*" >&2; exit 1; }

PASSED=0
SKIPPED=0

# ---------------------------------------------------------------------------
# Xeno-canto — requires XC_API_KEY
# ---------------------------------------------------------------------------
step "catalogs xc search (Xeno-canto)"
if [ -z "${XC_API_KEY:-}" ]; then
    skip "XC_API_KEY not set"
    SKIPPED=$((SKIPPED + 1))
else
    out=$(b catalogs xc search -s "robin" 2>&1) || die "xc search failed: $out"
    [ -n "$out" ] || die "xc search returned empty output"
    ok "xc search"
    PASSED=$((PASSED + 1))
fi

# ---------------------------------------------------------------------------
# eBird — requires EBIRD_API_KEY
# ---------------------------------------------------------------------------
step "catalogs ebird species (eBird)"
if [ -z "${EBIRD_API_KEY:-}" ]; then
    skip "EBIRD_API_KEY not set"
    SKIPPED=$((SKIPPED + 1))
else
    out=$(b catalogs ebird species "American Robin" 2>&1) || die "ebird species failed: $out"
    [ -n "$out" ] || die "ebird species returned empty output"
    ok "ebird species"
    PASSED=$((PASSED + 1))
fi

step "catalogs ebird search (eBird)"
if [ -z "${EBIRD_API_KEY:-}" ]; then
    skip "EBIRD_API_KEY not set"
    SKIPPED=$((SKIPPED + 1))
else
    out=$(b catalogs ebird search "robin" 2>&1) || die "ebird search failed: $out"
    [ -n "$out" ] || die "ebird search returned empty output"
    ok "ebird search"
    PASSED=$((PASSED + 1))
fi

# ---------------------------------------------------------------------------
# iNaturalist — no key required for taxon search
# ---------------------------------------------------------------------------
step "catalogs inat search (iNaturalist — keyless)"
out=$(b catalogs inat search -s "Rana" 2>&1) || die "inat search failed: $out"
[ -n "$out" ] || die "inat search returned empty output"
ok "inat search"
PASSED=$((PASSED + 1))

# ---------------------------------------------------------------------------
# HuggingFace cache — keyless read of local cache metadata
# ---------------------------------------------------------------------------
step "catalogs hf cache (HuggingFace — keyless)"
out=$(b catalogs hf cache 2>&1) || die "hf cache failed: $out"
ok "hf cache"
PASSED=$((PASSED + 1))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "==> CATALOG SMOKE DONE  passed=$PASSED  skipped=$SKIPPED"
if [ "$SKIPPED" -gt 0 ]; then
    echo "    (set missing API keys to run skipped checks)"
fi
