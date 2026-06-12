#!/usr/bin/env bash
# Smoke test for the `bioamla` CLI — drives the real installed command end to end.
#
# Generates a synthetic WAV (so it needs no fixture files), then runs a
# representative flow across domains: config / audio / indices / detect /
# visualize. Each step asserts a zero exit and, where a file is produced,
# that the artifact exists. Exits non-zero on the first failure.
#
# Usage:  ./smoke.sh
# Output: artifacts land in a temp dir, printed at the end (and the
#         spectrogram PNG path is echoed so an agent can open/inspect it).
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# uv-run wrapper so we use the project venv regardless of CWD shell state.
b() { uv run bioamla "$@"; }

WORK="$(mktemp -d /tmp/bioamla-smoke.XXXXXX)"
WAV="$WORK/tone.wav"
echo "==> work dir: $WORK"

step() { echo; echo "### $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "    FAIL: $*" >&2; exit 1; }

step "generate synthetic 5s WAV (440Hz + 880Hz tones, 22050 Hz)"
uv run python - "$WAV" <<'PY'
import sys, numpy as np, soundfile as sf
sr = 22050; dur = 5.0
t = np.linspace(0, dur, int(sr*dur), endpoint=False)
y = 0.4*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*880*t)
sf.write(sys.argv[1], y.astype(np.float32), sr)
print("wrote", sys.argv[1])
PY
[ -s "$WAV" ] || die "WAV not written"
ok "WAV created"

step "system version (env + version info)"
b system version >/dev/null || die "system version"
ok "system version"

step "system deps (system dependency check)"
b system deps >/dev/null || die "system deps"
ok "system deps"

step "audio info"
b audio info "$WAV" >/dev/null || die "audio info"
ok "audio info"

step "audio segment into 2s clips"
SEGDIR="$WORK/segments"
b audio segment "$WAV" "$SEGDIR" -d 2.0 >/dev/null || die "audio segment"
n=$(find "$SEGDIR" -name '*.wav' | wc -l)
[ "$n" -ge 2 ] || die "expected >=2 segments, got $n"
ok "audio segment produced $n clips"

step "audio convert to flac"
FLAC="$WORK/tone.flac"
b audio convert "$WAV" "$FLAC" >/dev/null || die "audio convert"
[ -s "$FLAC" ] || die "flac not written"
ok "audio convert -> flac"

step "audio visualize -> mel spectrogram PNG"
PNG="$WORK/spectrogram.png"
b audio visualize "$WAV" -o "$PNG" -t mel >/dev/null || die "audio visualize"
[ -s "$PNG" ] || die "PNG not written"
ok "spectrogram: $PNG"

step "indices compute (all acoustic indices)"
b indices compute "$WAV" --format json >/dev/null || die "indices compute"
ok "indices compute"

step "detect energy (band-limited energy detection)"
b detect energy "$WAV" --format json >/dev/null || die "detect energy"
ok "detect energy"

echo
echo "==> ALL SMOKE STEPS PASSED"
echo "==> artifacts in: $WORK"
echo "==> spectrogram PNG: $PNG"
