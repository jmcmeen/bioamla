---
name: run-bioamla
description: Build, run, and smoke-test the bioamla bioacoustics CLI. Use when asked to run, start, launch, build, smoke-test, or verify the bioamla command, or to generate a spectrogram / acoustic indices / detection from audio with the CLI.
---

# Run bioamla

`bioamla` is a Python **CLI + library** for bioacoustics (audio I/O, acoustic
indices, event detection, spectrograms, clustering, AST ML inference). There is
no GUI or server — the surface is the `bioamla` command, run through the project
venv via `uv run`. Most PRs touch a domain package (`audio`, `indices`,
`detect`, …) or its thin Click command in `cli/commands/`.

The driver is **`.claude/skills/run-bioamla/smoke.sh`**: it synthesizes a WAV
(no fixture files needed) and runs a representative flow across config / audio /
indices / detect / visualize, asserting exit codes and that artifacts exist.

> Paths below are relative to the repo root; the smoke driver `cd`s there itself.

## Prerequisites

- `uv` (already on PATH here; install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).
- `ffmpeg` + `ffprobe` for non-WAV formats. On Debian/Ubuntu:
  ```bash
  sudo apt-get install -y ffmpeg libsndfile1 libportaudio2
  ```
  Verify with `uv run bioamla config deps` (checks FFmpeg / libsndfile / PortAudio).

## Build / install

One command — `uv` resolves the lockfile and installs the full runtime stack + dev tooling:

```bash
make install      # == uv sync --extra dev
```

Confirm the CLI is wired up (startup is fast — heavy deps like torch load lazily):

```bash
uv run bioamla --version       # -> bioamla, version 0.2.0
uv run bioamla --help          # lists command groups
```

## Run (agent path) — the smoke driver

```bash
.claude/skills/run-bioamla/smoke.sh
```

It prints a per-step trace and ends with:

```
==> ALL SMOKE STEPS PASSED
==> artifacts in: /tmp/bioamla-smoke.XXXXXX
==> spectrogram PNG: /tmp/bioamla-smoke.XXXXXX/spectrogram.png
```

**Open that PNG and look at it** — for the synthetic 440 Hz + 880 Hz tones you
should see two bright horizontal bands near 440 and 880 Hz. A blank/dark image
means the visualize path is broken even though exit code was 0.

## Run individual commands

All real, all run this session. Make a WAV first, then call any domain command:

```bash
W=$(mktemp -d)
uv run python - "$W/t.wav" <<'PY'
import sys, numpy as np, soundfile as sf
sr = 22050; t = np.linspace(0, 5, sr*5, endpoint=False)
sf.write(sys.argv[1], (0.4*np.sin(2*np.pi*440*t)).astype(np.float32), sr)
PY

uv run bioamla audio info "$W/t.wav"
uv run bioamla indices compute "$W/t.wav"            # ACI/ADI/AEI/BIO/NDSI/entropy
uv run bioamla detect energy "$W/t.wav" --format json
uv run bioamla audio visualize "$W/t.wav" -o "$W/spec.png" -t mel   # mel|stft|mfcc|waveform
uv run bioamla audio segment "$W/t.wav" "$W/segs" -d 2.0
```

`indices compute` prints a table like:

```
  ACI:  8653.12
  ADI:  0.047
  AEI:  0.898
  BIO:  31.96
  NDSI: -0.773
  H (spectral): 1.254
  H (temporal): 8.751
```

Command groups: `annotation audio batch catalogs cluster config dataset detect
indices models`. Use `uv run bioamla <group> --help` to discover subcommands.

## Test

```bash
make test-fast    # pytest, skips slow/integration
make check        # lint + format-check + test  (the gate before "done")
```

## Gotchas

- **Always `uv run bioamla …`**, never a bare `bioamla`. The entry point lives
  in the project venv; a bare call hits the system PATH and won't find it.
- **WAV needs no ffmpeg; mp3/flac/ogg do.** `audio convert`/`segment` to a
  compressed format shells out to ffmpeg. If it's missing you get a
  `DependencyError`, not a crash — install ffmpeg (see Prerequisites).
- **`models ast` commands download a checkpoint from HuggingFace** (network +
  hundreds of MB) and load torch/transformers. The smoke driver deliberately
  skips them so it stays offline and fast. To exercise them you need network and
  optionally `HF_TOKEN`.
- **`catalogs` commands hit external APIs** (Xeno-canto, eBird, iNaturalist) and
  need keys (`XC_API_KEY`, `EBIRD_API_KEY`) — also out of scope for the smoke test.
- **`audio play` needs a real audio device** (PortAudio/sounddevice). It blocks
  for the file's duration (Ctrl-C to stop), and raises on a headless box with no
  output device — keep it out of automated checks.
- **Errors surface as `Error: ...` on exit 1**, not tracebacks — every failure
  is a `BioamlaError` caught centrally in `cli/cli.py:main`. A traceback escaping
  to the terminal is itself a bug.

## Troubleshooting

- `make install` slow on first run: `uv` is resolving + downloading torch and the
  full ML stack. Subsequent `uv run` calls are instant (cached).
- Smoke step "audio convert -> flac" fails with `DependencyError`: ffmpeg not
  installed — `sudo apt-get install -y ffmpeg`.
- `uv run bioamla` errors with a missing-module import: env drifted from the
  lockfile — re-run `make install`.
