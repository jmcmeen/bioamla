# bioamla

A Python library and CLI for **bioacoustics and machine-learning applications** — audio
I/O and signal processing, acoustic indices, event detection, spectrogram visualization,
embedding clustering, species catalogs, datasets, and AST-based ML inference.

> **Prerelease:** 0.2.x is a ground-up rebuild. APIs may still change.

## Design

`bioamla` is organized by **domain**, not by layer. Each domain is a focused subpackage you
import directly:

| Import | What it does |
|--------|--------------|
| `bioamla.audio` | audio I/O, analysis, signal processing (filter/denoise/normalize/resample/segment), playback |
| `bioamla.viz` | spectrograms, mel/MFCC, waveform plots |
| `bioamla.indices` | acoustic indices — ACI, ADI, AEI, BIO, NDSI, spectral/temporal entropy |
| `bioamla.detect` | event detection — energy, RIBBIT, peaks, accelerating-pattern |
| `bioamla.cluster` | embedding dimensionality reduction, clustering, novelty detection |
| `bioamla.catalogs` | external catalogs — Xeno-canto, iNaturalist, eBird, Macaulay, HuggingFace |
| `bioamla.datasets` | dataset merge / augment / licensing, annotation conversion |
| `bioamla.ml` | Audio Spectrogram Transformer (AST) inference, training, embeddings |
| `bioamla.batch` | generic batch engine (directory + CSV-metadata modes) |
| `bioamla.system` | configuration, dependency checks, environment info |

Two conventions matter for consumers:

- **Errors are exceptions.** Functions return plain data and raise from a single hierarchy
  rooted at `bioamla.exceptions.BioamlaError` (e.g. `AudioLoadError`, `InvalidInputError`,
  `DependencyError`). Catch the base class to handle everything.
- **Batteries included.** A single `pip install bioamla` installs the full runtime stack —
  audio/signal/indices/detect/viz, the PyTorch + AST ML stack, clustering, and playback. Heavy
  imports are still **lazy**, so `import bioamla` and `bioamla --help` stay fast and don't load
  torch until you actually call a feature that needs it.

## Install

```bash
pip install bioamla                 # the full library + CLI

pip install "bioamla[dev]"          # + contributor tooling (pytest, ruff, mkdocs)
```

Requires Python ≥ 3.10. Audio I/O uses `ffmpeg`/`ffprobe` — install them via your OS package
manager (`bioamla config deps` checks what's available).

## Configuration (API keys)

The `bioamla.catalogs` providers need API keys to reach their services:

| Variable | Used by | Get a key |
| --- | --- | --- |
| `EBIRD_API_KEY` | eBird catalog | <https://ebird.org/api/keygen> |
| `XC_API_KEY` | Xeno-canto catalog | <https://xeno-canto.org/account> |
| `HF_TOKEN` | HuggingFace push/pull (datasets, models) | <https://huggingface.co/settings/tokens> |

You can either export these as environment variables, or drop them in a `.env` file in your
working directory — `bioamla` loads it automatically on import (for both the CLI and library
use). A real exported variable always takes precedence over the file.

```bash
# .env
EBIRD_API_KEY=your_key_here
XC_API_KEY=your_key_here
# HF_TOKEN=your_token_here
```

Keys are only required for the catalog/HuggingFace features that use them; the rest of the
library works without any configuration.

## Library quickstart

```python
from bioamla.audio import load_audio_data
from bioamla.indices import compute_all_indices
from bioamla.viz import generate_spectrogram
from bioamla.exceptions import BioamlaError

try:
    audio = load_audio_data("recording.wav")          # -> AudioData (mono float32 + sample_rate)

    idx = compute_all_indices(audio.samples, audio.sample_rate, include_entropy=True)
    print(idx.aci, idx.ndsi, idx.h_spectral)

    generate_spectrogram("recording.wav", output_path="spec.png")
except BioamlaError as e:
    print(f"failed: {e}")
```

AST inference:

```python
from bioamla.ml import predict_file
pred = predict_file("frog.wav", model_path="bioamla/scp-frogs")
print(pred.predicted_label, pred.confidence)
```

## CLI quickstart

```bash
bioamla --help                                  # all command groups
bioamla audio info recording.wav                # metadata
bioamla audio filter in.wav -o out.wav --bandpass-low 500 --bandpass-high 8000
bioamla indices compute recording.wav           # ACI/ADI/AEI/BIO/NDSI + entropy
bioamla detect energy recording.wav             # energy-based detections
bioamla audio visualize recording.wav -o spec.png

# Batch — over a directory or a CSV metadata file (with a `file_name` column):
bioamla batch indices calculate --input-dir ./recordings --output-dir ./out
bioamla batch indices calculate --input-file meta.csv --output-dir ./out   # merges results into the CSV
bioamla batch audio convert --input-dir ./wavs --output-dir ./flac --format flac

# Catalogs, models, datasets, config:
bioamla catalogs xeno-canto search --species "Hyla cinerea"
bioamla models ast predict frog.wav --model-path bioamla/scp-frogs
bioamla config deps                                                    # check system deps
```

## Development

```bash
make install        # uv sync --extra dev (full stack + tooling)
make test           # pytest
make check          # lint + format-check + test
```

`make install` brings in the full runtime stack plus contributor tooling, so the whole test
suite runs.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
