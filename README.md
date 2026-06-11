# bioamla

A Python library and CLI for **bioacoustics and machine-learning applications** — audio
I/O and signal processing, acoustic indices, event detection, spectrogram visualization,
embedding clustering, species catalogs, datasets, and AST-based ML inference.

> **Prerelease:** 0.2.x is a ground-up rebuild. APIs may still change.

## Design

`bioamla` is organized by **domain**, not by layer. Each domain is a focused subpackage you
import directly:

| Import | What it does |
| --- | --- |
| `bioamla.audio` | audio I/O, analysis, signal processing (filter/denoise/normalize/resample/segment), playback |
| `bioamla.viz` | spectrograms, mel/MFCC, waveform plots |
| `bioamla.indices` | acoustic indices — ACI, ADI, AEI, BIO, NDSI, spectral/temporal entropy |
| `bioamla.detect` | event detection — energy, RIBBIT, peaks, accelerating-pattern |
| `bioamla.cluster` | embedding dimensionality reduction, clustering, novelty detection |
| `bioamla.catalogs` | external catalogs — Xeno-canto, iNaturalist, eBird, Macaulay, HuggingFace |
| `bioamla.datasets` | dataset merge / augment / licensing, annotation conversion, labeled-clip extraction |
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

# Audio editing transforms (deterministic, single-file):
bioamla audio pitch-shift in.wav out.wav --steps 2
bioamla audio time-stretch in.wav out.wav --rate 1.2
bioamla audio add-noise in.wav out.wav --snr-db 15

# Batch — over a directory or a CSV metadata file (with a `file_name` column):
bioamla batch indices calculate --input-dir ./recordings --output-dir ./out
bioamla batch indices calculate --input-file meta.csv --output-dir ./out   # merges results into the CSV
bioamla batch audio convert --input-dir ./wavs --output-dir ./flac --format flac

# Catalogs, models, datasets, config:
bioamla catalogs xc search --species "Hyla cinerea"
bioamla catalogs hf pull-dataset ashraq/esc50 ./esc50      # Hub dataset -> labeled-folder layout
bioamla catalogs hf cache --datasets                       # inspect/purge the HF cache (--purge)
bioamla models ast predict frog.wav --model-path bioamla/scp-frogs
bioamla models ast predict soundscape.wav --segment-duration 3 -o preds.csv   # classify each 3s segment
bioamla models ast train --train-dataset ashraq/esc50      # grab-and-go: train off a Hub id directly
bioamla models ast train --train-dataset ./esc50 --config train.toml   # or from local data + a config
bioamla config deps                                                    # check system deps
```

### Two kinds of augmentation

bioamla keeps a deliberate boundary between **audio editing** and the
**pre-training augmentation layer**:

- **Editing ops** (`bioamla.audio` — `pitch_shift`, `time_stretch`, `add_noise`,
  `apply_gain`, plus filter/denoise/normalize/resample/trim) are deterministic
  single-file transforms you apply with explicit parameters.
- **The augmentation layer** (`bioamla.datasets.create_augmentation_pipeline`) is
  the randomized, range+probability pipeline used both for synthetic dataset
  generation (`dataset augment`) and on-the-fly during `models ast train`.

## Example workflows

End-to-end bioacoustics studies wired from the CLI live in
[`examples/`](https://github.com/jmcmeen/bioamla/tree/main/examples) — catalog →
annotate → dataset → train → publish, pulling a Hub dataset to fine-tune,
soundscape analysis, and embedding clustering. The offline

## Development

```bash
make install        # uv sync --extra dev (full stack + tooling)
make test           # pytest
make check          # lint + format-check + test
```

`make install` brings in the full runtime stack plus contributor tooling, so the whole test
suite runs.

## Contributing

Contributions are welcome — see
[CONTRIBUTING.md](https://github.com/jmcmeen/bioamla/blob/main/CONTRIBUTING.md) for setup and
conventions, and the
[Code of Conduct](https://github.com/jmcmeen/bioamla/blob/main/CODE_OF_CONDUCT.md). Security
issues should be reported privately per our
[Security Policy](https://github.com/jmcmeen/bioamla/blob/main/SECURITY.md).

## License

GNU General Public License v3.0 — see [LICENSE](https://github.com/jmcmeen/bioamla/blob/main/LICENSE).
