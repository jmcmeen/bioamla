# bioamla

[![PyPI version](https://img.shields.io/pypi/v/bioamla.svg)](https://pypi.org/project/bioamla/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmcmeen/bioamla/blob/main/LICENSE)
[![CI](https://github.com/jmcmeen/bioamla/actions/workflows/ci.yml/badge.svg)](https://github.com/jmcmeen/bioamla/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://jmcmeen.github.io/bioamla)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python library and CLI for **bioacoustics and machine-learning applications** — audio
I/O and signal processing, acoustic indices, event detection, spectrogram visualization,
embedding clustering, species catalogs, datasets, and AST-based ML inference.

> **Prerelease:** 0.2.x is a ground-up rebuild. APIs may still change.

## Design

`bioamla` is organized by **domain**, not by layer. Each domain is a focused subpackage you
import directly:

| Import | What it does |
| --- | --- |
| `bioamla.audio` | audio I/O, analysis, signal processing (filter/denoise/normalize/resample/segment), playback, recorder metadata (GUANO, AudioMoth, Wildlife Acoustics, `bext`) |
| `bioamla.viz` | spectrograms, mel/MFCC, waveform plots, scrolling spectrogram videos |
| `bioamla.indices` | acoustic indices — ACI, ADI, AEI, BIO, NDSI, spectral/temporal entropy |
| `bioamla.detect` | event detection — energy, RIBBIT, peaks, accelerating-pattern |
| `bioamla.cluster` | embedding dimensionality reduction, clustering, novelty detection |
| `bioamla.catalogs` | external catalogs — Xeno-canto, iNaturalist, eBird, Macaulay, HuggingFace |
| `bioamla.datasets` | dataset merge / augment / licensing, annotation conversion, labeled-clip extraction |
| `bioamla.ml` | Audio Spectrogram Transformer (AST) inference, training, embeddings |
| `bioamla.batch` | generic batch engine (directory + CSV-metadata modes) |
| `bioamla.system` | system dependency checks, version & environment info |

Two conventions matter for consumers:

- **Errors are exceptions.** Functions return plain data and raise from a single hierarchy
  rooted at `bioamla.exceptions.BioamlaError` (e.g. `AudioLoadError`, `InvalidInputError`,
  `DependencyError`). Catch the base class to handle everything.
- **Slim base, opt-in extras.** `pip install bioamla` installs the light core: audio I/O and
  recorder metadata, file discovery, the batch engine, dataset tooling and the catalog clients.
  Signal processing, indices, detection, plotting, ML and clustering each come with an extra
  (see [Install](#install)). Calling a feature whose extra isn't installed raises
  `DependencyError` with the `pip install` command to run. Heavy imports stay **lazy**, so
  `import bioamla` and `bioamla --help` are fast either way.

## Install

Requires Python ≥ 3.10. Install into a **virtual environment** so bioamla's dependencies stay
isolated from other projects and your system Python:

```bash
python -m venv .venv && source .venv/bin/activate    # or: uv venv / conda create -n bioamla
pip install "bioamla[all,cli]"      # everything, plus the `bioamla` command
pip install bioamla                 # the slim base only
pip install "bioamla[audio,cli]"    # or pick the extras you need
```

> **Breaking change: the dependencies were split into extras.** Up to 0.2.x a plain
> `pip install bioamla` pulled in the whole stack (PyTorch, transformers, librosa, …). From
> 0.3.0 the base is slim and the heavy parts are extras. To keep the old behavior,
> install `bioamla[all]` (add `cli` for the command).

| Extra | Adds | Unlocks |
|---|---|---|
| *(base)* | numpy, scipy, pandas, pyarrow, soundfile, pydub, riffy, requests, huggingface_hub, pyinaturalist | audio I/O and info, resampling, recorder metadata, discovery, `run_batch`, `datasets` (entropy measurements need `indices`), `catalogs` |
| `audio` | librosa, sounddevice | librosa signal processing (denoise, segment, onsets, …), playback |
| `viz` | `audio` + matplotlib | `bioamla.viz`: spectrograms, plots, videos |
| `indices` | `audio` + numba | `bioamla.indices`: acoustic indices |
| `detect` | `audio` | `bioamla.detect`: event detection |
| `ml` | `audio` + torch, torchaudio, transformers, datasets, audiomentations, evaluate, mlflow, … | `bioamla.ml` (AST inference, training, embeddings; UMAP reduction also needs `cluster`), augmentation, Hugging Face datasets |
| `cluster` | umap-learn, hdbscan, scikit-learn | `bioamla.cluster` |
| `all` | every extra above | everything |
| `cli` | click, rich | the `bioamla` command |
| `dev` | `all` + `cli` + pytest, ruff, mkdocs, … | contributor tooling |

Without an extra, `import bioamla.viz` (or any function that needs it) raises `DependencyError`
naming the extra, e.g. `pip install 'bioamla[viz]'`; the `bioamla` command prints the same hint.
Without `[cli]`, the `bioamla` command prints a hint to install it.

### System dependencies

Beyond the Python packages, bioamla needs a working **FFmpeg** install on the host:

- **FFmpeg shared libraries (`libav*`), major version 4–8** — used by `torchcodec` (the backend
  behind `torchaudio.load`) to *decode* audio for waveform loading, AST inference/training, and
  HuggingFace dataset materialization.
- **The `ffmpeg`/`ffprobe` CLI** — used by `pydub` to *encode* non-WAV formats (FLAC/MP3/OGG/M4A)
  and to read their metadata. Plain WAV I/O works without FFmpeg: WAV headers and recorder
  metadata are read with [riffy](https://github.com/jmcmeen/riffy), a pure-Python RIFF parser.

Both come from a single FFmpeg install:

```bash
sudo apt-get install -y ffmpeg            # Debian/Ubuntu
sudo dnf install -y ffmpeg-free           # Fedora/RHEL
brew install ffmpeg                        # macOS
conda install -c conda-forge 'ffmpeg<9'    # conda env (no root); pin to a supported major version
```

Run `bioamla system deps` to check what's available. Without FFmpeg, audio decode/encode features
raise a clear error (and the affected tests skip rather than fail).

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
HF_TOKEN=your_token_here
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

Recorder metadata (what a field recorder wrote into the WAV: when, where, which unit):

```python
from bioamla.audio import get_recording_metadata
meta = get_recording_metadata("FROGNN-1_20230306_182817.wav")
print(meta.recorded_at, meta.latitude, meta.longitude, meta.device_model, meta.sources)
```

## CLI quickstart

```bash
bioamla --help                                  # all command groups
bioamla audio info recording.wav                # header: duration, rate, channels, bit depth
bioamla audio metadata recording.wav            # recorder metadata (GUANO, AudioMoth, WAMD, bext)
bioamla audio filter in.wav out.wav --bandpass-low 500 --bandpass-high 8000
bioamla indices compute recording.wav           # ACI/ADI/AEI/BIO/NDSI + entropy
bioamla detect energy recording.wav             # energy-based detections
bioamla audio visualize recording.wav -o spec.png
bioamla audio video recording.wav -o recording.mp4   # scrolling spectrogram, needs ffmpeg

# Audio editing transforms (deterministic, single-file):
bioamla audio pitch-shift in.wav out.wav --steps 2
bioamla audio time-stretch in.wav out.wav --rate 1.2
bioamla audio add-noise in.wav out.wav --snr-db 15

# Batch — over a directory or a CSV metadata file (with a `file_name` column):
bioamla batch index --input-dir ./recordings --output-dir ./out
bioamla batch index --input-file meta.csv --output-dir ./out   # merges results into the CSV
bioamla batch audio convert --input-dir ./wavs --output-dir ./flac --format flac
bioamla batch audio validate --input-dir ./wavs --recursive          # pre-flight: PCM WAV headers, exit 1 on damage
bioamla batch audio metadata --input-dir ./wavs --output-dir ./out   # recorder_metadata.csv per deployment

# Catalogs, models, datasets, system:
bioamla catalogs xc search --species "Hyla cinerea"
bioamla catalogs hf pull-dataset ashraq/esc50 ./esc50      # Hub dataset -> labeled-folder layout
bioamla catalogs hf cache --datasets                       # inspect/purge the HF cache (--purge)
bioamla models ast predict frog.wav --model-path bioamla/scp-frogs
bioamla models ast predict soundscape.wav --segment-seconds 3 -o preds.csv   # classify each 3s segment
bioamla models ast train --train-dataset ashraq/esc50      # grab-and-go: train off a Hub id directly
bioamla models ast train --train-dataset ./esc50 --config train.toml   # or from local data + a config
bioamla system deps                                                    # check system deps
```

Every command is self-documenting (`bioamla <group> <command> --help`), and the full
command tree is in the [CLI reference](https://jmcmeen.github.io/bioamla/cli/). The
[API reference](https://jmcmeen.github.io/bioamla/) documents the library.

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

Runnable end-to-end studies wired from the CLI live in
[`examples/`](https://github.com/jmcmeen/bioamla/tree/main/examples) — each is a
self-contained shell script (see the [examples README](https://github.com/jmcmeen/bioamla/tree/main/examples#readme)):
catalog → annotate → dataset → train → publish, fine-tuning from a Hub dataset,
soundscape analysis, embedding clustering, iNaturalist clip inference, spectrogram
videos in every style and frame shape, and describing a raw recorder deployment from
its WAV metadata.

## Development

```bash
make install        # uv sync --extra dev (every extra + tooling)
make test           # pytest
make check          # lint + format-check + test
```

`make install` brings in every extra plus contributor tooling, so the whole test
suite runs.

## Contributing

Contributions are welcome — see
[CONTRIBUTING.md](https://github.com/jmcmeen/bioamla/blob/main/CONTRIBUTING.md) for setup and
conventions, and the
[Code of Conduct](https://github.com/jmcmeen/bioamla/blob/main/CODE_OF_CONDUCT.md). Security
issues should be reported privately per our
[Security Policy](https://github.com/jmcmeen/bioamla/blob/main/SECURITY.md).

## Citation

If you use `bioamla` in your research, please cite it. Machine-readable metadata lives in
[`CITATION.cff`](https://github.com/jmcmeen/bioamla/blob/main/CITATION.cff) — GitHub's **"Cite this repository"** button generates APA and
BibTeX from it.

Each release is archived on [Zenodo](https://zenodo.org/) with a DOI. Prefer citing the DOI
for the specific version you used; otherwise cite the repository:

```bibtex
@software{mcmeen_bioamla,
  author  = {McMeen, John},
  title   = {bioamla: Bioacoustics \& Machine Learning Applications},
  year    = {2026},
  license = {MIT},
  url     = {https://github.com/jmcmeen/bioamla}
}
```

## License

MIT License — see [LICENSE](https://github.com/jmcmeen/bioamla/blob/main/LICENSE).
