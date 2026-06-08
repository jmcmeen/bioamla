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
- **Heavy dependencies are optional.** The base install is lightweight; `import bioamla` pulls
  no torch/opensoundscape/etc. Those load lazily when you call a feature that needs them and
  raise `DependencyError` telling you which extra to install.

## Install

```bash
pip install bioamla                 # slim core (audio, indices, detect-energy, catalogs, CLI)

pip install "bioamla[ml]"           # + AST inference/training/embeddings (torch, transformers)
pip install "bioamla[detect]"       # + RIBBIT and OpenSoundscape-backed detectors
pip install "bioamla[cluster]"      # + UMAP / HDBSCAN clustering
pip install "bioamla[playback]"     # + local audio playback (sounddevice)
pip install "bioamla[all]"          # everything above
```

Requires Python ≥ 3.10. Audio I/O uses `ffmpeg`/`ffprobe` — install them via your OS package
manager (`bioamla config deps` checks what's available).

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

AST inference (needs `bioamla[ml]`):

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
bioamla models ast predict frog.wav --model-path bioamla/scp-frogs     # needs [ml]
bioamla config deps                                                    # check system deps
```

## Development

```bash
make install        # uv sync with dev extras
make test           # pytest
make check          # lint + format-check + test
```

The test suite skips tests whose optional extra isn't installed, so it is green on a slim
install and exhaustive with `bioamla[all]`.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
