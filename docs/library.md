# Using the library

Everything the `bioamla` command does is a function you can import. This page is how the
package is laid out and what to expect from it; the [API reference](api/audio.md) has
every function.

## Packages

bioamla is organized by domain. Import the one you need:

| Import | What it does |
| --- | --- |
| `bioamla.audio` | audio I/O and analysis, signal processing (filter, denoise, normalize, resample, segment), playback, recorder metadata (GUANO, AudioMoth, Wildlife Acoustics, `bext`) |
| `bioamla.viz` | spectrograms, mel and MFCC, waveform plots, spectrogram videos |
| `bioamla.indices` | acoustic indices: ACI, ADI, AEI, BIO, NDSI, spectral and temporal entropy |
| `bioamla.detect` | event detection: energy, RIBBIT, peaks, accelerating patterns |
| `bioamla.cluster` | embedding dimensionality reduction, clustering, novelty detection |
| `bioamla.catalogs` | Xeno-canto, iNaturalist, eBird, Macaulay Library, Hugging Face |
| `bioamla.datasets` | merging, augmentation, licensing, annotation formats, labeled clips, splits |
| `bioamla.ml` | Audio Spectrogram Transformer (AST) prediction, training, embeddings |
| `bioamla.batch` | the batch engine: a folder of files, or a CSV with a `file_name` column |
| `bioamla.workspace` | project workspaces: artifacts, settings, run history |
| `bioamla.client` | a client for a bioamla server (`bioamla serve`) |
| `bioamla.system` | dependency checks, version and environment |

## Errors

Functions return plain data and raise on failure, always a subclass of
`bioamla.exceptions.BioamlaError` (`AudioLoadError`, `InvalidInputError`,
`DependencyError`, ...). Catch the base class to handle them all:

```python
from bioamla.exceptions import BioamlaError

try:
    ...
except BioamlaError as e:
    print(f"failed: {e}")
```

## Extras and imports

`pip install bioamla` installs a slim base. The rest comes in extras (see the
[README](index.md#install)); using a function whose extra is missing raises
`DependencyError` with the `pip install` command to run. Heavy dependencies (PyTorch,
librosa, ...) are imported only when used, so `import bioamla` stays fast.

## Editing audio versus augmenting it

Two kinds of audio change are kept apart:

- **Edits** (`bioamla.audio`: `pitch_shift`, `time_stretch`, `add_noise`, `apply_gain`,
  filter, denoise, normalize, resample, trim) are deterministic: one file, explicit
  parameters, the same result every time.
- **Augmentation** (`bioamla.datasets.create_augmentation_pipeline`) is randomized: ranges
  and probabilities, used to generate synthetic datasets (`dataset augment`) and during
  `models ast train`.
