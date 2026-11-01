# bioamla

[![PyPI version](https://img.shields.io/pypi/v/bioamla.svg)](https://pypi.org/project/bioamla/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jmcmeen/bioamla/blob/main/LICENSE)
[![CI](https://github.com/jmcmeen/bioamla/actions/workflows/ci.yml/badge.svg)](https://github.com/jmcmeen/bioamla/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://jmcmeen.github.io/bioamla)

A Python library and command-line tool for bioacoustics: audio and recorder metadata,
signal processing, acoustic indices, event detection, spectrograms, species catalogs,
datasets, and Audio Spectrogram Transformer (AST) models. Run it on your laptop, or on a
server next to your recordings.

## Install

Python 3.11 or newer, in a virtual environment:

```bash
pip install "bioamla[all,cli]"     # everything, plus the `bioamla` command
pip install "bioamla[audio,cli]"   # or only the parts you need
pip install bioamla                # the slim library alone: no PyTorch, librosa or plots
```

| Extra | For |
|---|---|
| *(none)* | audio I/O and info, recorder metadata, batch runs, datasets, catalogs |
| `audio` | signal processing (filter, denoise, segment, ...) and playback |
| `viz` | spectrograms, plots and videos |
| `indices` | acoustic indices |
| `detect` | event detection |
| `ml` | AST prediction, training and embeddings (PyTorch) |
| `cluster` | dimensionality reduction and clustering |
| `server` | `bioamla serve` |
| `cli` | the `bioamla` command |
| `all` | every extra but `cli` |

Using a feature whose extra is missing tells you what to install.

### System dependencies

Formats other than WAV, FLAC and OGG, and spectrogram videos, need FFmpeg
(`apt install ffmpeg`, `brew install ffmpeg`, or `conda install -c conda-forge ffmpeg`).
AST prediction can use its shared libraries (versions 4 to 8) to decode audio. Check what
you have with `bioamla system deps`.

## Quickstart

```bash
bioamla audio info recording.wav                  # duration, rate, channels, bit depth
bioamla audio metadata recording.wav              # recorder metadata: when, where, which unit
bioamla audio visualize recording.wav -o spec.png
bioamla indices compute recording.wav             # ACI, ADI, AEI, BIO, NDSI, entropy
bioamla detect energy recording.wav -o detections.csv
bioamla models ast predict frog.wav --model-path bioamla/scp-frogs --top-k 3

# The same over a folder, or a CSV with a file_name column (a copy with the results
# added goes to the output folder):
bioamla batch index --input-dir ./recordings --output-dir ./out
bioamla batch detect energy --input-file recordings.csv --output-dir ./out
```

`bioamla --json <command>` prints one JSON document, for scripts and agents. Every
command has `--help`, and the [CLI reference](https://jmcmeen.github.io/bioamla/cli/)
lists them all.

Everything the command does, the library does too:

```python
from bioamla.audio import load_audio_data
from bioamla.indices import compute_all_indices
from bioamla.ml import classify_file

audio = load_audio_data("recording.wav")
print(compute_all_indices(audio.samples, audio.sample_rate).aci)
print(classify_file("frog.wav", model_path="bioamla/scp-frogs").predictions[0].predicted_label)
```

Errors raise `bioamla.exceptions.BioamlaError` (or a subclass).

## Workspaces

A workspace is a project folder that remembers its recordings, settings and history:

```bash
bioamla workspace init costa-rica && cd costa-rica
bioamla workspace add /data/recordings/site1      # link recordings where they are
bioamla workspace config set detect.energy.threshold_db -25
bioamla workspace history                          # what ran, when, and what it wrote
```

## Working on a server

Run bioamla on the machine that holds the recordings (or the GPU), and drive it from
your laptop with the same commands:

```bash
# On the server (it listens on localhost only)
pip install "bioamla[all,cli]"
bioamla serve --root ~/bioamla-workspaces --allow-link /data/recordings

# On the laptop: reach it through an SSH tunnel, then add it as a remote
ssh -N -L 8765:localhost:8765 lab-server &
bioamla remote add lab http://localhost:8765
export BIOAMLA_LAB_TOKEN=...                       # from the server's token file
bioamla remote workspaces lab --create costa-rica
bioamla remote upload lab costa-rica rec.wav --to raw
bioamla --remote lab:costa-rica detect energy raw/rec.wav
```

The output is the same as a local run's. See
[Running the server](https://jmcmeen.github.io/bioamla/server/) for the token, a VPN
instead of a tunnel, and systemd.

## Agent skills

[`skills/`](https://github.com/jmcmeen/bioamla/tree/main/skills) has skills that let an
AI agent drive bioamla, locally or on a server: `inventory` describes a folder of
recordings, and `label` does a first pass of labeling, sending anything uncertain to a
person. See [Agent skills](https://jmcmeen.github.io/bioamla/skills/).

## API keys

The catalog commands read `EBIRD_API_KEY`, `XC_API_KEY` (Xeno-canto) and `HF_TOKEN`
(Hugging Face) from the environment, or from a `.env` file in the working directory or a
parent (only those three keys, and an exported variable wins). Nothing else needs a key.

## More

- Documentation: the [CLI reference](https://jmcmeen.github.io/bioamla/cli/),
  [using the library](https://jmcmeen.github.io/bioamla/library/) and its
  [API reference](https://jmcmeen.github.io/bioamla/api/audio/),
  [settings](https://jmcmeen.github.io/bioamla/settings/) and
  [recorder metadata](https://jmcmeen.github.io/bioamla/recorder-metadata/).
- [Examples](https://github.com/jmcmeen/bioamla/tree/main/examples): end-to-end scripts,
  from catalog downloads to a trained and published model.
- [Contributing](https://github.com/jmcmeen/bioamla/blob/main/CONTRIBUTING.md),
  [Code of Conduct](https://github.com/jmcmeen/bioamla/blob/main/CODE_OF_CONDUCT.md),
  [Security Policy](https://github.com/jmcmeen/bioamla/blob/main/SECURITY.md).

## Citation

If you use bioamla in research, please cite the Zenodo DOI of the version you used, or
use GitHub's "Cite this repository" button
([`CITATION.cff`](https://github.com/jmcmeen/bioamla/blob/main/CITATION.cff)):

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

MIT. See [LICENSE](https://github.com/jmcmeen/bioamla/blob/main/LICENSE).
