# bioamla example workflows

End-to-end bioacoustics workflows wired together from the `bioamla` CLI. Each
script is heavily commented and meant to be **read first, then run** — they are
templates to adapt, not fixtures. Unlike [`dev-data/cli_test.sh`](../dev-data/cli_test.sh)
(an offline smoke test), these touch the network, the HuggingFace Hub, and (for
training) a GPU.

| Script | Flow | Needs |
| --- | --- | --- |
| [`01_catalog_to_model.sh`](01_catalog_to_model.sh) | catalog download → annotate → dataset → train → publish | `XC_API_KEY`/`EBIRD_API_KEY`, HF login, GPU |
| [`02_hf_dataset_to_model.sh`](02_hf_dataset_to_model.sh) | pull a Hub dataset (`ashraq/esc50`) → partition → train (config-driven) | network, GPU |
| [`03_soundscape_analysis.sh`](03_soundscape_analysis.sh) | segment → acoustic indices → event detection → AST predict → annotations | a trained/published model |
| [`04_embedding_clustering.sh`](04_embedding_clustering.sh) | embed → dimensionality reduction → cluster → novelty | a model |
| [`05_grab_and_go.sh`](05_grab_and_go.sh) | train AST straight off a Hub dataset id (no local steps) | network, GPU |

## Conventions

- Each script uses `set -euo pipefail` and writes outputs under `./out/<workflow>/`.
- Steps a human must do by hand (reviewing/correcting annotations) are marked
  **`# >>> MANUAL`** — the script pauses or expects you to edit a file before
  continuing.
- Set the variables at the top of each script (species, model id, HF repo) before
  running.

## Prerequisites

```bash
make install                      # full runtime stack + tooling
huggingface-cli login             # for pulling/pushing private repos and publishing
export XC_API_KEY=...             # Xeno-canto (catalog downloads)
export EBIRD_API_KEY=...          # eBird (range validation)
```
