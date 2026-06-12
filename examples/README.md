# bioamla example workflows

End-to-end bioacoustics workflows wired together from the `bioamla` CLI. Each
script is heavily commented and meant to be **read first, then run** — they are
templates to adapt, not fixtures. Unlike [`dev-data/cli_test.sh`](../dev-data/cli_test.sh)
(an offline smoke test), these touch the network, the HuggingFace Hub, and (for
training) a GPU.

| Script | Flow | Needs |
| --- | --- | --- |
| [`01_catalog_to_model.sh`](01_catalog_to_model.sh) | catalog download → annotate → dataset → train → publish | `XC_API_KEY`/`EBIRD_API_KEY`, HF login, GPU |
| [`02_hf_dataset_train_ast.sh`](02_hf_dataset_train_ast.sh) | grab and go: train AST straight off a Hub dataset id (no local steps) | network, GPU |
| [`03_hf_dataset_curate_and_train.sh`](03_hf_dataset_curate_and_train.sh) | pull a Hub dataset (`ashraq/esc50`) → materialize → inspect → partition → train (config-driven) | network, GPU |
| [`04_soundscape_analysis.sh`](04_soundscape_analysis.sh) | acoustic indices → event detection → segment → AST predict → annotations | a trained/published model |
| [`05_embedding_clustering.sh`](05_embedding_clustering.sh) | embed → dimensionality reduction → cluster → novelty | a model |
| [`06_inat_clip_inference.sh`](06_inat_clip_inference.sh) | iNaturalist clip download → AST inference with `bioamla/ast-esc50` | network |

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
