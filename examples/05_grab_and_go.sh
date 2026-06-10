#!/usr/bin/env bash
#
# Workflow 5: grab and go
# -----------------------
# The shortest path from a Hub dataset to a fine-tuned AST model — no pull, no
# partition, no local files. `models ast train` loads the dataset straight off
# the Hub by id and does its own train/test split. Set it up nicely on the Hub
# once, then just grab and go.
#
# Needs: network (dataset + base-model download) and a GPU for training.

set -euo pipefail

OUT=./out/05_grab_and_go
mkdir -p "$OUT"

DATASET="ashraq/esc50"   # any labeled HF audio dataset (audio + label columns)

# Train directly off the Hub id.
bioamla models ast train --train-dataset "$DATASET" --training-dir "$OUT/train" \
  --num-train-epochs 10 --per-device-train-batch-size 16 --fp16 --report-to none

# The dataset is cached in HF format for fast repeat runs; reclaim it when done.
bioamla catalogs hf cache --datasets                  # list cached datasets
# bioamla catalogs hf cache --datasets --purge -y     # free the dataset cache

echo "Done. Best model at $OUT/train/best_model"
