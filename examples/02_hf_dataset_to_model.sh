#!/usr/bin/env bash
#
# Workflow 2: HuggingFace dataset → train (config-driven)
# -------------------------------------------------------
# The fastest path to a model: pull an existing labeled audio dataset from the
# Hub, materialize it into bioamla's labeled-folder layout, and fine-tune AST
# using a TOML config file (so the run is reproducible and easy to tweak).
#
# Needs: network (dataset + base-model download) and a GPU for training.

set -euo pipefail

OUT=./out/02_hf_dataset_to_model
mkdir -p "$OUT"

DATASET="ashraq/esc-50"   # 50-class environmental sound dataset
HF_USER="your-username"   # <-- set me (only needed for the optional push)

# 1. Pull the dataset and materialize it as label subdirs + metadata.csv
#    (auto-detects the audio + label columns; resamples to 16 kHz for AST).
bioamla catalogs hf pull-dataset "$DATASET" "$OUT/data" --split train

# 2. Inspect and split. The pulled layout drops straight into the dataset tools.
bioamla dataset stats "$OUT/data" --json > "$OUT/stats.json"
bioamla dataset partition "$OUT/data" --train 0.8 --val 0.1 --test 0.1 --seed 0

# 3. Write a training config (flags still override file values at run time).
cat > "$OUT/train.toml" <<'TOML'
[models]
default_ast_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

[training]
learning_rate = 5e-5
epochs = 15
batch_size = 16
eval_strategy = "epoch"
save_strategy = "epoch"
TOML

# 4. Fine-tune from the config. Override a flag ad hoc to show precedence
#    (here --fp16 isn't in the file; --num-train-epochs would override it).
bioamla models ast train --train-dataset "$OUT/data" --training-dir "$OUT/train" \
  --config "$OUT/train.toml" --fp16

# 5. (Optional) publish the fine-tuned model.
# bioamla catalogs hf push-model "$OUT/train/best_model" "$HF_USER/esc50-ast"

echo "Done. Best model at $OUT/train/best_model"
