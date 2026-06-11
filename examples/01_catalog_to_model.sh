#!/usr/bin/env bash
#
# Workflow 1: catalog → annotate → dataset → train → publish
# ----------------------------------------------------------
# Build a species classifier from scratch: source recordings from a public
# catalog, review/annotate them, cut a labeled dataset, fine-tune AST, and push
# both the dataset and the model to the HuggingFace Hub.
#
# Needs: XC_API_KEY (Xeno-canto), `huggingface-cli login`, and a GPU for training.

set -euo pipefail

OUT=./out/01_catalog_to_model
mkdir -p "$OUT"

# Two target species (edit these). Each becomes a class label.
SPECIES_A="Lithobates catesbeianus"   # American bullfrog
SPECIES_B="Hyla cinerea"              # Green treefrog
HF_USER="your-username"               # <-- set me
MODEL_REPO="$HF_USER/frog-ast"
DATASET_REPO="$HF_USER/frog-calls"

# 1. Download recordings from Xeno-canto into per-species folders (each download
#    writes a metadata.csv carrying license/attribution for traceability).
bioamla catalogs xc download --species "$SPECIES_A" --quality A --max-recordings 25 \
  --output-dir "$OUT/raw/bullfrog"
bioamla catalogs xc download --species "$SPECIES_B" --quality A --max-recordings 25 \
  --output-dir "$OUT/raw/treefrog"

# 2. Seed annotations automatically, then correct them by hand. If you already
#    have a rough model, `models ast predict --segment-duration N -o preds.csv`
#    gives you a per-segment prediction CSV to review; otherwise start from an
#    empty template per file.
for f in "$OUT"/raw/bullfrog/*.wav; do
  bioamla annotation template "$f" "${f%.wav}.json" --label bullfrog
done
for f in "$OUT"/raw/treefrog/*.wav; do
  bioamla annotation template "$f" "${f%.wav}.json" --label treefrog
done
# >>> MANUAL: open the .json annotations next to each recording, adjust the time
# >>> bounds/labels to the actual calls, then continue. (Raven/Audacity export
# >>> to CSV/Raven also works — see `bioamla annotation convert`.)
read -r -p "Press Enter once annotations are reviewed... "

# 3. Cut annotated regions into per-species labeled clip datasets (label subdirs
#    + metadata.csv), resampled to 16 kHz for AST. Provenance from the catalog
#    metadata.csv is joined onto each clip automatically. Then merge into one
#    dataset (merge combines + de-duplicates metadata across sources).
bioamla dataset extract-clips "$OUT/raw/bullfrog" "$OUT/clips/bullfrog" \
  --layout both --sample-rate 16000
bioamla dataset extract-clips "$OUT/raw/treefrog" "$OUT/clips/treefrog" \
  --layout both --sample-rate 16000
bioamla dataset merge "$OUT/dataset" "$OUT/clips/bullfrog" "$OUT/clips/treefrog"

# 4. Split, summarize, and record attribution for the combined dataset.
bioamla dataset partition "$OUT/dataset" --train 0.8 --val 0.1 --test 0.1 --seed 0
bioamla dataset stats "$OUT/dataset" --json > "$OUT/dataset_stats.json"
bioamla dataset license "$OUT/dataset" -o "$OUT/ATTRIBUTION.md"

# 5. Fine-tune AST on the labeled folder.
bioamla models ast train --train-dataset "$OUT/dataset" --training-dir "$OUT/train" \
  --num-train-epochs 10 --per-device-train-batch-size 8 --fp16

# 6. Publish dataset and model to the Hub (a dataset card is generated from the
#    manifest/metadata before pushing).
bioamla catalogs hf push-dataset "$OUT/dataset" "$DATASET_REPO"
bioamla catalogs hf push-model "$OUT/train/best_model" "$MODEL_REPO"

echo "Done. Model: https://huggingface.co/$MODEL_REPO"
