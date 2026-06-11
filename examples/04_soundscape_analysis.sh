#!/usr/bin/env bash
#
# Workflow 4: soundscape analysis
# -------------------------------
# Characterize a long field recording: chop it into clips, compute acoustic
# indices, detect events, classify with a trained AST model, and turn the
# predictions into an annotation file you can review.
#
# Needs: a recording (set RECORDING) and a model (set MODEL — a Hub id or local
# path). The model download/inference is the only heavy step.

set -euo pipefail

OUT=./out/03_soundscape_analysis
mkdir -p "$OUT"/{segments,indices,detections}

RECORDING=./out/03_soundscape_analysis/soundscape.wav   # <-- point at your file
MODEL="your-username/frog-ast"                          # <-- trained model id/path

if [[ ! -f "$RECORDING" ]]; then
  echo "Set RECORDING to a real audio file (e.g. a dawn-chorus recording)." >&2
  exit 1
fi

# 1. Acoustic indices over time (soundscape ecology summary metrics).
bioamla indices compute "$RECORDING" -o "$OUT/indices/summary.json"
bioamla indices temporal "$RECORDING" --segment-duration 60 -o "$OUT/indices/timeline.csv"

# 2. Event detection — find the energetic/structured regions worth looking at.
bioamla detect energy "$RECORDING" -o "$OUT/detections/energy.csv" --low-freq 500 --high-freq 10000
bioamla detect ribbit "$RECORDING" -o "$OUT/detections/ribbit.csv"

# 3. Split into fixed clips and classify each with the model (batch, directory
#    mode → one prediction row per file merged into a CSV).
bioamla audio segment "$RECORDING" "$OUT/segments/" -d 3.0 -o 0.5
bioamla batch models predict --input-dir "$OUT/segments" --output-dir "$OUT/predictions" \
  --model "$MODEL" --min-confidence 0.5

# 4. Turn segment-level predictions into an editable annotation file. Drop a
#    background class if your model has one, and keep only confident calls.
bioamla models ast annotate "$RECORDING" -o "$OUT/predicted.csv" \
  --model-path "$MODEL" --segment-duration 3 --min-confidence 0.6

# >>> MANUAL: review $OUT/predicted.csv, correct mislabeled/spurious rows, then
# >>> feed it back into `dataset extract-clips --annotations` to grow your
# >>> training set (active learning loop) — see workflow 01.
echo "Done. Review $OUT/predicted.csv and the indices/detections under $OUT/."
