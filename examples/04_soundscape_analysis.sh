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

OUT=./out/04_soundscape_analysis
mkdir -p "$OUT"/{indices,detections,predictions}

RECORDING="$OUT/soundscape.wav"   # <-- point at your file
MODEL="your-username/frog-ast"    # <-- trained model id/path

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

# 3. Classify the recording in fixed-length segments — one step, no pre-chopping.
#    `--segment-duration` splits the file internally and writes one prediction row
#    per segment (filepath,start,stop,prediction,confidence).
bioamla models ast predict "$RECORDING" -o "$OUT/predicted.csv" \
  --model-path "$MODEL" --segment-duration 3 --overlap 1 --min-confidence 0.6

# (To classify many recordings at once, the same flags work on the batch command:
#  bioamla batch models predict --input-dir ./recordings --output-dir "$OUT/predictions" \
#    --model "$MODEL" --segment-duration 3 --overlap 1 --min-confidence 0.6)

# >>> MANUAL: review $OUT/predicted.csv, keep/correct the confident calls, then
# >>> feed them back into your training set (active-learning loop) — see workflow 01.
# >>> A future release will fold this predict → review → dataset bridge into an
# >>> "auto-annotate" command so the manual step can be automated.
echo "Done. Review $OUT/predicted.csv and the indices/detections under $OUT/."
