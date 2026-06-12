#!/usr/bin/env bash
#
# Workflow 6: iNaturalist clip → AST inference
# --------------------------------------------
# The quickest end-to-end taste of bioamla: grab a real recording from
# iNaturalist and classify it with a ready-made model off the Hub
# (bioamla/ast-esc50, fine-tuned on the 50-class ESC-50 environmental-sound set).
# No training, no annotation, no API key — just download and predict.
#
# Needs: network (iNaturalist download + model download). No GPU required for a
# single clip; runs fine on CPU.

set -euo pipefail

OUT=./out/06_inat_clip_inference
mkdir -p "$OUT"

# ESC-50 covers a handful of animal classes (frog, dog, crow, insects, ...).
# Frogs are well represented on iNaturalist, so they make a clean demo.
TAXON="Lithobates catesbeianus"   # American bullfrog (an ESC-50 "frog")
MODEL="bioamla/ast-esc50"

# (Optional) See what classes the model can predict before you run it.
bioamla models ast info "$MODEL"

# 1. Download a single research-grade audio observation from iNaturalist. Files
#    land under $OUT/clips/<taxon>/ (iNat audio is typically .mp3); a clip with
#    a permissive license keeps the result reusable.
bioamla catalogs inat download "$OUT/clips" \
  --taxon-name "$TAXON" \
  --quality-grade research \
  --license cc0,cc-by,cc-by-nc \
  --obs-per-taxon 1

# 2. Grab the first downloaded clip.
CLIP="$(find "$OUT/clips" -type f \( -name '*.mp3' -o -name '*.wav' -o -name '*.m4a' \) | head -n 1)"
if [[ -z "$CLIP" ]]; then
  echo "No audio downloaded — try a different TAXON or relax the license filter." >&2
  exit 1
fi
echo "Classifying: $CLIP"

# 3. Run AST inference. predict resamples to the model's expected rate on its own
#    and prints the predicted class + confidence.
bioamla models ast predict "$CLIP" --model-path "$MODEL"

echo "Done."
