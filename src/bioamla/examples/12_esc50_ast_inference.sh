#!/bin/bash
# =============================================================================
# ESC-50 Audio Spectrogram Transformer Inference
# =============================================================================
# PURPOSE: Run inference using an AST model trained on ESC-50 environmental
#          sound classification dataset.
#
# MODELS AVAILABLE:
#   - bioamla/ast-esc50: Pre-trained ESC-50 model (HuggingFace)
#   - ./esc50_model/best_model: Locally trained model (from esc50_ast_train.sh)
#
# ESC-50 CLASSES (50 total):
#   Animals: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow
#   Natural: rain, sea_waves, crackling_fire, crickets, chirping_birds,
#            water_drops, wind, pouring_water, toilet_flush, thunderstorm
#   Human: crying_baby, sneezing, clapping, breathing, coughing, footsteps,
#          laughing, brushing_teeth, snoring, drinking_sipping
#   Domestic: door_wood_knock, mouse_click, keyboard_typing, door_wood_creaks,
#             can_opening, washing_machine, vacuum_cleaner, clock_alarm,
#             clock_tick, glass_breaking
#   Urban: helicopter, chainsaw, siren, car_horn, engine, train,
#          church_bells, airplane, fireworks, hand_saw
# =============================================================================

set -e

# Configuration - modify these paths as needed
AUDIO_INPUT="${1:-./test_audio}"  # Directory or file to classify
MODEL_PATH="${2:-bioamla/ast-esc50}"  # Use HuggingFace model by default
OUTPUT_DIR="./esc50_predictions"

echo "=== ESC-50 AST Inference Workflow ==="
echo "Input: $AUDIO_INPUT"
echo "Model: $MODEL_PATH"
echo ""

mkdir -p "$OUTPUT_DIR"

# Single file inference
if [ -f "$AUDIO_INPUT" ]; then
    echo "Running single file inference..."
    bioamla models predict ast "$AUDIO_INPUT" \
        --model-path "$MODEL_PATH" \
        --top-k 5

# Batch inference on directory
elif [ -d "$AUDIO_INPUT" ]; then
    echo "Running batch inference..."
    bioamla models predict ast "$AUDIO_INPUT" \
        --batch \
        --model-path "$MODEL_PATH" \
        --output "$OUTPUT_DIR/predictions.csv" \
        --top-k 5 \
        --threshold 0.1

    echo ""
    echo "Predictions saved to: $OUTPUT_DIR/predictions.csv"
else
    echo "Error: $AUDIO_INPUT is not a valid file or directory"
    exit 1
fi

echo ""
echo "=== Inference Complete ==="
echo ""
echo "Alternative models you can use:"
echo "  - MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet, 527 classes)"
echo "  - bioamla/ast-esc50 (ESC-50, 50 environmental sound classes)"
echo "  - bioamla/scp-frogs (Frog species classification)"
