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

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
INPUT_DIR="${PROJECT_DIR}/raw_recordings"
MODEL_PATH="${1:-bioamla/ast-esc50}"  # Use HuggingFace model by default
OUTPUT_DIR="${PROJECT_DIR}/esc50_predictions"

echo "=== ESC-50 AST Inference Workflow ==="
echo "Input: $INPUT_DIR"
echo "Model: $MODEL_PATH"
echo ""

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    echo "Create the directory and add audio files first."
    echo "See 00_starting_a_project.sh to set up a project structure."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running batch inference..."
bioamla models predict ast "$INPUT_DIR" \
    --batch \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/predictions.csv" \
    --top-k 5 \
    --threshold 0.1

echo ""
echo "Predictions saved to: $OUTPUT_DIR/predictions.csv"

echo ""
echo "=== Inference Complete ==="
echo ""
echo "Alternative models you can use:"
echo "  - MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet, 527 classes)"
echo "  - bioamla/ast-esc50 (ESC-50, 50 environmental sound classes)"
echo "  - bioamla/scp-frogs (Frog species classification)"
