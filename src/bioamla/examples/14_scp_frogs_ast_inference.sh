#!/bin/bash
# =============================================================================
# SCP Frogs Audio Spectrogram Transformer Inference
# =============================================================================
# PURPOSE: Run inference using an AST model trained on frog vocalizations
#          to identify frog species from audio recordings.
#
# MODELS AVAILABLE:
#   - bioamla/scp-frogs: Pre-trained frog species classifier (HuggingFace)
#   - ./scp_frogs_model/best_model: Locally trained model
#
# TEST DATA:
#   - bioamla/scp-frogs-small: Small test dataset for validation
#
# USE CASES:
#   - Frog species identification from field recordings
#   - Passive acoustic monitoring of amphibian populations
#   - Biodiversity surveys in wetland habitats
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
INPUT_DIR="${PROJECT_DIR}/raw_recordings"
MODEL_PATH="${1:-bioamla/scp-frogs}"
OUTPUT_DIR="${PROJECT_DIR}/frog_predictions"

echo "=== SCP Frogs AST Inference Workflow ==="
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
echo "Other frog/amphibian detection options:"
echo "  - Use RIBBIT detector for periodic frog calls:"
echo "    bioamla detect ribbit <audio> --pulse-rate 10 --low-freq 500 --high-freq 4000"
echo ""
echo "  - Use energy detector for general vocalization detection:"
echo "    bioamla detect energy <audio> --low-freq 500 --high-freq 3000"
