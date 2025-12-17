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
AUDIO_INPUT="${1:-./frog_recordings}"
MODEL_PATH="${2:-bioamla/scp-frogs}"
OUTPUT_DIR="./frog_predictions"

echo "=== SCP Frogs AST Inference Workflow ==="
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
    echo "Running batch inference on directory..."
    bioamla models predict ast "$AUDIO_INPUT" \
        --batch \
        --model-path "$MODEL_PATH" \
        --output "$OUTPUT_DIR/predictions.csv" \
        --top-k 5 \
        --threshold 0.1 \
        --segment-duration 5.0 \
        --segment-overlap 1.0

    echo ""
    echo "Predictions saved to: $OUTPUT_DIR/predictions.csv"

    # Generate summary
    echo ""
    echo "Generating prediction summary..."
    bioamla annotation summary \
        --path "$OUTPUT_DIR/predictions.csv" \
        --file-format csv \
        --output-json "$OUTPUT_DIR/summary.json"
else
    echo "Error: $AUDIO_INPUT is not a valid file or directory"
    echo ""
    echo "Usage: $0 <audio_file_or_directory> [model_path]"
    echo ""
    echo "Examples:"
    echo "  $0 ./recording.wav"
    echo "  $0 ./field_recordings/"
    echo "  $0 ./field_recordings/ bioamla/scp-frogs"
    exit 1
fi

echo ""
echo "=== Inference Complete ==="
echo ""
echo "Other frog/amphibian detection options:"
echo "  - Use RIBBIT detector for periodic frog calls:"
echo "    bioamla detect ribbit <audio> --pulse-rate 10 --low-freq 500 --high-freq 4000"
echo ""
echo "  - Use energy detector for general vocalization detection:"
echo "    bioamla detect energy <audio> --low-freq 500 --high-freq 3000"
