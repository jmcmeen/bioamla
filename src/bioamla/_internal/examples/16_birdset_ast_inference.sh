#!/bin/bash
# =============================================================================
# BirdSet Audio Spectrogram Transformer Inference
# =============================================================================
# PURPOSE: Run inference using models trained on BirdSet for bird species
#          identification from audio recordings.
#
# MODELS AVAILABLE:
#   - MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet, includes bird classes)
#   - ./birdset_model_HSN/best_model (locally trained on BirdSet HSN)
#   - DBD-research-group/Bird-MAE-Base (specialized bird encoder)
#
# TEST DATA: samuelstevens/BirdSet (subsets: HSN, NBP, NES, PER)
#
# USE CASES:
#   - Bird species identification from field recordings
#   - Automated bird monitoring stations
#   - Dawn chorus analysis
#   - Migration pattern studies
# =============================================================================

set -e

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
INPUT_DIR="${PROJECT_DIR}/raw_recordings"
MODEL_PATH="${1:-MIT/ast-finetuned-audioset-10-10-0.4593}"
OUTPUT_DIR="${PROJECT_DIR}/bird_predictions"

echo "=== BirdSet AST Inference Workflow ==="
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
    --output-csv "$OUTPUT_DIR/predictions.csv"

echo ""
echo "Predictions saved to: $OUTPUT_DIR/predictions.csv"

echo ""
echo "=== Inference Complete ==="
echo ""
echo "Bird sound analysis options:"
echo ""
echo "  - For general audio (birds in AudioSet classes):"
echo "    bioamla models predict ast <audio> --model-path MIT/ast-finetuned-audioset-10-10-0.4593"
echo ""
echo "  - For specialized bird identification (train your own):"
echo "    See birdset_ast_train.sh"
echo ""
echo "  - For clustering unknown bird sounds:"
echo "    bioamla cluster reduce --embeddings embeddings.npy --output reduced.npy --method umap"
echo "    bioamla cluster cluster --embeddings reduced.npy --output labels.npy --method hdbscan"
