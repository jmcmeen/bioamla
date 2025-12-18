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
# TEST DATA: DBD-research-group/BirdSet (any subset for evaluation)
#
# USE CASES:
#   - Bird species identification from field recordings
#   - Automated bird monitoring stations
#   - Dawn chorus analysis
#   - Migration pattern studies
# =============================================================================

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-./my_project}"
AUDIO_INPUT="${1:-${PROJECT_DIR}/bird_recordings}"
MODEL_PATH="${2:-MIT/ast-finetuned-audioset-10-10-0.4593}"
OUTPUT_DIR="${PROJECT_DIR}/bird_predictions"

echo "=== BirdSet AST Inference Workflow ==="
echo "Input: $AUDIO_INPUT"
echo "Model: $MODEL_PATH"
echo ""

mkdir -p "$OUTPUT_DIR"

# Single file inference
if [ -f "$AUDIO_INPUT" ]; then
    echo "Running single file inference..."
    bioamla models predict ast "$AUDIO_INPUT" \
        --model-path "$MODEL_PATH" \
        --top-k 10

# Batch inference on directory
elif [ -d "$AUDIO_INPUT" ]; then
    echo "Running batch inference on directory..."
    bioamla models predict ast "$AUDIO_INPUT" \
        --batch \
        --model-path "$MODEL_PATH" \
        --output "$OUTPUT_DIR/predictions.csv" \
        --top-k 10 \
        --threshold 0.05 \
        --segment-duration 5.0 \
        --segment-overlap 2.5

    echo ""
    echo "Predictions saved to: $OUTPUT_DIR/predictions.csv"

    # Extract embeddings for further analysis
    echo ""
    echo "Extracting embeddings for clustering analysis..."
    bioamla models embed "$AUDIO_INPUT" \
        --model-type ast \
        --model-path "$MODEL_PATH" \
        --output "$OUTPUT_DIR/embeddings.npy" \
        --batch

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
    echo "  $0 ./dawn_chorus.wav"
    echo "  $0 ./field_recordings/"
    echo "  $0 ./field_recordings/ ./birdset_model_HSN/best_model"
    exit 1
fi

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
