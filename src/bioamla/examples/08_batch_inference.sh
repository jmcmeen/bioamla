#!/bin/bash
# =============================================================================
# Batch Inference and Prediction Workflow
# =============================================================================
# PURPOSE: Run large-scale species classification on audio datasets using
#          trained models. Process thousands of recordings efficiently.
#
# FEATURES DEMONSTRATED:
#   - Batch prediction with AST models
#   - Using pre-trained HuggingFace models
#   - Multiple output formats (CSV, JSON)
#   - Embedding extraction for downstream analysis
#   - Annotation format conversion
#
# AVAILABLE MODELS (HuggingFace):
#   - MIT/ast-finetuned-audioset-10-10-0.4593: General audio (527 classes)
#   - bioamla/ast-esc50: Environmental sounds (50 classes)
#   - bioamla/scp-frogs: Frog species classifier
#
# INPUT: Directory of audio files to classify
# OUTPUT: Predictions with confidence scores in CSV/JSON format
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_NAME="frog_acoustic_study"
PROJECT_DIR="./${PROJECT_NAME}"
AUDIO_DIR="${PROJECT_DIR}/raw_recordings"
OUTPUT_SUBDIR="predictions"
OUTPUT_DIR="${PROJECT_DIR}/$OUTPUT_SUBDIR"

# Choose your model based on use case:
# General audio classification (birds, frogs, music, speech, etc.)
MODEL_AUDIOSET="MIT/ast-finetuned-audioset-10-10-0.4593"
# Environmental sounds (50 categories)
MODEL_ESC50="bioamla/ast-esc50"
# Frog species identification
MODEL_FROGS="bioamla/scp-frogs"

echo "=== Batch Inference Workflow ==="
echo "Input directory: $AUDIO_DIR"
echo ""

# Note: The CLI creates the output directory automatically, but we create it
# here for commands that don't (like embed, annotation convert, etc.)
mkdir -p "$OUTPUT_DIR"

# Step 1: Quick audio file inventory
echo "Step 1: Listing audio files..."
bioamla audio list "$AUDIO_DIR"

# Step 2: Run batch inference with AudioSet model (general classification)
echo ""
echo "Step 2: Running batch inference with AudioSet model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "$MODEL_AUDIOSET" \
    --output-csv "$OUTPUT_SUBDIR/audioset_predictions.csv" \
    --segment-duration 5 \
    --segment-overlap 1

# Step 3: Run inference with ESC-50 model (environmental sounds)
echo ""
echo "Step 3: Running inference with ESC-50 model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "$MODEL_ESC50" \
    --output-csv "$OUTPUT_SUBDIR/esc50_predictions.csv" \
    --segment-duration 5 \
    --segment-overlap 1

# Step 4: Run inference with frog species model
echo ""
echo "Step 4: Running inference with frog species model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "$MODEL_FROGS" \
    --output-csv "$OUTPUT_SUBDIR/frog_predictions.csv" \
    --segment-duration 5 \
    --segment-overlap 1

# Step 5: Extract embeddings for clustering analysis
echo ""
echo "Step 5: Extracting embeddings..."
bioamla models embed "$AUDIO_DIR" \
    --model-type ast \
    --model-path "$MODEL_AUDIOSET" \
    --output "$OUTPUT_DIR/embeddings.npy" \
    --batch

echo ""
echo "=== Batch Inference Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
echo "  - audioset_predictions.csv: General audio classification (527 classes)"
echo "  - esc50_predictions.csv: Environmental sound classification (50 classes)"
echo "  - frog_predictions.csv: Frog species classification"
echo "  - embeddings.npy: Audio embeddings for clustering"
echo ""
echo "CSV output columns:"
echo "  - filepath: Path to audio file"
echo "  - start, stop: Segment boundaries (in samples)"
echo "  - prediction: Top prediction label"
echo ""
echo "Additional analysis:"
echo "  - Cluster embeddings: bioamla cluster reduce --embeddings embeddings.npy ..."
echo "  - Novelty detection: bioamla cluster novelty --embeddings embeddings.npy ..."
