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
AUDIO_DIR="${1:-./raw_recordings}"
OUTPUT_DIR="./predictions"

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
    --output-csv "$OUTPUT_DIR/audioset_predictions.csv" \
    --clip-seconds 5 \
    --overlap-seconds 1

# Step 3: Run inference with ESC-50 model (environmental sounds)
echo ""
echo "Step 3: Running inference with ESC-50 model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "$MODEL_ESC50" \
    --output-csv "$OUTPUT_DIR/esc50_predictions.csv"

# Step 4: Run inference with frog species model
echo ""
echo "Step 4: Running inference with frog species model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "$MODEL_FROGS" \
    --output-csv "$OUTPUT_DIR/frog_predictions.csv"

# Step 5: Extract embeddings for clustering analysis
echo ""
echo "Step 5: Extracting embeddings..."
bioamla models embed "$AUDIO_DIR" \
    --model-type ast \
    --model-path "$MODEL_AUDIOSET" \
    --output "$OUTPUT_DIR/embeddings.npy" \
    --batch

# Step 6: Convert annotation format (for use with Raven Pro)
echo ""
echo "Step 6: Converting prediction format to Raven..."
bioamla annotation convert \
    "$OUTPUT_DIR/audioset_predictions.csv" \
    "$OUTPUT_DIR/predictions_raven.txt" \
    --from csv \
    --to raven

# Step 7: Generate prediction summary
echo ""
echo "Step 7: Generating prediction summary..."
bioamla annotation summary \
    --path "$OUTPUT_DIR/audioset_predictions.csv" \
    --file-format csv \
    --output-json "$OUTPUT_DIR/prediction_summary.json"

echo ""
echo "=== Batch Inference Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
echo "  - audioset_predictions.csv: General audio classification (527 classes)"
echo "  - esc50_predictions.csv: Environmental sound classification (50 classes)"
echo "  - frog_predictions.csv: Frog species classification"
echo "  - embeddings.npy: Audio embeddings for clustering"
echo "  - predictions_raven.txt: Raven-compatible format"
echo "  - prediction_summary.json: Summary statistics"
echo ""
echo "CSV output columns:"
echo "  - file_path: Path to audio file"
echo "  - start_time, end_time: Segment boundaries"
echo "  - predicted_label: Top prediction"
echo "  - confidence: Prediction confidence (0-1)"
echo "  - top_k_labels: Top-k predictions with scores"
echo ""
echo "Additional analysis:"
echo "  - Cluster embeddings: bioamla cluster reduce --embeddings embeddings.npy ..."
echo "  - Novelty detection: bioamla cluster novelty --embeddings embeddings.npy ..."
