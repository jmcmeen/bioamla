#!/bin/bash
# =============================================================================
# Batch Inference and Prediction Workflow
# =============================================================================
# PURPOSE: Run large-scale species classification on audio datasets using
#          trained models. Process thousands of recordings efficiently.
#
# FEATURES DEMONSTRATED:
#   - Batch prediction with AST models
#   - Generic model inference
#   - Multiple output formats (CSV, JSON)
#   - Embedding extraction for downstream analysis
#   - Multi-model ensemble predictions
#   - Annotation format conversion
#
# INPUT: Directory of audio files to classify
# OUTPUT: Predictions with confidence scores in CSV/JSON format
# =============================================================================

set -e  # Exit on error

# Configuration
AUDIO_DIR="./recordings_to_classify"
MODEL_PATH="./trained_models/ast_model/best_model"
OUTPUT_DIR="./predictions"

echo "=== Batch Inference Workflow ==="
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Quick audio file inventory
echo "Step 1: Listing audio files..."
bioamla audio list "$AUDIO_DIR"

# Step 2: Run batch inference with AST model
echo ""
echo "Step 2: Running batch inference with AST model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/ast_predictions.csv" \
    --top-k 5 \
    --threshold 0.1 \
    --segment-duration 5.0 \
    --segment-overlap 0.5

# Step 3: Run inference with HuggingFace model
echo ""
echo "Step 3: Running inference with pretrained model..."
bioamla models predict ast "$AUDIO_DIR" \
    --batch \
    --model-path "MIT/ast-finetuned-audioset-10-10-0.4593" \
    --output "$OUTPUT_DIR/audioset_predictions.csv" \
    --top-k 3

# Step 4: Generic model prediction
echo ""
echo "Step 4: Running generic model inference..."
bioamla models predict generic "$AUDIO_DIR" \
    --model-type ast \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/generic_predictions.json" \
    --output-format json \
    --batch

# Step 5: Extract embeddings for further analysis
echo ""
echo "Step 5: Extracting embeddings..."
bioamla models embed "$AUDIO_DIR" \
    --model-type ast \
    --model-path "$MODEL_PATH" \
    --output "$OUTPUT_DIR/embeddings.npy" \
    --batch \
    --layer -2

# Step 6: Multi-model ensemble predictions
echo ""
echo "Step 6: Running ensemble predictions..."
# Assumes multiple trained models exist
if [ -d "./trained_models/cnn_model" ]; then
    bioamla models ensemble \
        --model-dirs "$MODEL_PATH" "./trained_models/cnn_model" \
        --output "$OUTPUT_DIR/ensemble_predictions.csv" \
        --strategy weighted \
        --weights 0.7 0.3
else
    echo "   (Skipped - only one model available)"
fi

# Step 7: Convert annotation format
echo ""
echo "Step 7: Converting prediction format..."
bioamla annotation convert \
    --input "$OUTPUT_DIR/ast_predictions.csv" \
    --output "$OUTPUT_DIR/predictions_raven.txt" \
    --from-format csv \
    --to-format raven

# Step 8: Generate prediction summary
echo ""
echo "Step 8: Generating prediction summary..."
bioamla annotation summary \
    --path "$OUTPUT_DIR/ast_predictions.csv" \
    --file-format csv \
    --output-json "$OUTPUT_DIR/prediction_summary.json"

# Step 9: Filter predictions by criteria
echo ""
echo "Step 9: Filtering high-confidence predictions..."
bioamla annotation filter \
    --input "$OUTPUT_DIR/ast_predictions.csv" \
    --output "$OUTPUT_DIR/high_confidence_predictions.csv" \
    --min-duration 0.5

# Step 10: Remap labels to custom categories
echo ""
echo "Step 10: (Example) Remapping labels..."
echo "   bioamla annotation remap \\"
echo "       --input \"$OUTPUT_DIR/ast_predictions.csv\" \\"
echo "       --output \"$OUTPUT_DIR/remapped_predictions.csv\" \\"
echo "       --mapping \"species_a:group_1,species_b:group_1,species_c:group_2\""

echo ""
echo "=== Batch Inference Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
echo "  - ast_predictions.csv: Custom model predictions"
echo "  - audioset_predictions.csv: AudioSet model predictions"
echo "  - generic_predictions.json: JSON format predictions"
echo "  - embeddings.npy: Audio embeddings for clustering"
echo "  - ensemble_predictions.csv: Multi-model ensemble results"
echo "  - predictions_raven.txt: Raven-compatible format"
echo "  - prediction_summary.json: Summary statistics"
echo "  - high_confidence_predictions.csv: Filtered predictions"
echo ""
echo "CSV output columns:"
echo "  - file_path: Path to audio file"
echo "  - start_time, end_time: Segment boundaries"
echo "  - predicted_label: Top prediction"
echo "  - confidence: Prediction confidence (0-1)"
echo "  - top_k_labels: Top-k predictions with scores"
