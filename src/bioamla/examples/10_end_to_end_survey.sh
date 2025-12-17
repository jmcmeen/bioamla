#!/bin/bash
# =============================================================================
# End-to-End Wildlife Survey Workflow
# =============================================================================
# PURPOSE: Complete workflow for processing field recordings from a wildlife
#          acoustic survey, from raw data to species inventory and reports.
#
# FEATURES DEMONSTRATED:
#   - Project initialization and management
#   - Complete audio preprocessing pipeline
#   - Multi-detector species detection
#   - ML-based species classification with HuggingFace models
#   - Soundscape ecological assessment
#   - Results export and visualization
#
# AVAILABLE MODELS (HuggingFace):
#   - MIT/ast-finetuned-audioset-10-10-0.4593: General audio (birds, frogs, etc.)
#   - bioamla/ast-esc50: Environmental sounds
#   - bioamla/scp-frogs: Frog species classifier
#
# AVAILABLE DATASETS (HuggingFace):
#   - bioamla/scp-frogs-small: Test frog recordings
#   - ashraq/esc50: Environmental sound samples
#
# INPUT: Directory of raw field recordings from acoustic recorders
# OUTPUT: Species detections, acoustic indices, visualizations, reports
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_NAME="wetland_survey_2024"
RAW_DATA="${1:-./raw_field_recordings}"

echo "=============================================="
echo "  End-to-End Wildlife Survey Workflow"
echo "=============================================="
echo ""

# Step 1: Initialize bioamla project
echo "Step 1: Initializing project..."
bioamla project init \
    --name "$PROJECT_NAME" \
    --description "Wetland acoustic survey - biodiversity assessment" \
    --template research

# Verify project status
bioamla project status

# Step 2: Check system and list audio files
echo ""
echo "Step 2: System check and audio inventory..."
bioamla version
bioamla devices
bioamla audio list "$RAW_DATA" 2>/dev/null || echo "Note: Place recordings in $RAW_DATA"

# Step 3: Preprocess audio files
echo ""
echo "Step 3: Preprocessing audio..."

mkdir -p "./processed"

# Convert to standard format
bioamla audio convert "$RAW_DATA" wav \
    --output "./processed/converted" 2>/dev/null || echo "Skipping conversion (no files)"

# Resample to consistent rate (if files exist)
if [ -d "./processed/converted" ] && [ "$(ls -A ./processed/converted 2>/dev/null)" ]; then
    bioamla audio resample "./processed/converted" \
        --output "./processed/resampled" \
        --rate 16000 \
        --batch
fi

# Step 4: Compute acoustic indices for soundscape assessment
echo ""
echo "Step 4: Computing acoustic indices..."
mkdir -p "./results"

if [ -d "./processed/resampled" ] && [ "$(ls -A ./processed/resampled 2>/dev/null)" ]; then
    bioamla indices compute "./processed/resampled" \
        --output "./results/acoustic_indices.csv" \
        --format csv
fi

# Step 5: Run species detection algorithms
echo ""
echo "Step 5: Running acoustic detectors..."

if [ -d "./processed/resampled" ] && [ "$(ls -A ./processed/resampled 2>/dev/null)" ]; then
    # Energy-based detection for general vocalizations
    bioamla detect energy "./processed/resampled" \
        --low-freq 500 \
        --high-freq 8000 \
        --threshold 0.4 \
        --min-duration 0.1 \
        --output "./results/energy_detections.csv"

    # RIBBIT detection for frog calls
    bioamla detect ribbit "./processed/resampled" \
        --pulse-rate 8 \
        --tolerance 0.25 \
        --low-freq 500 \
        --high-freq 4000 \
        --output "./results/frog_detections.csv"
fi

# Step 6: ML-based species classification using HuggingFace models
echo ""
echo "Step 6: Running ML classification with HuggingFace models..."

if [ -d "./processed/resampled" ] && [ "$(ls -A ./processed/resampled 2>/dev/null)" ]; then
    # General audio classification (AudioSet - 527 classes including birds, frogs)
    echo "  - AudioSet model (general classification)..."
    bioamla models predict ast "./processed/resampled" \
        --batch \
        --model-path "MIT/ast-finetuned-audioset-10-10-0.4593" \
        --output "./results/audioset_predictions.csv" \
        --top-k 5 \
        --threshold 0.1

    # ESC-50 environmental sounds
    echo "  - ESC-50 model (environmental sounds)..."
    bioamla models predict ast "./processed/resampled" \
        --batch \
        --model-path "bioamla/ast-esc50" \
        --output "./results/esc50_predictions.csv" \
        --top-k 5 \
        --threshold 0.1

    # Frog species classification
    echo "  - Frog species model..."
    bioamla models predict ast "./processed/resampled" \
        --batch \
        --model-path "bioamla/scp-frogs" \
        --output "./results/frog_predictions.csv" \
        --top-k 5 \
        --threshold 0.1

    # Extract embeddings for clustering analysis
    echo "  - Extracting embeddings..."
    bioamla models embed "./processed/resampled" \
        --model-type ast \
        --model-path "MIT/ast-finetuned-audioset-10-10-0.4593" \
        --output "./results/embeddings.npy" \
        --batch
fi

# Step 7: Clustering for unknown sound discovery
echo ""
echo "Step 7: Clustering for sound discovery..."

if [ -f "./results/embeddings.npy" ]; then
    # Reduce dimensionality
    bioamla cluster reduce \
        "./results/embeddings.npy" \
        --output "./results/umap_embeddings.npy" \
        --method umap \
        --n-components 2

    # Cluster sounds
    bioamla cluster cluster \
        "./results/umap_embeddings.npy" \
        --output "./results/cluster_labels.npy" \
        --method hdbscan \
        --min-cluster-size 5

    # Detect novel sounds
    bioamla cluster novelty \
        "./results/embeddings.npy" \
        --output "./results/novelty_scores.npy" \
        --method isolation_forest \
        --contamination 0.05
fi

# Step 8: Generate summary reports
echo ""
echo "Step 8: Generating reports..."

if [ -f "./results/audioset_predictions.csv" ]; then
    bioamla annotation summary \
        --path "./results/audioset_predictions.csv" \
        --file-format csv \
        --output-json "./results/prediction_summary.json"

    # Convert to Raven format for review
    bioamla annotation convert \
        "./results/audioset_predictions.csv" \
        "./results/predictions_raven.txt" \
        --from csv \
        --to raven
fi

# Step 9: View command history
echo ""
echo "Step 9: Saving command history..."
bioamla log show --limit 100 > "./results/command_history.txt" 2>/dev/null || true
bioamla log stats 2>/dev/null || true

echo ""
echo "=============================================="
echo "  Survey Processing Complete!"
echo "=============================================="
echo ""
echo "Results directory structure:"
echo "  ./processed/"
echo "    ├── converted/     - WAV format files"
echo "    └── resampled/     - Consistent sample rate"
echo ""
echo "  ./results/"
echo "    ├── acoustic_indices.csv    - Soundscape metrics"
echo "    ├── energy_detections.csv   - Energy detector results"
echo "    ├── frog_detections.csv     - RIBBIT detector results"
echo "    ├── audioset_predictions.csv - AudioSet model predictions"
echo "    ├── esc50_predictions.csv   - ESC-50 model predictions"
echo "    ├── frog_predictions.csv    - Frog model predictions"
echo "    ├── embeddings.npy          - Audio embeddings"
echo "    ├── cluster_labels.npy      - Sound clusters"
echo "    ├── novelty_scores.npy      - Novelty detection"
echo "    └── predictions_raven.txt   - Raven-compatible format"
echo ""
echo "Models used (all from HuggingFace):"
echo "  - MIT/ast-finetuned-audioset-10-10-0.4593"
echo "  - bioamla/ast-esc50"
echo "  - bioamla/scp-frogs"
echo ""
echo "Next steps:"
echo "  1. Review predictions in results/*.csv"
echo "  2. Check novelty_scores.npy for unusual sounds"
echo "  3. Train custom model on your annotated data"
echo "     See: 07_model_training.sh"
