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
#   - ML-based species classification
#   - Soundscape ecological assessment
#   - eBird observation integration
#   - Results export and visualization
#
# This workflow combines multiple bioamla features into a realistic research
# scenario for biodiversity monitoring studies.
#
# INPUT: Directory of raw field recordings from acoustic recorders
# OUTPUT: Species detections, acoustic indices, visualizations, reports
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_NAME="wetland_survey_2024"
RAW_DATA="./raw_field_recordings"
EBIRD_API_KEY="${EBIRD_API_KEY:-your_api_key_here}"
SURVEY_LAT=40.7128
SURVEY_LNG=-74.0060

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

# Step 2: Explore and inventory raw data
echo ""
echo "Step 2: Exploring raw dataset..."
bioamla explore "$RAW_DATA"
bioamla audio list "$RAW_DATA"

# Step 3: Preprocess audio files
echo ""
echo "Step 3: Preprocessing audio..."

# Convert to standard format
bioamla audio convert "$RAW_DATA" wav \
    --output "./processed/converted"

# Resample to consistent rate
bioamla audio resample "./processed/converted" \
    --output "./processed/resampled" \
    --rate 22050 \
    --batch

# Apply bandpass filter for bioacoustic frequencies
bioamla audio filter "./processed/resampled" \
    --output "./processed/filtered" \
    --bandpass \
    --low 300 \
    --high 10000 \
    --batch

# Denoise recordings
bioamla audio denoise "./processed/filtered" \
    --output "./processed/clean" \
    --method spectral \
    --strength 0.3 \
    --batch

# Normalize audio levels
bioamla audio normalize "./processed/clean" \
    --output "./processed/normalized" \
    --target-db -20 \
    --batch

# Step 4: Compute acoustic indices for soundscape assessment
echo ""
echo "Step 4: Computing acoustic indices..."
bioamla indices compute "./processed/normalized" \
    --output "./results/acoustic_indices.csv" \
    --output-format csv

# Temporal analysis of indices
bioamla indices temporal "./processed/normalized" \
    --window 60.0 \
    --hop 30.0 \
    --output "./results/temporal_indices.csv" \
    --output-format csv

# Step 5: Run species detection algorithms
echo ""
echo "Step 5: Running acoustic detectors..."

# Energy-based detection for general vocalizations
bioamla detect energy "./processed/normalized" \
    --low-freq 500 \
    --high-freq 8000 \
    --threshold 0.4 \
    --min-duration 0.1 \
    --output "./results/energy_detections.csv"

# RIBBIT detection for frog calls
bioamla detect ribbit "./processed/normalized" \
    --pulse-rate 8 \
    --tolerance 0.25 \
    --low-freq 500 \
    --high-freq 4000 \
    --output "./results/frog_detections.csv"

# Peak sequence detection
bioamla detect peaks "./processed/normalized" \
    --snr 8 \
    --low-freq 1000 \
    --high-freq 6000 \
    --output "./results/peak_detections.csv"

# Step 6: ML-based species classification
echo ""
echo "Step 6: Running ML classification..."

# Segment audio for classification
bioamla audio segment "./processed/normalized" \
    --output "./processed/segments" \
    --silence-threshold -35 \
    --min-silence 0.5 \
    --min-segment 1.0

# Run AST model prediction
bioamla models predict ast "./processed/segments" \
    --batch \
    --model-path "MIT/ast-finetuned-audioset-10-10-0.4593" \
    --output "./results/ml_predictions.csv" \
    --top-k 5 \
    --threshold 0.2

# Extract embeddings for clustering analysis
bioamla models embed "./processed/segments" \
    --model-type ast \
    --model-path "MIT/ast-finetuned-audioset-10-10-0.4593" \
    --output "./results/embeddings.npy" \
    --batch

# Step 7: Clustering for unknown sound discovery
echo ""
echo "Step 7: Clustering for sound discovery..."

# Reduce dimensionality
bioamla cluster reduce \
    --embeddings "./results/embeddings.npy" \
    --output "./results/umap_embeddings.npy" \
    --method umap \
    --n-components 2

# Cluster sounds
bioamla cluster cluster \
    --embeddings "./results/umap_embeddings.npy" \
    --output "./results/cluster_labels.npy" \
    --method hdbscan \
    --min-cluster-size 5

# Detect novel sounds
bioamla cluster novelty \
    --embeddings "./results/embeddings.npy" \
    --output "./results/novelty_scores.npy" \
    --method isolation_forest \
    --contamination 0.05

# Step 8: Cross-reference with eBird observations
echo ""
echo "Step 8: Checking eBird observations..."
if [ "$EBIRD_API_KEY" != "your_api_key_here" ]; then
    bioamla services ebird nearby \
        --lat $SURVEY_LAT \
        --lng $SURVEY_LNG \
        --api-key "$EBIRD_API_KEY" \
        --distance 10 \
        --back 14 \
        --output-format json > "./results/ebird_observations.json"
else
    echo "   (Skipped - set EBIRD_API_KEY environment variable)"
fi

# Step 9: Generate visualizations
echo ""
echo "Step 9: Generating visualizations..."

mkdir -p "./results/spectrograms"

# Generate spectrograms for detected segments
for segment in ./processed/segments/*.wav; do
    if [ -f "$segment" ]; then
        filename=$(basename "$segment" .wav)
        bioamla audio visualize "$segment" \
            --output "./results/spectrograms/${filename}.png" \
            --type mel \
            --colormap viridis \
            --db-scale
    fi
done 2>/dev/null || echo "   (No segments to visualize)"

# Step 10: Generate summary reports
echo ""
echo "Step 10: Generating reports..."

# Prediction summary
bioamla annotation summary \
    --path "./results/ml_predictions.csv" \
    --file-format csv \
    --output-json "./results/prediction_summary.json"

# Analysis report
bioamla audio analyze "./processed/normalized" \
    --batch \
    --output "./results/audio_analysis.csv" \
    --output-format csv \
    --recursive

# Step 11: Export to external formats
echo ""
echo "Step 11: Exporting results..."

# Convert to Raven format for review
bioamla annotation convert \
    --input "./results/ml_predictions.csv" \
    --output "./results/predictions_raven.txt" \
    --from-format csv \
    --to-format raven

# Step 12: View command history
echo ""
echo "Step 12: Saving command history..."
bioamla log show --limit 100 > "./results/command_history.txt"
bioamla log stats

echo ""
echo "=============================================="
echo "  Survey Processing Complete!"
echo "=============================================="
echo ""
echo "Results directory structure:"
echo "  ./processed/"
echo "    ├── converted/     - WAV format files"
echo "    ├── resampled/     - Consistent sample rate"
echo "    ├── filtered/      - Bandpass filtered"
echo "    ├── clean/         - Denoised"
echo "    ├── normalized/    - Level normalized"
echo "    └── segments/      - Detected segments"
echo ""
echo "  ./results/"
echo "    ├── acoustic_indices.csv    - Soundscape metrics"
echo "    ├── temporal_indices.csv    - Time-series indices"
echo "    ├── energy_detections.csv   - Energy detector results"
echo "    ├── frog_detections.csv     - RIBBIT detector results"
echo "    ├── peak_detections.csv     - Peak detector results"
echo "    ├── ml_predictions.csv      - ML classifications"
echo "    ├── embeddings.npy          - Audio embeddings"
echo "    ├── cluster_labels.npy      - Sound clusters"
echo "    ├── novelty_scores.npy      - Novelty detection"
echo "    ├── spectrograms/           - Visual spectrograms"
echo "    ├── prediction_summary.json - Classification stats"
echo "    ├── audio_analysis.csv      - File analysis"
echo "    └── predictions_raven.txt   - Raven-compatible"
echo ""
echo "Next steps:"
echo "  1. Review high-confidence predictions in ml_predictions.csv"
echo "  2. Check novelty_scores.npy for unusual sounds to investigate"
echo "  3. Use active learning to refine classifications"
echo "  4. Compare acoustic indices across sites/times"
