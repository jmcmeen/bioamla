#!/bin/bash
# =============================================================================
# Acoustic Indices Analysis Workflow
# =============================================================================
# PURPOSE: Compute soundscape ecology metrics to assess biodiversity and
#          habitat quality from acoustic recordings.
#
# FEATURES DEMONSTRATED:
#   - Acoustic Complexity Index (ACI) - measures irregularity in intensity
#   - Acoustic Diversity Index (ADI) - Shannon diversity across frequency bands
#   - Acoustic Evenness Index (AEI) - evenness of energy distribution
#   - Bioacoustic Index (BIO) - area under mean spectrum
#   - Normalized Difference Soundscape Index (NDSI) - biophony vs anthropophony
#   - Temporal entropy analysis
#   - Batch processing with CSV/JSON output
#
# INPUT: Directory of audio recordings from field surveys
# OUTPUT: CSV files with acoustic indices for soundscape analysis
# =============================================================================

set -e  # Exit on error

# Configuration
AUDIO_DIR="${1:-./raw_recordings}"
OUTPUT_DIR="./indices_results"

echo "=== Acoustic Indices Analysis Workflow ==="
echo ""

# Check if input directory exists and has audio files
if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Input directory '$AUDIO_DIR' does not exist."
    echo "Usage: $0 [input_directory]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Step 1: Compute all acoustic indices for the dataset (batch mode)
# This is the recommended approach for processing multiple files
echo "Step 1: Computing all acoustic indices for all files..."
bioamla indices compute "$AUDIO_DIR" \
    --output "$OUTPUT_DIR/all_indices.csv" \
    --format csv \
    --n-fft 2048 \
    --aci-min-freq 500 \
    --aci-max-freq 10000

# Step 2: Demonstrate individual index commands on a sample file
# Find the first audio file in the directory
SAMPLE_FILE=$(find "$AUDIO_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | head -1)

if [ -z "$SAMPLE_FILE" ]; then
    echo "No audio files found for individual index demonstration"
else
    echo ""
    echo "Step 2: Demonstrating individual index commands on: $(basename "$SAMPLE_FILE")"
    echo ""

    # ACI - Acoustic Complexity Index
    # Higher values indicate more complex acoustic environment (more species activity)
    echo "Step 2a: Computing Acoustic Complexity Index (ACI)..."
    bioamla indices aci "$SAMPLE_FILE" \
        --min-freq 1000 \
        --max-freq 8000 \
        --n-fft 1024

    # ADI - Acoustic Diversity Index
    # Higher values indicate more diverse frequency usage (more species)
    echo ""
    echo "Step 2b: Computing Acoustic Diversity Index (ADI)..."
    bioamla indices adi "$SAMPLE_FILE" \
        --max-freq 10000 \
        --freq-step 1000 \
        --db-threshold -50

    # AEI - Acoustic Evenness Index
    # Lower values indicate more even distribution (balanced ecosystem)
    echo ""
    echo "Step 2c: Computing Acoustic Evenness Index (AEI)..."
    bioamla indices aei "$SAMPLE_FILE" \
        --max-freq 10000 \
        --freq-step 1000 \
        --db-threshold -50

    # BIO - Bioacoustic Index
    # Higher values indicate more biophonic activity
    echo ""
    echo "Step 2d: Computing Bioacoustic Index (BIO)..."
    bioamla indices bio "$SAMPLE_FILE" \
        --min-freq 2000 \
        --max-freq 8000

    # NDSI - Normalized Difference Soundscape Index
    # Values closer to 1 indicate natural soundscape, closer to -1 indicate anthropogenic
    echo ""
    echo "Step 2e: Computing NDSI..."
    bioamla indices ndsi "$SAMPLE_FILE" \
        --anthro-min 1000 \
        --anthro-max 2000 \
        --bio-min 2000 \
        --bio-max 8000

    # Entropy metrics
    echo ""
    echo "Step 2f: Computing entropy metrics..."
    bioamla indices entropy "$SAMPLE_FILE" \
        --spectral \
        --temporal
fi

# Step 3: Temporal analysis - track indices over time within a recording
if [ -n "$SAMPLE_FILE" ]; then
    echo ""
    echo "Step 3: Computing temporal variation of indices on sample file..."
    bioamla indices temporal "$SAMPLE_FILE" \
        --window 10.0 \
        --hop 5.0 \
        --output "$OUTPUT_DIR/temporal_indices.csv" \
        --format csv
fi

echo ""
echo "=== Acoustic Indices Analysis Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Output files:"
echo "  - all_indices.csv: All indices for all files"
echo "  - temporal_indices.csv: Time-windowed indices"
echo ""
echo "Interpretation guide:"
echo "  ACI: Higher = more acoustic complexity (species activity)"
echo "  ADI: Higher = more acoustic diversity (species richness)"
echo "  AEI: Lower = more even frequency distribution"
echo "  BIO: Higher = more biophonic activity"
echo "  NDSI: +1 = natural, -1 = anthropogenic dominated"
