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

mkdir -p "$OUTPUT_DIR"

# Step 1: Compute all acoustic indices for the dataset
echo "Step 1: Computing all acoustic indices..."
bioamla indices compute "$AUDIO_DIR" \
    --output "$OUTPUT_DIR/all_indices.csv" \
    --output-format csv \
    --n-fft 2048 \
    --aci-min-freq 500 \
    --aci-max-freq 10000

# Step 2: Compute individual indices with custom parameters

# ACI - Acoustic Complexity Index
# Higher values indicate more complex acoustic environment (more species activity)
echo ""
echo "Step 2a: Computing Acoustic Complexity Index (ACI)..."
bioamla indices aci "$AUDIO_DIR" \
    --min-freq 1000 \
    --max-freq 8000 \
    --n-fft 1024

# ADI - Acoustic Diversity Index
# Higher values indicate more diverse frequency usage (more species)
echo ""
echo "Step 2b: Computing Acoustic Diversity Index (ADI)..."
bioamla indices adi "$AUDIO_DIR" \
    --max-freq 10000 \
    --freq-step 1000 \
    --db-threshold -50

# AEI - Acoustic Evenness Index
# Lower values indicate more even distribution (balanced ecosystem)
echo ""
echo "Step 2c: Computing Acoustic Evenness Index (AEI)..."
bioamla indices aei "$AUDIO_DIR" \
    --max-freq 10000 \
    --freq-step 1000 \
    --db-threshold -50

# BIO - Bioacoustic Index
# Higher values indicate more biophonic activity
echo ""
echo "Step 2d: Computing Bioacoustic Index (BIO)..."
bioamla indices bio "$AUDIO_DIR" \
    --min-freq 2000 \
    --max-freq 8000

# NDSI - Normalized Difference Soundscape Index
# Values closer to 1 indicate natural soundscape, closer to -1 indicate anthropogenic
echo ""
echo "Step 2e: Computing NDSI..."
bioamla indices ndsi "$AUDIO_DIR" \
    --anthro-min 1000 \
    --anthro-max 2000 \
    --bio-min 2000 \
    --bio-max 8000

# Step 3: Compute entropy metrics
echo ""
echo "Step 3: Computing entropy metrics..."
bioamla indices entropy "$AUDIO_DIR" \
    --spectral \
    --temporal

# Step 4: Temporal analysis - track indices over time within recordings
echo ""
echo "Step 4: Computing temporal variation of indices..."
bioamla indices temporal "$AUDIO_DIR" \
    --window 10.0 \
    --hop 5.0 \
    --output "$OUTPUT_DIR/temporal_indices.csv" \
    --output-format csv

echo ""
echo "=== Acoustic Indices Analysis Complete ==="
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Interpretation guide:"
echo "  ACI: Higher = more acoustic complexity (species activity)"
echo "  ADI: Higher = more acoustic diversity (species richness)"
echo "  AEI: Lower = more even frequency distribution"
echo "  BIO: Higher = more biophonic activity"
echo "  NDSI: +1 = natural, -1 = anthropogenic dominated"
