#!/bin/bash
# =============================================================================
# Audio Preprocessing Workflow
# =============================================================================
# PURPOSE: Prepare raw field recordings for analysis by cleaning, normalizing,
#          and segmenting audio files.
#
# FEATURES DEMONSTRATED:
#   - Audio format conversion
#   - Bandpass filtering to isolate frequency ranges
#   - Noise reduction
#   - Audio normalization
#   - Silence-based segmentation
#   - Resampling for consistent sample rates
#
# INPUT: Directory of raw audio recordings (any supported format)
# OUTPUT: Cleaned, normalized, and segmented audio files ready for analysis
# =============================================================================

set -e  # Exit on error

# Configuration
INPUT_DIR="./raw_recordings"
OUTPUT_DIR="./processed_audio"
SAMPLE_RATE=22050
TARGET_DB=-20

echo "=== Audio Preprocessing Workflow ==="
echo ""

# Step 1: Convert all audio files to WAV format
echo "Step 1: Converting audio files to WAV format..."
bioamla audio convert "$INPUT_DIR" wav --output "$OUTPUT_DIR/converted"

# Step 2: Resample to consistent sample rate
echo ""
echo "Step 2: Resampling audio to ${SAMPLE_RATE}Hz..."
bioamla audio resample "$OUTPUT_DIR/converted" \
    --output "$OUTPUT_DIR/resampled" \
    --rate $SAMPLE_RATE \
    --batch

# Step 3: Apply bandpass filter to isolate biophony frequencies (500Hz - 10kHz)
echo ""
echo "Step 3: Applying bandpass filter (500Hz - 10kHz)..."
bioamla audio filter "$OUTPUT_DIR/resampled" \
    --output "$OUTPUT_DIR/filtered" \
    --bandpass \
    --low 500 \
    --high 10000 \
    --batch

# Step 4: Denoise using spectral subtraction
echo ""
echo "Step 4: Applying spectral denoising..."
bioamla audio denoise "$OUTPUT_DIR/filtered" \
    --output "$OUTPUT_DIR/denoised" \
    --method spectral \
    --strength 0.5 \
    --batch

# Step 5: Normalize audio levels
echo ""
echo "Step 5: Normalizing audio to ${TARGET_DB}dB..."
bioamla audio normalize "$OUTPUT_DIR/denoised" \
    --output "$OUTPUT_DIR/normalized" \
    --target-db $TARGET_DB \
    --batch

# Step 6: Segment audio on silence to extract vocalization events
echo ""
echo "Step 6: Segmenting audio on silence..."
bioamla audio segment "$OUTPUT_DIR/normalized" \
    --output "$OUTPUT_DIR/segments" \
    --silence-threshold -40 \
    --min-silence 0.3 \
    --min-segment 0.5

# Step 7: Analyze the processed files
echo ""
echo "Step 7: Analyzing processed audio files..."
bioamla audio analyze "$OUTPUT_DIR/segments" \
    --batch \
    --output "$OUTPUT_DIR/analysis_report.csv" \
    --output-format csv

echo ""
echo "=== Preprocessing Complete ==="
echo "Processed files are in: $OUTPUT_DIR/segments"
echo "Analysis report: $OUTPUT_DIR/analysis_report.csv"
