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
PROJECT_DIR="frog_acoustic_study"
INPUT_DIR="${PROJECT_DIR}/raw_recordings"
OUTPUT_DIR="${PROJECT_DIR}/processed_audio"
SAMPLE_RATE=22050
TARGET_DB=-20

echo "=== Audio Preprocessing Workflow ==="
echo "Input directory: $INPUT_DIR"
echo ""

# Check if input directory exists and has audio files
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    echo "Usage: $0 [input_directory]"
    echo "Create the directory and add audio files, or specify a different path."
    exit 1
fi

# Count audio files
AUDIO_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "Error: No audio files found in '$INPUT_DIR'."
    echo "Supported formats: wav, mp3, flac, ogg, m4a"
    echo ""
    echo "Add some audio files and try again, or specify a different directory:"
    echo "  $0 /path/to/audio/files"
    exit 1
fi

echo "Found $AUDIO_COUNT audio file(s) to process."
echo ""

# Step 1: Convert all audio files to WAV format
echo "Step 1: Converting audio files to WAV format..."
bioamla audio convert "$INPUT_DIR" wav --output "$OUTPUT_DIR/converted" --batch

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
    --bandpass "500-10000" \
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
    --batch \
    --silence-threshold -40 \
    --min-silence 0.3 \
    --min-segment 0.5

# Step 7: Analyze the processed files
echo ""
echo "Step 7: Analyzing processed audio files..."
bioamla audio analyze "$OUTPUT_DIR/segments" \
    --batch \
    --output "$OUTPUT_DIR/analysis_report.csv" \
    --format csv

echo ""
echo "=== Preprocessing Complete ==="
echo "Processed files are in: $OUTPUT_DIR/segments"
echo "Analysis report: $OUTPUT_DIR/analysis_report.csv"
