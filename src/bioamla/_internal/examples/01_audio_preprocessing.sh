#!/bin/bash
# =============================================================================
# Audio Preprocessing Workflow
# =============================================================================
# PURPOSE: Prepare raw field recordings for analysis by cleaning, normalizing,
#          and segmenting audio files.
#
# FEATURES DEMONSTRATED:
#   - Audio format conversion (batch-convert)
#   - Resampling for consistent sample rates
#   - Audio normalization
#   - Fixed-duration segmentation with overlap
#
# INPUT: Directory of raw audio recordings (any supported format)
# OUTPUT: Cleaned, normalized, and segmented audio files ready for analysis
# =============================================================================

set -e  # Exit on error

# Configuration

AUDIO_DIR="./scp_small"
OUTPUT_DIR="./processed_audio"
SAMPLE_RATE=22050
TARGET_DB=-20

bioamla dataset download "https://www.bioamla.org/datasets/scp_small.zip" .
bioamla dataset unzip ./scp_small.zip .

echo "=== Audio Preprocessing Workflow ==="
echo "Input directory: $AUDIO_DIR"
echo ""

# Check if input directory exists and has audio files
if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Input directory '$AUDIO_DIR' does not exist."
    echo "Usage: $0 [AUDIO_DIRectory]"
    echo "Create the directory and add audio files, or specify a different path."
    exit 1
fi

# Count audio files
AUDIO_COUNT=$(find "$AUDIO_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.ogg" -o -name "*.m4a" \) 2>/dev/null | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo "Error: No audio files found in '$AUDIO_DIR'."
    echo "Supported formats: wav, mp3, flac, ogg, m4a"
    echo ""
    echo "Add some audio files and try again, or specify a different directory:"
    echo "  $0 /path/to/audio/files"
    exit 1
fi

echo "Found $AUDIO_COUNT audio file(s) to process."
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/converted"
mkdir -p "$OUTPUT_DIR/normalized"
mkdir -p "$OUTPUT_DIR/segments"

# Step 1: Batch convert all audio files to WAV format at target sample rate
echo "Step 1: Converting audio files to WAV format at ${SAMPLE_RATE}Hz..."
bioamla audio batch-convert "$AUDIO_DIR" "$OUTPUT_DIR/converted" \
    --format wav \
    --sample-rate $SAMPLE_RATE

# Step 2: Normalize audio levels
echo ""
echo "Step 2: Normalizing audio to ${TARGET_DB}dB..."
shopt -s globstar nullglob
for file in "$OUTPUT_DIR/converted"/**/*.wav; do
    if [ -f "$file" ]; then
        # Get relative path from converted dir
        rel_path="${file#$OUTPUT_DIR/converted/}"
        output_file="$OUTPUT_DIR/normalized/$rel_path"
        mkdir -p "$(dirname "$output_file")"
        echo "  Normalizing: $rel_path"
        bioamla audio normalize "$file" "$output_file" --target-db $TARGET_DB
    fi
done

# Step 3: Segment audio into fixed-duration clips
echo ""
echo "Step 3: Segmenting audio into 3-second clips..."
for file in "$OUTPUT_DIR/normalized"/**/*.wav; do
    if [ -f "$file" ]; then
        # Get relative path and create segment directory
        rel_path="${file#$OUTPUT_DIR/normalized/}"
        base_name="${rel_path%.wav}"
        segment_dir="$OUTPUT_DIR/segments/$base_name"
        mkdir -p "$segment_dir"
        echo "  Segmenting: $rel_path"
        bioamla audio segment "$file" "$segment_dir" --duration 3.0 --overlap 0.5
    fi
done

# Step 4: Display info for a sample of processed files
echo ""
echo "Step 4: Displaying info for sample processed files..."
count=0
for file in "$OUTPUT_DIR/normalized"/**/*.wav; do
    if [ -f "$file" ] && [ $count -lt 3 ]; then
        bioamla audio info "$file"
        count=$((count + 1))
    fi
done

echo ""
echo "=== Preprocessing Complete ==="
echo "Converted files: $OUTPUT_DIR/converted"
echo "Normalized files: $OUTPUT_DIR/normalized"
echo "Segmented files: $OUTPUT_DIR/segments"
