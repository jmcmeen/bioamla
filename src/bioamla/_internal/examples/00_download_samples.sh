#!/bin/bash
# =============================================================================
# Audio Preprocessing Workflow
# =============================================================================
# PURPOSE: Prepare raw field recordings for analysis by cleaning, normalizing,
#          and segmenting audio files.
#
# FEATURES DEMONSTRATED:
#   - Audio format conversion (batch-convert)
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
